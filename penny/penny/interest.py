"""Interest score computation for user-entity relationships."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import UTC, datetime

from penny.constants import PennyConstants
from penny.database.models import Engagement, Entity, Fact
from penny.ollama.embeddings import deserialize_embedding, find_similar


def _recency_weight(
    created_at: datetime,
    now: datetime | None = None,
    *,
    half_life_days: float,
) -> float:
    """Compute exponential recency decay weight for an engagement.

    Uses a half-life decay: weight = 2^(-age_days / half_life_days).
    A engagement at age 0 has weight 1.0.
    A engagement at age = half_life has weight 0.5.

    Args:
        created_at: When the engagement was recorded
        now: Current time (defaults to UTC now; injectable for testing)

    Returns:
        Decay weight between 0.0 and 1.0
    """
    if now is None:
        now = datetime.now(UTC)
    # SQLite returns naive datetimes; treat them as UTC
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    age_days = (now - created_at).total_seconds() / 86400.0
    if age_days < 0:
        age_days = 0.0
    return math.pow(2.0, -age_days / half_life_days)


def _valence_sign(valence: str) -> float:
    """Convert valence to a numeric sign multiplier.

    Args:
        valence: EngagementValence enum value

    Returns:
        +1.0 for positive, -1.0 for negative, 0.0 for neutral
    """
    if valence == PennyConstants.EngagementValence.POSITIVE:
        return 1.0
    if valence == PennyConstants.EngagementValence.NEGATIVE:
        return -1.0
    return 0.0


def compute_interest_score(
    engagements: list[Engagement],
    now: datetime | None = None,
    *,
    half_life_days: float,
) -> float:
    """Compute interest score from a list of engagements.

    Formula: sum(valence_sign * strength * recency_decay(created_at))

    This is a pure function that takes pre-fetched engagements. The caller
    is responsible for fetching engagements from the database (e.g., via
    db.get_entity_engagements()).

    Args:
        engagements: List of Engagement objects for a single entity/user pair
        now: Current time (defaults to UTC now; injectable for testing)

    Returns:
        Interest score. Positive means interest, negative means disinterest.
        Returns 0.0 for empty engagement list.
    """
    if not engagements:
        return 0.0

    score = 0.0
    for engagement in engagements:
        sign = _valence_sign(engagement.valence)
        decay = _recency_weight(engagement.created_at, now=now, half_life_days=half_life_days)
        score += sign * engagement.strength * decay

    return score


def compute_notification_interest(
    engagements: list[Engagement],
    now: datetime | None = None,
    *,
    half_life_days: float,
) -> float:
    """Compute interest score using only notification-relevant engagement types.

    Filters to NOTIFICATION_ENGAGEMENT_TYPES (excludes USER_SEARCH and
    SEARCH_DISCOVERY), then delegates to compute_interest_score.
    """
    notification_engs = [
        e for e in engagements if e.engagement_type in PennyConstants.NOTIFICATION_ENGAGEMENT_TYPES
    ]
    return compute_interest_score(notification_engs, now=now, half_life_days=half_life_days)


def compute_notification_score(
    engagements: list[Engagement],
    unnotified_fact_count: int,
    notified_fact_count: int,
    *,
    half_life_days: float,
) -> float:
    """Full notification priority: interest * log2(facts + 1) / fatigue.

    Multiplicative: zero interest means zero score, so discovery-only
    entities (no explicit user engagement) are never surfaced.
    """
    interest = compute_notification_interest(engagements, half_life_days=half_life_days)
    if interest == 0.0:
        return 0.0
    fatigue = math.log2(notified_fact_count + 2)
    return interest * math.log2(unnotified_fact_count + 1) / fatigue


def compute_neighbor_boost(
    eid: int,
    entity: Entity,
    all_interest: dict[int, float],
    embedding_candidates: list[tuple[int, list[float]]],
    *,
    neighbor_k: int,
    neighbor_min_similarity: float,
) -> float:
    """Mean neighbor interest weighted by embedding similarity."""
    if entity.embedding is None or not embedding_candidates:
        return 0.0

    query_vec = deserialize_embedding(entity.embedding)
    neighbors = find_similar(
        query_vec, embedding_candidates, top_k=neighbor_k + 1, threshold=neighbor_min_similarity
    )
    engaged = [
        (nid, sim) for nid, sim in neighbors if nid != eid and all_interest.get(nid, 0.0) != 0.0
    ]
    if not engaged:
        return 0.0

    weighted = [all_interest[nid] * sim for nid, sim in engaged]
    return sum(weighted) / len(weighted)


def scored_entities_for_user(
    entities: list[Entity],
    engagements: list[Engagement],
    facts_by_entity: dict[int, list[Fact]],
    notified_counts: dict[int, int],
    embedding_candidates: list[tuple[int, list[float]]],
    *,
    half_life_days: float,
    neighbor_k: int,
    neighbor_min_similarity: float,
    neighbor_factor: float,
) -> list[tuple[float, Entity]]:
    """Score entities by full notification priority, sorted by absolute score descending."""
    engagements_by_entity: dict[int, list[Engagement]] = defaultdict(list)
    for eng in engagements:
        if eng.entity_id is not None:
            engagements_by_entity[eng.entity_id].append(eng)

    all_interest = _build_interest_map(engagements_by_entity, half_life_days=half_life_days)

    scored: list[tuple[float, Entity]] = []
    for entity in entities:
        assert entity.id is not None
        facts = facts_by_entity.get(entity.id, [])
        unnotified = sum(1 for f in facts if f.notified_at is None)
        notified = notified_counts.get(entity.id, 0)
        base = compute_notification_score(
            engagements_by_entity.get(entity.id, []),
            unnotified,
            notified,
            half_life_days=half_life_days,
        )
        boost = compute_neighbor_boost(
            entity.id,
            entity,
            all_interest,
            embedding_candidates,
            neighbor_k=neighbor_k,
            neighbor_min_similarity=neighbor_min_similarity,
        )
        score = base * (1.0 + neighbor_factor * boost)
        scored.append((score, entity))

    scored.sort(key=lambda x: abs(x[0]), reverse=True)
    return scored


def _build_interest_map(
    engagements_by_entity: dict[int, list[Engagement]],
    *,
    half_life_days: float,
) -> dict[int, float]:
    """Precompute notification interest score for every entity with engagements."""
    return {
        eid: compute_notification_interest(engs, half_life_days=half_life_days)
        for eid, engs in engagements_by_entity.items()
    }

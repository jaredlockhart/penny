"""Interest score computation for user-entity relationships."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import UTC, datetime

from penny.constants import PennyConstants
from penny.database.models import Engagement, Entity, Fact
from penny.ollama.embeddings import cosine_similarity, deserialize_embedding


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


# --- Loyalty + novelty scoring ---


def compute_loyalty_score(
    engagements: list[Engagement],
    fact_count: int,
    now: datetime | None = None,
    *,
    half_life_days: float,
) -> float:
    """Day-based loyalty score: distinct positive engagement days * strength * recency.

    Rewards entities the user engages with across multiple distinct days.
    Search days count at half weight. Negative notification-type days
    subtract at half weight.
    """
    if now is None:
        now = datetime.now(UTC)

    pos_notif, pos_search, neg_notif = _split_engagements(engagements)

    all_pos = pos_notif + pos_search
    if not all_pos:
        if neg_notif:
            return _negative_only_score(neg_notif, now, half_life_days=half_life_days)
        return _loyalty_baseline(engagements, fact_count)

    distinct_days = _count_loyalty_days(pos_notif, pos_search)
    avg_strength = _avg_positive_strength(pos_notif)
    recency = _most_recent_recency(all_pos, now, half_life_days=half_life_days)
    neg_days = len({_eng_date(e, now) for e in neg_notif})

    return max((distinct_days - 0.5 * neg_days) * avg_strength * recency, -5.0)


def _split_engagements(
    engagements: list[Engagement],
) -> tuple[list[Engagement], list[Engagement], list[Engagement]]:
    """Split into positive notification-type, positive search, and negative notification-type."""
    pos_notif = []
    pos_search = []
    neg_notif = []
    for e in engagements:
        is_positive = e.valence == PennyConstants.EngagementValence.POSITIVE
        is_negative = e.valence == PennyConstants.EngagementValence.NEGATIVE
        is_notif = e.engagement_type in PennyConstants.NOTIFICATION_ENGAGEMENT_TYPES
        is_search = e.engagement_type == PennyConstants.EngagementType.USER_SEARCH

        if is_positive and is_notif:
            pos_notif.append(e)
        elif is_positive and is_search:
            pos_search.append(e)
        elif is_negative and is_notif:
            neg_notif.append(e)
    return pos_notif, pos_search, neg_notif


def _negative_only_score(
    neg_notif: list[Engagement],
    now: datetime,
    *,
    half_life_days: float,
) -> float:
    """Score for entities with only negative engagement (e.g., downvoted)."""
    neg_days = len({_eng_date(e, now) for e in neg_notif})
    recency = _most_recent_recency(neg_notif, now, half_life_days=half_life_days)
    return max(-0.5 * neg_days * recency, -5.0)


def _loyalty_baseline(engagements: list[Engagement], fact_count: int) -> float:
    """Tiny baseline for searched entities with facts but no positive notification engagement."""
    has_search = any(
        e.engagement_type == PennyConstants.EngagementType.USER_SEARCH for e in engagements
    )
    if has_search and fact_count > 0:
        return 0.01 * math.log2(fact_count + 1)
    return 0.0


def _count_loyalty_days(pos_notif: list[Engagement], pos_search: list[Engagement]) -> float:
    """Count distinct positive days: notification days full weight, search-only days half."""
    notif_dates = {_eng_date_naive(e) for e in pos_notif}
    search_dates = {_eng_date_naive(e) for e in pos_search} - notif_dates
    return len(notif_dates) + 0.5 * len(search_dates)


def _avg_positive_strength(pos_notif: list[Engagement]) -> float:
    """Average strength of positive notification engagements, or default 0.3."""
    if not pos_notif:
        return 0.3
    return sum(e.strength for e in pos_notif) / len(pos_notif)


def _most_recent_recency(
    engagements: list[Engagement],
    now: datetime,
    *,
    half_life_days: float,
) -> float:
    """Recency decay based on the most recent engagement."""
    most_recent = max(engagements, key=lambda e: e.created_at)
    return _recency_weight(most_recent.created_at, now=now, half_life_days=half_life_days)


def _eng_date(e: Engagement, now: datetime) -> str:
    """Engagement date as string key, handling naive datetimes."""
    dt = e.created_at
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.date().isoformat()


def _eng_date_naive(e: Engagement) -> str:
    """Engagement date as string key (no now needed)."""
    dt = e.created_at
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.date().isoformat()


def compute_novelty_bonus(
    entity: Entity,
    loyal_embeddings: list[tuple[int, list[float]]],
    *,
    half_life_days: float,
    now: datetime | None = None,
) -> float:
    """Novelty bonus: max similarity to loyal entities * entity freshness.

    Surfaces newly discovered entities that are semantically close to
    the user's established interests.
    """
    if entity.embedding is None or not loyal_embeddings:
        return 0.0
    if now is None:
        now = datetime.now(UTC)

    similarity = _best_loyal_similarity(entity, loyal_embeddings)
    freshness = _entity_freshness(entity, now, half_life_days=half_life_days)
    return similarity * freshness


def _best_loyal_similarity(
    entity: Entity,
    loyal_embeddings: list[tuple[int, list[float]]],
) -> float:
    """Max cosine similarity between entity and any loyal entity."""
    assert entity.embedding is not None
    query = deserialize_embedding(entity.embedding)
    best = 0.0
    for lid, loyal_vec in loyal_embeddings:
        if lid == entity.id:
            continue
        sim = cosine_similarity(query, loyal_vec)
        best = max(best, sim)
    return best


def _entity_freshness(
    entity: Entity,
    now: datetime,
    *,
    half_life_days: float,
) -> float:
    """Entity freshness based on creation age."""
    created = entity.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    age_days = (now - created).total_seconds() / 86400.0
    if age_days < 0:
        age_days = 0.0
    return math.pow(2.0, -age_days / half_life_days)


# --- Main scoring entry point ---


def scored_entities_for_user(
    entities: list[Entity],
    engagements: list[Engagement],
    facts_by_entity: dict[int, list[Fact]],
    embedding_candidates: list[tuple[int, list[float]]],
    *,
    half_life_days: float,
    novelty_weight: float,
    loyal_pool_size: int,
) -> list[tuple[float, Entity]]:
    """Score entities by loyalty + novelty, sorted by score descending.

    Loyalty rewards entities engaged with across multiple distinct days.
    Novelty surfaces fresh entities semantically near loyal ones.
    """
    engagements_by_entity = _group_engagements(engagements)
    # Compute loyalty for ALL engaged entities (not just scored ones) so the
    # loyal pool includes highly-engaged entities that may not have unnotified facts.
    all_loyalty = _compute_loyalty_from_engagements(
        engagements_by_entity, half_life_days=half_life_days
    )
    loyal_embeddings = _build_loyal_embeddings(
        all_loyalty, embedding_candidates, pool_size=loyal_pool_size
    )
    novelty_half_life = half_life_days / 2.0

    scored: list[tuple[float, Entity]] = []
    for entity in entities:
        assert entity.id is not None
        loyalty = all_loyalty.get(entity.id, 0.0)
        facts = facts_by_entity.get(entity.id, [])
        has_facts = len(facts) > 0

        novelty = 0.0
        if has_facts:
            novelty = compute_novelty_bonus(
                entity, loyal_embeddings, half_life_days=novelty_half_life
            )

        score = loyalty + novelty_weight * novelty
        scored.append((score, entity))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _group_engagements(
    engagements: list[Engagement],
) -> dict[int, list[Engagement]]:
    """Group engagements by entity ID."""
    by_entity: dict[int, list[Engagement]] = defaultdict(list)
    for eng in engagements:
        if eng.entity_id is not None:
            by_entity[eng.entity_id].append(eng)
    return by_entity


def _compute_loyalty_from_engagements(
    engagements_by_entity: dict[int, list[Engagement]],
    *,
    half_life_days: float,
) -> dict[int, float]:
    """Compute loyalty score for every entity with engagements."""
    return {
        entity_id: compute_loyalty_score(engs, 0, half_life_days=half_life_days)
        for entity_id, engs in engagements_by_entity.items()
    }


def _build_loyal_embeddings(
    loyalty_scores: dict[int, float],
    embedding_candidates: list[tuple[int, list[float]]],
    *,
    pool_size: int,
) -> list[tuple[int, list[float]]]:
    """Build embedding list for the top-N loyal entities."""
    top_loyal = sorted(loyalty_scores, key=lambda eid: loyalty_scores[eid], reverse=True)[
        :pool_size
    ]
    top_set = set(top_loyal)
    return [(eid, vec) for eid, vec in embedding_candidates if eid in top_set]

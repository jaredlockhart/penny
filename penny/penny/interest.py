"""Interest score computation for user-entity relationships."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import UTC, datetime

from penny.constants import PennyConstants
from penny.database.models import Engagement, Entity


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


def scored_entities_for_user(
    entities: list[Entity],
    engagements: list[Engagement],
    *,
    half_life_days: float,
) -> list[tuple[float, Entity]]:
    """Score entities by notification interest, sorted by absolute score descending."""
    engagements_by_entity: dict[int, list[Engagement]] = defaultdict(list)
    for eng in engagements:
        if eng.entity_id is not None:
            engagements_by_entity[eng.entity_id].append(eng)

    scored: list[tuple[float, Entity]] = []
    for entity in entities:
        assert entity.id is not None
        score = compute_notification_interest(
            engagements_by_entity.get(entity.id, []),
            half_life_days=half_life_days,
        )
        scored.append((score, entity))

    scored.sort(key=lambda x: abs(x[0]), reverse=True)
    return scored

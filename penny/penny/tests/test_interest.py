"""Tests for interest score computation."""

from datetime import UTC, datetime, timedelta

import pytest

from penny.config_params import RUNTIME_CONFIG_PARAMS
from penny.constants import PennyConstants
from penny.database.models import Engagement
from penny.interest import (
    _recency_weight,
    _valence_sign,
    compute_interest_score,
    compute_loyalty_score,
    compute_notification_interest,
)

_HALF_LIFE = RUNTIME_CONFIG_PARAMS["INTEREST_SCORE_HALF_LIFE_DAYS"].default

_DEFAULT_USER = "+1234"
_DEFAULT_TYPE = PennyConstants.EngagementType.MESSAGE_MENTION
_DEFAULT_VALENCE = PennyConstants.EngagementValence.POSITIVE
_DEFAULT_STRENGTH = 0.5


class TestRecencyWeight:
    """Tests for exponential recency decay."""

    def test_zero_age_returns_one(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        assert _recency_weight(now, now=now, half_life_days=_HALF_LIFE) == pytest.approx(1.0)

    def test_half_life_returns_half(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        half_life_ago = now - timedelta(days=_HALF_LIFE)
        assert _recency_weight(half_life_ago, now=now, half_life_days=_HALF_LIFE) == pytest.approx(
            0.5
        )

    def test_two_half_lives_returns_quarter(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        two_half_lives = now - timedelta(days=_HALF_LIFE * 2)
        assert _recency_weight(two_half_lives, now=now, half_life_days=_HALF_LIFE) == pytest.approx(
            0.25
        )

    def test_future_engagement_clamped_to_one(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        future = now + timedelta(days=1)
        assert _recency_weight(future, now=now, half_life_days=_HALF_LIFE) == pytest.approx(1.0)

    def test_very_old_engagement_approaches_zero(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        ancient = now - timedelta(days=365)
        weight = _recency_weight(ancient, now=now, half_life_days=_HALF_LIFE)
        assert weight < 0.001


class TestValenceSign:
    """Tests for valence to sign conversion."""

    def test_positive(self):
        assert _valence_sign(PennyConstants.EngagementValence.POSITIVE) == 1.0

    def test_negative(self):
        assert _valence_sign(PennyConstants.EngagementValence.NEGATIVE) == -1.0

    def test_neutral(self):
        assert _valence_sign(PennyConstants.EngagementValence.NEUTRAL) == 0.0


class TestComputeInterestScore:
    """Tests for the overall interest score computation."""

    def test_empty_engagements_returns_zero(self):
        assert compute_interest_score([], half_life_days=_HALF_LIFE) == 0.0

    def test_single_positive_recent_engagement(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        engagement = Engagement(
            user=_DEFAULT_USER,
            engagement_type=_DEFAULT_TYPE,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.7,
            created_at=now,
        )
        score = compute_interest_score([engagement], now=now, half_life_days=_HALF_LIFE)
        assert score == pytest.approx(0.7)

    def test_single_negative_engagement(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        engagement = Engagement(
            user=_DEFAULT_USER,
            engagement_type=_DEFAULT_TYPE,
            valence=PennyConstants.EngagementValence.NEGATIVE,
            strength=0.8,
            created_at=now,
        )
        score = compute_interest_score([engagement], now=now, half_life_days=_HALF_LIFE)
        assert score == pytest.approx(-0.8)

    def test_neutral_engagement_contributes_nothing(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        engagement = Engagement(
            user=_DEFAULT_USER,
            engagement_type=_DEFAULT_TYPE,
            valence=PennyConstants.EngagementValence.NEUTRAL,
            strength=1.0,
            created_at=now,
        )
        assert compute_interest_score([engagement], now=now, half_life_days=_HALF_LIFE) == 0.0

    def test_recency_reduces_contribution(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        old_engagement = Engagement(
            user=_DEFAULT_USER,
            engagement_type=_DEFAULT_TYPE,
            valence=_DEFAULT_VALENCE,
            strength=1.0,
            created_at=now - timedelta(days=_HALF_LIFE),
        )
        score = compute_interest_score([old_engagement], now=now, half_life_days=_HALF_LIFE)
        assert score == pytest.approx(0.5)

    def test_mixed_engagements_can_cancel(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        like = Engagement(
            user=_DEFAULT_USER,
            engagement_type=_DEFAULT_TYPE,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.5,
            created_at=now,
        )
        dislike = Engagement(
            user=_DEFAULT_USER,
            engagement_type=_DEFAULT_TYPE,
            valence=PennyConstants.EngagementValence.NEGATIVE,
            strength=0.5,
            created_at=now,
        )
        score = compute_interest_score([like, dislike], now=now, half_life_days=_HALF_LIFE)
        assert score == pytest.approx(0.0)

    def test_multiple_engagements_accumulate(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        engagements = [
            Engagement(
                user=_DEFAULT_USER,
                engagement_type=_DEFAULT_TYPE,
                valence=_DEFAULT_VALENCE,
                strength=0.3,
                created_at=now,
            ),
            Engagement(
                user=_DEFAULT_USER,
                engagement_type=_DEFAULT_TYPE,
                valence=_DEFAULT_VALENCE,
                strength=0.2,
                created_at=now,
            ),
        ]
        score = compute_interest_score(engagements, now=now, half_life_days=_HALF_LIFE)
        assert score == pytest.approx(0.5)

    def test_negative_score_possible(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        engagements = [
            Engagement(
                user=_DEFAULT_USER,
                engagement_type=_DEFAULT_TYPE,
                valence=PennyConstants.EngagementValence.NEGATIVE,
                strength=0.8,
                created_at=now,
            ),
            Engagement(
                user=_DEFAULT_USER,
                engagement_type=_DEFAULT_TYPE,
                valence=PennyConstants.EngagementValence.POSITIVE,
                strength=0.2,
                created_at=now,
            ),
        ]
        score = compute_interest_score(engagements, now=now, half_life_days=_HALF_LIFE)
        assert score == pytest.approx(-0.6)


class TestComputeNotificationInterest:
    """Tests for notification-filtered interest scoring."""

    def test_filters_out_user_search(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        eng = Engagement(
            user=_DEFAULT_USER,
            engagement_type=PennyConstants.EngagementType.USER_SEARCH,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=1.0,
            created_at=now,
        )
        assert compute_notification_interest([eng], now=now, half_life_days=_HALF_LIFE) == 0.0

    def test_filters_out_search_discovery(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        eng = Engagement(
            user=_DEFAULT_USER,
            engagement_type=PennyConstants.EngagementType.SEARCH_DISCOVERY,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.5,
            created_at=now,
        )
        assert compute_notification_interest([eng], now=now, half_life_days=_HALF_LIFE) == 0.0

    def test_keeps_notification_types(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        eng = Engagement(
            user=_DEFAULT_USER,
            engagement_type=PennyConstants.EngagementType.MESSAGE_MENTION,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.5,
            created_at=now,
        )
        score = compute_notification_interest([eng], now=now, half_life_days=_HALF_LIFE)
        assert score == pytest.approx(0.5)

    def test_mixed_types_filters_correctly(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        search = Engagement(
            user=_DEFAULT_USER,
            engagement_type=PennyConstants.EngagementType.USER_SEARCH,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=1.0,
            created_at=now,
        )
        mention = Engagement(
            user=_DEFAULT_USER,
            engagement_type=PennyConstants.EngagementType.MESSAGE_MENTION,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.3,
            created_at=now,
        )
        score = compute_notification_interest([search, mention], now=now, half_life_days=_HALF_LIFE)
        assert score == pytest.approx(0.3)


class TestComputeLoyaltyScore:
    """Tests for day-based loyalty scoring."""

    def test_no_engagements_returns_zero(self):
        assert compute_loyalty_score([], fact_count=5, half_life_days=_HALF_LIFE) == 0.0

    def test_search_only_counts_at_half_weight_day(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        eng = Engagement(
            user=_DEFAULT_USER,
            engagement_type=PennyConstants.EngagementType.USER_SEARCH,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=1.0,
            created_at=now,
        )
        score = compute_loyalty_score([eng], fact_count=10, now=now, half_life_days=_HALF_LIFE)
        # 0.5 day (search at half weight) * 0.3 (default strength) * 1.0 (recency) = 0.15
        assert score == pytest.approx(0.15)

    def test_no_positive_no_negative_no_search_returns_zero(self):
        assert compute_loyalty_score([], fact_count=5, half_life_days=_HALF_LIFE) == 0.0

    def test_single_positive_day(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        eng = Engagement(
            user=_DEFAULT_USER,
            engagement_type=PennyConstants.EngagementType.EMOJI_REACTION,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.5,
            created_at=now,
        )
        score = compute_loyalty_score([eng], fact_count=5, now=now, half_life_days=_HALF_LIFE)
        # 1 day * 0.5 strength * 1.0 recency = 0.5
        assert score == pytest.approx(0.5)

    def test_multi_day_engagement_scores_higher(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        day1 = Engagement(
            user=_DEFAULT_USER,
            engagement_type=PennyConstants.EngagementType.EMOJI_REACTION,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.5,
            created_at=now,
        )
        day2 = Engagement(
            user=_DEFAULT_USER,
            engagement_type=PennyConstants.EngagementType.FOLLOW_UP_QUESTION,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.5,
            created_at=now - timedelta(days=1),
        )
        single = compute_loyalty_score([day1], fact_count=5, now=now, half_life_days=_HALF_LIFE)
        multi = compute_loyalty_score(
            [day1, day2], fact_count=5, now=now, half_life_days=_HALF_LIFE
        )
        assert multi > single

    def test_search_days_count_at_half_weight(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        mention = Engagement(
            user=_DEFAULT_USER,
            engagement_type=PennyConstants.EngagementType.MESSAGE_MENTION,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.5,
            created_at=now,
        )
        search = Engagement(
            user=_DEFAULT_USER,
            engagement_type=PennyConstants.EngagementType.USER_SEARCH,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=1.0,
            created_at=now - timedelta(days=1),
        )
        mention_only = compute_loyalty_score(
            [mention], fact_count=5, now=now, half_life_days=_HALF_LIFE
        )
        with_search = compute_loyalty_score(
            [mention, search], fact_count=5, now=now, half_life_days=_HALF_LIFE
        )
        # Search on a different day adds 0.5 to distinct_days
        assert with_search > mention_only

    def test_negative_days_reduce_score(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        positive = Engagement(
            user=_DEFAULT_USER,
            engagement_type=PennyConstants.EngagementType.EMOJI_REACTION,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.5,
            created_at=now,
        )
        negative = Engagement(
            user=_DEFAULT_USER,
            engagement_type=PennyConstants.EngagementType.EMOJI_REACTION,
            valence=PennyConstants.EngagementValence.NEGATIVE,
            strength=0.8,
            created_at=now - timedelta(days=1),
        )
        pos_only = compute_loyalty_score(
            [positive], fact_count=5, now=now, half_life_days=_HALF_LIFE
        )
        with_neg = compute_loyalty_score(
            [positive, negative], fact_count=5, now=now, half_life_days=_HALF_LIFE
        )
        assert with_neg < pos_only

    def test_recency_decay_reduces_old_engagement(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        recent = Engagement(
            user=_DEFAULT_USER,
            engagement_type=PennyConstants.EngagementType.EMOJI_REACTION,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.5,
            created_at=now,
        )
        old = Engagement(
            user=_DEFAULT_USER,
            engagement_type=PennyConstants.EngagementType.EMOJI_REACTION,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.5,
            created_at=now - timedelta(days=30),
        )
        recent_score = compute_loyalty_score(
            [recent], fact_count=5, now=now, half_life_days=_HALF_LIFE
        )
        old_score = compute_loyalty_score([old], fact_count=5, now=now, half_life_days=_HALF_LIFE)
        assert recent_score > old_score

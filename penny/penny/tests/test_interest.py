"""Tests for interest score computation."""

from datetime import UTC, datetime, timedelta

import pytest

from penny.constants import PennyConstants
from penny.database.models import Engagement
from penny.interest import _recency_weight, _valence_sign, compute_interest_score

_DEFAULT_USER = "+1234"
_DEFAULT_TYPE = PennyConstants.EngagementType.MESSAGE_MENTION
_DEFAULT_VALENCE = PennyConstants.EngagementValence.POSITIVE
_DEFAULT_STRENGTH = 0.5


class TestRecencyWeight:
    """Tests for exponential recency decay."""

    def test_zero_age_returns_one(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        assert _recency_weight(now, now=now) == pytest.approx(1.0)

    def test_half_life_returns_half(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        half_life_ago = now - timedelta(days=PennyConstants.INTEREST_SCORE_HALF_LIFE_DAYS)
        assert _recency_weight(half_life_ago, now=now) == pytest.approx(0.5)

    def test_two_half_lives_returns_quarter(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        two_half_lives = now - timedelta(days=PennyConstants.INTEREST_SCORE_HALF_LIFE_DAYS * 2)
        assert _recency_weight(two_half_lives, now=now) == pytest.approx(0.25)

    def test_future_engagement_clamped_to_one(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        future = now + timedelta(days=1)
        assert _recency_weight(future, now=now) == pytest.approx(1.0)

    def test_very_old_engagement_approaches_zero(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        ancient = now - timedelta(days=365)
        weight = _recency_weight(ancient, now=now)
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
        assert compute_interest_score([]) == 0.0

    def test_single_positive_recent_engagement(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        engagement = Engagement(
            user=_DEFAULT_USER,
            engagement_type=_DEFAULT_TYPE,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.7,
            created_at=now,
        )
        score = compute_interest_score([engagement], now=now)
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
        score = compute_interest_score([engagement], now=now)
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
        assert compute_interest_score([engagement], now=now) == 0.0

    def test_recency_reduces_contribution(self):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        old_engagement = Engagement(
            user=_DEFAULT_USER,
            engagement_type=_DEFAULT_TYPE,
            valence=_DEFAULT_VALENCE,
            strength=1.0,
            created_at=now - timedelta(days=PennyConstants.INTEREST_SCORE_HALF_LIFE_DAYS),
        )
        score = compute_interest_score([old_engagement], now=now)
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
        score = compute_interest_score([like, dislike], now=now)
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
        score = compute_interest_score(engagements, now=now)
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
        score = compute_interest_score(engagements, now=now)
        assert score == pytest.approx(-0.6)

"""Tests for interest score computation and heat engine."""

from datetime import UTC, datetime, timedelta

import pytest

from penny.config_params import RUNTIME_CONFIG_PARAMS, RuntimeParams
from penny.constants import PennyConstants
from penny.database import Database
from penny.database.models import Engagement, Entity
from penny.interest import (
    HeatEngine,
    _recency_weight,
    _valence_sign,
    compute_interest_score,
    compute_notification_interest,
    scored_entities_for_user,
)
from penny.ollama.embeddings import serialize_embedding

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


class TestScoredEntitiesForUser:
    """Tests for heat-based entity sorting."""

    def test_sorts_by_heat_descending(self):
        entities = [
            Entity(id=1, user=_DEFAULT_USER, name="cold", heat=0.5),
            Entity(id=2, user=_DEFAULT_USER, name="hot", heat=5.0),
            Entity(id=3, user=_DEFAULT_USER, name="warm", heat=2.0),
        ]
        scored = scored_entities_for_user(entities)
        names = [e.name for _, e in scored]
        assert names == ["hot", "warm", "cold"]

    def test_empty_returns_empty(self):
        assert scored_entities_for_user([]) == []

    def test_returns_heat_as_score(self):
        entity = Entity(id=1, user=_DEFAULT_USER, name="test", heat=3.14)
        scored = scored_entities_for_user([entity])
        assert scored[0][0] == pytest.approx(3.14)


class TestHeatEngine:
    """Integration tests for the HeatEngine using a real database."""

    @pytest.fixture
    def heat_db(self, tmp_path):
        """Create a temporary database with tables."""
        db_path = str(tmp_path / "heat_test.db")
        db = Database(db_path)
        db.create_tables()
        return db

    @pytest.fixture
    def engine(self, heat_db):
        """Create a HeatEngine with default runtime params."""
        runtime = RuntimeParams()
        return HeatEngine(db=heat_db, runtime=runtime)

    def test_touch_adds_heat(self, heat_db, engine):
        entity = heat_db.entities.get_or_create(_DEFAULT_USER, "test")
        assert entity is not None and entity.id is not None
        assert entity.heat == 0.0

        engine.touch(entity.id)

        refreshed = heat_db.entities.get(entity.id)
        assert refreshed is not None
        assert refreshed.heat == pytest.approx(3.0)  # default HEAT_TOUCH_AMOUNT

    def test_veto_zeroes_heat(self, heat_db, engine):
        entity = heat_db.entities.get_or_create(_DEFAULT_USER, "test")
        assert entity is not None and entity.id is not None
        heat_db.entities.update_heat(entity.id, 5.0)

        engine.veto(entity.id)

        refreshed = heat_db.entities.get(entity.id)
        assert refreshed is not None
        assert refreshed.heat == 0.0

    def test_penalize_ignore_reduces_heat(self, heat_db, engine):
        entity = heat_db.entities.get_or_create(_DEFAULT_USER, "test")
        assert entity is not None and entity.id is not None
        heat_db.entities.update_heat(entity.id, 10.0)

        engine.penalize_ignore(entity.id)

        refreshed = heat_db.entities.get(entity.id)
        assert refreshed is not None
        assert refreshed.heat == pytest.approx(6.0)  # 10.0 * 0.6

    def test_decay_reduces_all_heat(self, heat_db, engine):
        e1 = heat_db.entities.get_or_create(_DEFAULT_USER, "hot")
        e2 = heat_db.entities.get_or_create(_DEFAULT_USER, "warm")
        assert e1 is not None and e1.id is not None
        assert e2 is not None and e2.id is not None
        heat_db.entities.update_heat(e1.id, 10.0)
        heat_db.entities.update_heat(e2.id, 4.0)

        engine.run_cycle(_DEFAULT_USER)

        r1 = heat_db.entities.get(e1.id)
        r2 = heat_db.entities.get(e2.id)
        assert r1 is not None and r2 is not None
        # After decay (0.85): 10*0.85=8.5, 4*0.85=3.4
        # Radiation may transfer some heat, but total should be conserved
        total_before = 10.0 + 4.0
        total_after = r1.heat + r2.heat
        # Decay reduces total: 14 * 0.85 = 11.9
        assert total_after == pytest.approx(total_before * 0.85, abs=0.1)

    def test_cooldown_start_and_tick(self, heat_db, engine):
        entity = heat_db.entities.get_or_create(_DEFAULT_USER, "test")
        assert entity is not None and entity.id is not None

        engine.start_cooldown(entity.id)
        refreshed = heat_db.entities.get(entity.id)
        assert refreshed is not None
        assert refreshed.heat_cooldown == 3  # default HEAT_COOLDOWN_CYCLES

        # Run one cycle — cooldown should decrement
        engine.run_cycle(_DEFAULT_USER)
        refreshed = heat_db.entities.get(entity.id)
        assert refreshed is not None
        assert refreshed.heat_cooldown == 2

    def test_radiation_conserves_heat(self, heat_db, engine):
        """Radiation transfers heat from hot to cold, conserving total (post-decay)."""
        hot = heat_db.entities.get_or_create(_DEFAULT_USER, "hot entity")
        cold = heat_db.entities.get_or_create(_DEFAULT_USER, "cold entity")
        assert hot is not None and hot.id is not None
        assert cold is not None and cold.id is not None

        heat_db.entities.update_heat(hot.id, 10.0)
        heat_db.entities.update_heat(cold.id, 0.0)

        # Give both embeddings that are similar (same vector = similarity 1.0)
        vec = [1.0, 0.0, 0.0]
        emb = serialize_embedding(vec)
        heat_db.entities.update_embedding(hot.id, emb)
        heat_db.entities.update_embedding(cold.id, emb)

        engine.run_cycle(_DEFAULT_USER)

        r_hot = heat_db.entities.get(hot.id)
        r_cold = heat_db.entities.get(cold.id)
        assert r_hot is not None and r_cold is not None

        # Cold entity should have gained heat from radiation
        assert r_cold.heat > 0
        # Hot entity should have lost heat
        assert r_hot.heat < 10.0
        # Total should equal decayed total (10 * 0.85 = 8.5)
        assert r_hot.heat + r_cold.heat == pytest.approx(10.0 * 0.85, abs=0.1)

    def test_radiation_skips_dissimilar_entities(self, heat_db, engine):
        """Entities below similarity threshold don't receive radiation."""
        hot = heat_db.entities.get_or_create(_DEFAULT_USER, "hot entity")
        cold = heat_db.entities.get_or_create(_DEFAULT_USER, "cold entity")
        assert hot is not None and hot.id is not None
        assert cold is not None and cold.id is not None

        heat_db.entities.update_heat(hot.id, 10.0)
        heat_db.entities.update_heat(cold.id, 0.0)

        # Give orthogonal embeddings (cosine similarity = 0)
        heat_db.entities.update_embedding(hot.id, serialize_embedding([1.0, 0.0, 0.0]))
        heat_db.entities.update_embedding(cold.id, serialize_embedding([0.0, 1.0, 0.0]))

        engine.run_cycle(_DEFAULT_USER)

        r_cold = heat_db.entities.get(cold.id)
        assert r_cold is not None
        # No radiation received — cold stays at 0 (after decay of 0 = 0)
        assert r_cold.heat == pytest.approx(0.0)

    def test_seed_intrinsic_heat(self, heat_db, engine):
        """Newly discovered entity gets intrinsic heat from similar hot entities."""
        existing = heat_db.entities.get_or_create(_DEFAULT_USER, "existing hot")
        assert existing is not None and existing.id is not None
        heat_db.entities.update_heat(existing.id, 8.0)
        heat_db.entities.update_embedding(existing.id, serialize_embedding([1.0, 0.0, 0.0]))

        new_entity = heat_db.entities.get_or_create(_DEFAULT_USER, "new discovery")
        assert new_entity is not None and new_entity.id is not None
        # Similar embedding to existing hot entity
        heat_db.entities.update_embedding(new_entity.id, serialize_embedding([0.9, 0.1, 0.0]))

        engine.seed_intrinsic_heat(new_entity.id, _DEFAULT_USER)

        refreshed = heat_db.entities.get(new_entity.id)
        assert refreshed is not None
        # Should have some intrinsic heat (similarity * avg_heat)
        assert refreshed.heat > 0

    def test_seed_intrinsic_heat_no_hot_entities(self, heat_db, engine):
        """No intrinsic heat when there are no hot entities."""
        new_entity = heat_db.entities.get_or_create(_DEFAULT_USER, "lonely")
        assert new_entity is not None and new_entity.id is not None
        heat_db.entities.update_embedding(new_entity.id, serialize_embedding([1.0, 0.0, 0.0]))

        engine.seed_intrinsic_heat(new_entity.id, _DEFAULT_USER)

        refreshed = heat_db.entities.get(new_entity.id)
        assert refreshed is not None
        assert refreshed.heat == 0.0

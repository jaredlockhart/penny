"""Tests for heat engine and entity scoring."""

from datetime import UTC, datetime, timedelta

import pytest

from penny.config_params import RuntimeParams
from penny.database import Database
from penny.database.models import Entity
from penny.interest import (
    HeatEngine,
    _time_decay_factor,
    scored_entities_for_user,
)
from penny.ollama.embeddings import serialize_embedding

_DEFAULT_USER = "+1234"


class TestTimeDecayFactor:
    """Tests for the time-based decay factor computation."""

    def test_zero_elapsed_returns_one(self):
        assert _time_decay_factor(0.0, half_life_days=7.0) == pytest.approx(1.0)

    def test_negative_elapsed_returns_one(self):
        assert _time_decay_factor(-100.0, half_life_days=7.0) == pytest.approx(1.0)

    def test_one_half_life_returns_half(self):
        seven_days_seconds = 7.0 * 86400.0
        assert _time_decay_factor(seven_days_seconds, half_life_days=7.0) == pytest.approx(0.5)

    def test_two_half_lives_returns_quarter(self):
        fourteen_days_seconds = 14.0 * 86400.0
        assert _time_decay_factor(fourteen_days_seconds, half_life_days=7.0) == pytest.approx(0.25)

    def test_short_elapsed_barely_decays(self):
        ten_seconds = 10.0
        factor = _time_decay_factor(ten_seconds, half_life_days=7.0)
        assert factor > 0.999  # Barely any decay in 10 seconds


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

    def test_discovery_heat_novelty_only(self, heat_db, engine):
        """Entity with no embedding gets novelty heat only (no intrinsic)."""
        entity = heat_db.entities.get_or_create(_DEFAULT_USER, "brand new")
        assert entity is not None and entity.id is not None
        assert entity.heat == 0.0

        engine.seed_discovery_heat(entity.id, _DEFAULT_USER)

        refreshed = heat_db.entities.get(entity.id)
        assert refreshed is not None
        assert refreshed.heat == pytest.approx(1.0)  # default HEAT_NOVELTY_AMOUNT

    def test_discovery_heat_scaled_by_relevance(self, heat_db, engine):
        """Relevance scales the novelty component."""
        entity = heat_db.entities.get_or_create(_DEFAULT_USER, "partial match")
        assert entity is not None and entity.id is not None

        engine.seed_discovery_heat(entity.id, _DEFAULT_USER, relevance=0.5)

        refreshed = heat_db.entities.get(entity.id)
        assert refreshed is not None
        assert refreshed.heat == pytest.approx(0.5)  # 1.0 * 0.5

    def test_penalize_ignore_reduces_heat(self, heat_db, engine):
        entity = heat_db.entities.get_or_create(_DEFAULT_USER, "test")
        assert entity is not None and entity.id is not None
        heat_db.entities.update_heat(entity.id, 10.0)

        engine.penalize_ignore(entity.id)

        refreshed = heat_db.entities.get(entity.id)
        assert refreshed is not None
        assert refreshed.heat == pytest.approx(6.0)  # 10.0 * 0.6

    def test_decay_is_time_based(self, heat_db, engine):
        """Decay uses elapsed wall-clock time, not fixed per-cycle multiplier."""
        e1 = heat_db.entities.get_or_create(_DEFAULT_USER, "hot")
        assert e1 is not None and e1.id is not None
        heat_db.entities.update_heat(e1.id, 10.0)

        # Stamp initial decay time in the past (1 half-life = 7 days ago)
        seven_days_ago = datetime.now(UTC) - timedelta(days=7)
        heat_db.entities.apply_heat_decay(_DEFAULT_USER, 1.0, seven_days_ago)

        engine.run_cycle(_DEFAULT_USER)

        r1 = heat_db.entities.get(e1.id)
        assert r1 is not None
        # After one half-life, heat should be approximately halved
        assert r1.heat == pytest.approx(5.0, abs=0.1)

    def test_first_decay_stamps_time_without_decaying(self, heat_db, engine):
        """First decay on a fresh DB stamps time but doesn't reduce heat."""
        entity = heat_db.entities.get_or_create(_DEFAULT_USER, "test")
        assert entity is not None and entity.id is not None
        heat_db.entities.update_heat(entity.id, 10.0)

        engine.run_cycle(_DEFAULT_USER)

        refreshed = heat_db.entities.get(entity.id)
        assert refreshed is not None
        # First decay: factor=1.0, no reduction
        assert refreshed.heat == pytest.approx(10.0)
        assert refreshed.heat_decayed_at is not None

    def test_rapid_cycles_barely_decay(self, heat_db, engine):
        """Multiple rapid cycles (milliseconds apart) cause negligible decay."""
        entity = heat_db.entities.get_or_create(_DEFAULT_USER, "test")
        assert entity is not None and entity.id is not None
        heat_db.entities.update_heat(entity.id, 10.0)

        # Run 100 rapid cycles
        for _ in range(100):
            engine.run_cycle(_DEFAULT_USER)

        refreshed = heat_db.entities.get(entity.id)
        assert refreshed is not None
        # With 7-day half-life, 100 rapid cycles should barely reduce heat
        assert refreshed.heat > 9.9

    def test_cooldown_is_time_based(self, heat_db, engine):
        """Cooldown uses a deadline timestamp, not cycle counting."""
        entity = heat_db.entities.get_or_create(_DEFAULT_USER, "test")
        assert entity is not None and entity.id is not None

        engine.start_cooldown(entity.id)
        refreshed = heat_db.entities.get(entity.id)
        assert refreshed is not None
        assert refreshed.heat_cooldown_until is not None
        # Cooldown should be ~1 hour in the future (default HEAT_COOLDOWN_SECONDS=3600)
        cooldown_until = refreshed.heat_cooldown_until
        if cooldown_until.tzinfo is None:
            cooldown_until = cooldown_until.replace(tzinfo=UTC)
        remaining = (cooldown_until - datetime.now(UTC)).total_seconds()
        assert 3500 < remaining < 3700

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

        # Stamp decay time so first run_cycle applies negligible time-decay
        heat_db.entities.apply_heat_decay(_DEFAULT_USER, 1.0, datetime.now(UTC))

        engine.run_cycle(_DEFAULT_USER)

        r_hot = heat_db.entities.get(hot.id)
        r_cold = heat_db.entities.get(cold.id)
        assert r_hot is not None and r_cold is not None

        # Cold entity should have gained heat from radiation
        assert r_cold.heat > 0
        # Hot entity should have lost heat
        assert r_hot.heat < 10.0
        # Total should be approximately preserved (negligible time decay)
        assert r_hot.heat + r_cold.heat == pytest.approx(10.0, abs=0.1)

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

        # Stamp decay time
        heat_db.entities.apply_heat_decay(_DEFAULT_USER, 1.0, datetime.now(UTC))

        engine.run_cycle(_DEFAULT_USER)

        r_cold = heat_db.entities.get(cold.id)
        assert r_cold is not None
        # No radiation received — cold stays at 0
        assert r_cold.heat == pytest.approx(0.0)

    def test_discovery_heat_includes_intrinsic(self, heat_db, engine):
        """Discovery heat adds intrinsic heat from similar hot entities."""
        existing = heat_db.entities.get_or_create(_DEFAULT_USER, "existing hot")
        assert existing is not None and existing.id is not None
        heat_db.entities.update_heat(existing.id, 8.0)
        heat_db.entities.update_embedding(existing.id, serialize_embedding([1.0, 0.0, 0.0]))

        new_entity = heat_db.entities.get_or_create(_DEFAULT_USER, "new discovery")
        assert new_entity is not None and new_entity.id is not None
        heat_db.entities.update_embedding(new_entity.id, serialize_embedding([0.9, 0.1, 0.0]))

        engine.seed_discovery_heat(new_entity.id, _DEFAULT_USER)

        refreshed = heat_db.entities.get(new_entity.id)
        assert refreshed is not None
        # novelty (1.0) + intrinsic (similarity * avg_heat > 0)
        assert refreshed.heat > 1.0

    def test_discovery_heat_no_hot_entities(self, heat_db, engine):
        """Without hot entities, discovery heat is novelty only."""
        new_entity = heat_db.entities.get_or_create(_DEFAULT_USER, "lonely")
        assert new_entity is not None and new_entity.id is not None
        heat_db.entities.update_embedding(new_entity.id, serialize_embedding([1.0, 0.0, 0.0]))

        engine.seed_discovery_heat(new_entity.id, _DEFAULT_USER)

        refreshed = heat_db.entities.get(new_entity.id)
        assert refreshed is not None
        assert refreshed.heat == pytest.approx(1.0)  # Novelty only

    def test_discovery_heat_none_relevance_gives_full_novelty(self, heat_db, engine):
        """None relevance (unscored) gives full novelty amount."""
        entity = heat_db.entities.get_or_create(_DEFAULT_USER, "unscored")
        assert entity is not None and entity.id is not None

        engine.seed_discovery_heat(entity.id, _DEFAULT_USER, relevance=None)

        refreshed = heat_db.entities.get(entity.id)
        assert refreshed is not None
        assert refreshed.heat == pytest.approx(1.0)

    def test_discovery_heat_zero_relevance_gives_zero_novelty(self, heat_db, engine):
        """Zero relevance gives zero novelty (not full amount)."""
        entity = heat_db.entities.get_or_create(_DEFAULT_USER, "irrelevant")
        assert entity is not None and entity.id is not None

        engine.seed_discovery_heat(entity.id, _DEFAULT_USER, relevance=0.0)

        refreshed = heat_db.entities.get(entity.id)
        assert refreshed is not None
        assert refreshed.heat == pytest.approx(0.0)

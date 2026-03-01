"""Heat engine for entity interest scoring.

The heat model is the single source of truth for entity interest:
- Heat accumulates from user engagement (touch)
- Heat decays based on wall-clock time (configurable half-life in days)
- Hot entities radiate heat to cooler semantic neighbors (conservative transfer)
- Negative engagement (thumbs down) zeroes heat
- Ignored notifications penalize heat multiplicatively
- Newly discovered entities get intrinsic heat from similarity to hot entities
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from penny.database.models import Entity
from penny.ollama.embeddings import cosine_similarity, deserialize_embedding

if TYPE_CHECKING:
    from penny.config_params import RuntimeParams
    from penny.database import Database

logger = logging.getLogger(__name__)

_SECONDS_PER_DAY = 86400.0


def _time_decay_factor(elapsed_seconds: float, half_life_days: float) -> float:
    """Compute multiplicative decay factor from elapsed wall-clock time.

    factor = 2^(-elapsed_seconds / (half_life_days * 86400))
    At elapsed = 0: factor = 1.0 (no decay)
    At elapsed = half_life: factor = 0.5
    """
    if elapsed_seconds <= 0:
        return 1.0
    return math.pow(2.0, -elapsed_seconds / (half_life_days * _SECONDS_PER_DAY))


# --- Heat engine ---


class HeatEngine:
    """Thermodynamic heat model for entity interest scoring.

    Heat is the single persistent score for each entity. The engine runs
    cycles (decay → radiate) and handles engagement-driven heat touches,
    ignore penalties, veto resets, and intrinsic heat seeding.

    Decay is time-based: the decay factor is computed from wall-clock time
    since the last decay, using a configurable half-life in days.
    """

    def __init__(self, db: Database, runtime: RuntimeParams) -> None:
        self._db = db
        self._runtime = runtime

    def run_cycle(self, user: str) -> None:
        """Run one heat cycle: decay → radiate."""
        self._decay(user)
        self._radiate(user)

    def touch(self, entity_id: int) -> None:
        """Add heat on positive engagement."""
        self._db.entities.add_heat(entity_id, self._runtime.HEAT_TOUCH_AMOUNT)

    def penalize_ignore(self, entity_id: int) -> None:
        """Reduce heat when a notification is ignored."""
        self._db.entities.update_heat(
            entity_id,
            self._get_heat(entity_id) * self._runtime.HEAT_IGNORE_PENALTY,
        )

    def veto(self, entity_id: int) -> None:
        """Zero out heat on negative engagement (thumbs down)."""
        self._db.entities.update_heat(entity_id, 0.0)

    def seed_discovery_heat(
        self, entity_id: int, user: str, relevance: float | None = None
    ) -> None:
        """Set starting heat for a newly discovered entity.

        Combines two additive signals:
        - Novelty: HEAT_NOVELTY_AMOUNT scaled by relevance (None = full amount)
        - Intrinsic: similarity to existing hot entities × their average heat
        """
        novelty = self._novelty_amount(relevance)
        intrinsic = self._intrinsic_amount(entity_id, user)
        heat = novelty + intrinsic
        self._db.entities.update_heat(entity_id, heat)
        logger.info(
            "Seeded discovery heat %.2f for entity %d (novelty=%.2f, intrinsic=%.2f)",
            heat,
            entity_id,
            novelty,
            intrinsic,
        )

    def start_cooldown(self, entity_id: int) -> None:
        """Put an entity on cooldown after being notified."""
        until = datetime.now(UTC) + timedelta(seconds=self._runtime.HEAT_COOLDOWN_SECONDS)
        self._db.entities.update_heat_cooldown_until(entity_id, until)

    # --- Internal methods ---

    def _novelty_amount(self, relevance: float | None) -> float:
        """Compute novelty heat: base amount scaled by relevance (None = full)."""
        amount = self._runtime.HEAT_NOVELTY_AMOUNT
        if relevance is not None:
            amount *= relevance
        return amount

    def _intrinsic_amount(self, entity_id: int, user: str) -> float:
        """Compute intrinsic heat from similarity to existing hot entities."""
        entity = self._db.entities.get(entity_id)
        if entity is None or entity.embedding is None:
            return 0.0

        hot_entities = self._get_hot_entities_with_embeddings(user)
        if not hot_entities:
            return 0.0

        similarity, avg_heat = self._compute_intrinsic_params(entity, hot_entities)
        if similarity <= 0.0:
            return 0.0

        return similarity * avg_heat

    def _decay(self, user: str) -> None:
        """Apply time-based multiplicative decay to all entity heat for a user.

        Computes elapsed time since last decay and derives the decay factor
        from the configured half-life. On first call (no prior timestamp),
        stamps current time without decaying.
        """
        now = datetime.now(UTC)
        last_decayed_at = self._db.entities.get_heat_decayed_at(user)

        if last_decayed_at is None:
            # First decay ever — stamp time, no decay applied
            self._db.entities.apply_heat_decay(user, 1.0, now)
            return

        if last_decayed_at.tzinfo is None:
            last_decayed_at = last_decayed_at.replace(tzinfo=UTC)

        elapsed = (now - last_decayed_at).total_seconds()
        factor = _time_decay_factor(elapsed, self._runtime.HEAT_DECAY_HALF_LIFE_DAYS)
        self._db.entities.apply_heat_decay(user, factor, now)

    def _radiate(self, user: str) -> None:
        """Transfer heat from hot entities to cooler semantic neighbors.

        Conservative: source loses exactly what neighbors gain.
        Only radiates to neighbors above the similarity threshold.
        """
        entities = self._db.entities.get_with_embeddings(user)
        if len(entities) < 2:
            return

        entity_map, embeddings = self._prepare_radiation_data(entities)
        transfers = self._compute_radiation_transfers(entity_map, embeddings)
        self._apply_transfers(entity_map, transfers)

    def _prepare_radiation_data(
        self, entities: list[Entity]
    ) -> tuple[dict[int, Entity], dict[int, list[float]]]:
        """Build lookup maps for radiation computation."""
        entity_map: dict[int, Entity] = {}
        embeddings: dict[int, list[float]] = {}
        for e in entities:
            assert e.id is not None
            entity_map[e.id] = e
            assert e.embedding is not None
            embeddings[e.id] = deserialize_embedding(e.embedding)
        return entity_map, embeddings

    def _compute_radiation_transfers(
        self,
        entity_map: dict[int, Entity],
        embeddings: dict[int, list[float]],
    ) -> dict[int, float]:
        """Compute net heat transfers for all entities.

        Returns a dict of entity_id -> net heat change (positive = gains, negative = losses).
        """
        threshold = self._runtime.HEAT_RADIATION_THRESHOLD
        rate = self._runtime.HEAT_RADIATION_RATE
        top_k = int(self._runtime.HEAT_RADIATION_TOP_K)

        transfers: dict[int, float] = dict.fromkeys(entity_map, 0.0)

        for src_id, src_entity in entity_map.items():
            if src_entity.heat <= 0:
                continue
            neighbors = self._find_radiation_neighbors(
                src_id, embeddings, entity_map, threshold, top_k
            )
            if not neighbors:
                continue
            self._distribute_radiation(src_id, src_entity.heat, rate, neighbors, transfers)

        return transfers

    def _find_radiation_neighbors(
        self,
        src_id: int,
        embeddings: dict[int, list[float]],
        entity_map: dict[int, Entity],
        threshold: float,
        top_k: int,
    ) -> list[tuple[int, float]]:
        """Find the top-K cooler neighbors above similarity threshold."""
        src_vec = embeddings[src_id]
        src_heat = entity_map[src_id].heat

        candidates: list[tuple[int, float]] = []
        for other_id, other_vec in embeddings.items():
            if other_id == src_id:
                continue
            if entity_map[other_id].heat >= src_heat:
                continue  # Only radiate to cooler entities
            sim = cosine_similarity(src_vec, other_vec)
            if sim >= threshold:
                candidates.append((other_id, sim))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def _distribute_radiation(
        self,
        src_id: int,
        src_heat: float,
        rate: float,
        neighbors: list[tuple[int, float]],
        transfers: dict[int, float],
    ) -> None:
        """Distribute radiation from source to neighbors, weighted by similarity.

        Conservative: total given to neighbors equals total lost by source.
        """
        total_sim = sum(sim for _, sim in neighbors)
        if total_sim <= 0:
            return

        budget = src_heat * rate
        for neighbor_id, sim in neighbors:
            share = budget * (sim / total_sim)
            transfers[neighbor_id] += share
            transfers[src_id] -= share

    def _apply_transfers(self, entity_map: dict[int, Entity], transfers: dict[int, float]) -> None:
        """Apply computed heat transfers to the database."""
        for eid, delta in transfers.items():
            if abs(delta) < 0.001:
                continue
            new_heat = max(entity_map[eid].heat + delta, 0.0)
            self._db.entities.update_heat(eid, new_heat)

    def _get_heat(self, entity_id: int) -> float:
        """Get current heat for an entity."""
        entity = self._db.entities.get(entity_id)
        return entity.heat if entity else 0.0

    def _get_hot_entities_with_embeddings(self, user: str) -> list[tuple[Entity, list[float]]]:
        """Get entities with heat > 0 and embeddings for intrinsic heat computation."""
        entities = self._db.entities.get_with_embeddings(user)
        result: list[tuple[Entity, list[float]]] = []
        for e in entities:
            if e.heat > 0 and e.embedding is not None:
                result.append((e, deserialize_embedding(e.embedding)))
        return result

    def _compute_intrinsic_params(
        self,
        entity: Entity,
        hot_entities: list[tuple[Entity, list[float]]],
    ) -> tuple[float, float]:
        """Compute max similarity and average heat for intrinsic heat seeding."""
        assert entity.embedding is not None
        entity_vec = deserialize_embedding(entity.embedding)

        best_sim = 0.0
        total_heat = 0.0
        for hot_entity, hot_vec in hot_entities:
            if hot_entity.id == entity.id:
                continue
            sim = cosine_similarity(entity_vec, hot_vec)
            best_sim = max(best_sim, sim)
            total_heat += hot_entity.heat

        avg_heat = total_heat / len(hot_entities) if hot_entities else 0.0
        return best_sim, avg_heat


# --- Main scoring entry point ---


def scored_entities_for_user(entities: list[Entity]) -> list[tuple[float, Entity]]:
    """Score entities by persistent heat, sorted descending.

    Heat is maintained by the HeatEngine — this function simply reads
    the current heat values and sorts.
    """
    scored = [(e.heat, e) for e in entities]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored

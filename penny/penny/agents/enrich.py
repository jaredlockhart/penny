"""Adaptive enrichment agent — background research driven by interest scores."""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from penny.agents.base import Agent
from penny.agents.extraction import _is_valid_entity_name
from penny.constants import PennyConstants
from penny.database.models import Engagement, Entity, Fact
from penny.interest import compute_notification_interest
from penny.ollama.embeddings import (
    build_entity_embed_text,
    deserialize_embedding,
    serialize_embedding,
)
from penny.ollama.similarity import (
    DedupStrategy,
    check_relevance,
    dedup_facts_by_embedding,
    embed_text,
    is_embedding_duplicate,
    normalize_fact,
)
from penny.prompts import Prompt
from penny.tools.models import SearchResult

if TYPE_CHECKING:
    from penny.tools import Tool

logger = logging.getLogger(__name__)


class ExtractedFacts(BaseModel):
    """Schema for LLM response: facts extracted from search results."""

    facts: list[str] = Field(
        default_factory=list,
        description="NEW specific, verifiable facts about the entity from the text",
    )


class _DiscoveredEntity(BaseModel):
    """A candidate entity discovered during enrichment search results."""

    name: str = Field(description="Entity name (e.g., 'Uni-Q driver', 'Andrew Jones')")
    tagline: str = Field(
        default="",
        description="Short 3-8 word description of what the entity is",
    )


class _DiscoveredEntities(BaseModel):
    """Schema for LLM response: related entities found in enrichment search results."""

    entities: list[_DiscoveredEntity] = Field(default_factory=list)


class _ScoredEntity:
    """Internal container for entity priority scoring."""

    __slots__ = ("entity", "user", "interest", "fact_count", "facts", "priority")

    def __init__(
        self,
        entity: Entity,
        user: str,
        interest: float,
        fact_count: int,
        facts: list[Fact],
        priority: float,
    ) -> None:
        self.entity = entity
        self.user = user
        self.interest = interest
        self.fact_count = fact_count
        self.facts = facts
        self.priority = priority


class EnrichAgent(Agent):
    """Background agent that adaptively researches entities based on interest scores.

    Picks the highest-priority entity across all users each cycle:
    - Enrichment mode (few facts): broad search to build knowledge
    - Briefing mode (many facts): targeted search for new developments
    - Skip: if entity has negative interest

    Uses SearchTool for Perplexity searches and OllamaClient for fact extraction
    and message composition.
    """

    def __init__(self, search_tool: Tool | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._search_tool = search_tool
        self._last_enrich_time: float | None = None

    @property
    def name(self) -> str:
        """Task name for logging."""
        return "enrich"

    def _should_enrich(self) -> bool:
        """Check if the fixed enrichment interval has elapsed."""
        if self._last_enrich_time is None:
            return True
        elapsed = time.monotonic() - self._last_enrich_time
        return elapsed >= self.config.runtime.ENRICHMENT_INTERVAL

    def _mark_enrichment_done(self) -> None:
        """Record that an enrichment search was performed."""
        self._last_enrich_time = time.monotonic()
        logger.info("Enrichment done, next in %.0fs", self.config.runtime.ENRICHMENT_INTERVAL)

    async def execute(self) -> bool:
        """Run one cycle of the enrich agent.

        Returns:
            True if work was done, False if nothing to research.
        """
        if not self._search_tool:
            logger.debug("EnrichAgent: no search tool configured")
            return False

        if not self._should_enrich():
            return False

        candidate = self._select_candidate()
        if candidate is None:
            return False

        did_work = await self._research_entity(candidate)
        if did_work:
            self._mark_enrichment_done()
        return did_work

    def _select_candidate(self) -> _ScoredEntity | None:
        """Score all entities and return the highest-priority candidate, or None."""
        candidates = self._score_candidates()
        if not candidates:
            logger.debug("EnrichAgent: no candidates to research")
            return None
        return max(candidates, key=lambda c: c.priority)

    async def _research_entity(self, candidate: _ScoredEntity) -> bool:
        """Search, extract facts, and store results for the selected entity.

        Returns:
            True if search succeeded, False if search failed.
        """
        entity = candidate.entity
        assert entity.id is not None

        is_enrichment = candidate.fact_count < self.config.runtime.LEARN_ENRICHMENT_FACT_THRESHOLD
        mode_label = "enrichment" if is_enrichment else "briefing"
        logger.info(
            "Learn: %s mode for '%s' (user=%s, interest=%.2f, facts=%d, priority=%.3f)",
            mode_label,
            entity.name,
            candidate.user,
            candidate.interest,
            candidate.fact_count,
            candidate.priority,
        )

        query = self._build_query(entity.name, is_enrichment, candidate.facts, entity.tagline)
        logger.info("Learn search query: '%s'", query)

        search_result = await self._search(query)
        if search_result is None:
            return False

        new_facts, confirmed_fact_ids = await self._extract_and_dedup_facts(
            entity, candidate.facts, search_result
        )
        logger.info(
            "Learn extracted %d new facts, %d confirmed existing for '%s'",
            len(new_facts),
            len(confirmed_fact_ids),
            entity.name,
        )

        stored_facts = await self._store_new_facts(entity, new_facts, search_result)

        if stored_facts and self._embedding_model_client:
            await self._update_entity_embedding(entity)
            logger.info("Updated entity embedding for '%s'", entity.name)

        # Discover related entities from the same search results
        if self._embedding_model_client:
            discovered = await self._discover_related_entities(
                entity, candidate.user, search_result
            )
            if discovered:
                logger.info(
                    "Discovered %d related entities for '%s'",
                    len(discovered),
                    entity.name,
                )

        self.db.entities.update_last_enriched_at(entity.id)
        return True

    def _score_candidates(self) -> list[_ScoredEntity]:
        """Score all entities across all users by behavioral interest.

        Returns candidates above the minimum interest threshold, sorted by priority.
        """
        users = self.db.users.get_all_senders()
        if not users:
            return []

        candidates: list[_ScoredEntity] = []
        cooldown = self.config.runtime.ENRICHMENT_ENTITY_COOLDOWN
        now = datetime.now(UTC)

        for user in users:
            entities = self.db.entities.get_for_user(user)
            if not entities:
                continue

            all_engagements = self.db.engagements.get_for_user(user)
            engagements_by_entity: dict[int, list[Engagement]] = defaultdict(list)
            for eng in all_engagements:
                if eng.entity_id is not None:
                    engagements_by_entity[eng.entity_id].append(eng)

            for entity in entities:
                assert entity.id is not None
                if not self._is_entity_enrichment_eligible(entity, now, cooldown):
                    continue
                scored = self._compute_entity_priority(entity, engagements_by_entity, user)
                if scored is not None:
                    candidates.append(scored)

        return candidates

    def _is_entity_enrichment_eligible(
        self, entity: Entity, now: datetime, cooldown: float
    ) -> bool:
        """Check if an entity is eligible for enrichment (not in cooldown window)."""
        if entity.last_enriched_at is None:
            return True
        last = entity.last_enriched_at
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
        elapsed = (now - last).total_seconds()
        if elapsed < cooldown:
            logger.debug(
                "EnrichAgent: skipping '%s' (enriched %.0fs ago, cooldown=%.0fs)",
                entity.name,
                elapsed,
                cooldown,
            )
            return False
        return True

    def _compute_entity_priority(
        self,
        entity: Entity,
        engagements_by_entity: dict[int, list[Engagement]],
        user: str,
    ) -> _ScoredEntity | None:
        """Compute priority score for an entity. Returns None if ineligible."""
        assert entity.id is not None

        entity_engagements = engagements_by_entity.get(entity.id, [])
        interest = compute_notification_interest(
            entity_engagements,
            half_life_days=self.config.runtime.INTEREST_SCORE_HALF_LIFE_DAYS,
        )
        if interest < self.config.runtime.LEARN_MIN_INTEREST_SCORE:
            return None

        facts = self.db.facts.get_for_entity(entity.id)
        fact_count = len(facts)

        # Skip entities with unannounced facts — notification hasn't
        # surfaced them yet, so don't pile on more.
        if any(f.notified_at is None for f in facts):
            logger.debug(
                "EnrichAgent: skipping '%s' (has unannounced facts)",
                entity.name,
            )
            return None

        # Log-diminishing returns: high-interest entities stay on top,
        # but gradually yield as facts accumulate, allowing rotation.
        # log2(0+2)=1.0, log2(3+2)=2.3, log2(7+2)=3.2, log2(15+2)=4.1
        priority = interest / math.log2(fact_count + 2)

        return _ScoredEntity(
            entity=entity,
            user=user,
            interest=interest,
            fact_count=fact_count,
            facts=facts,
            priority=priority,
        )

    def _build_query(
        self,
        entity_name: str,
        is_enrichment: bool,
        existing_facts: list[Fact],
        tagline: str | None = None,
    ) -> str:
        """Build a search query based on mode.

        For enrichment, includes existing facts so Perplexity can focus on
        information we don't already have. The tagline disambiguates entities
        with common names (e.g., "genesis" → "genesis (british prog rock band)").
        """
        label = f"{entity_name} ({tagline})" if tagline else entity_name
        if is_enrichment:
            if not existing_facts:
                return label
            facts_text = "\n".join(f"- {f.content}" for f in existing_facts)
            return (
                f"Tell me more about {label}. "
                f"I already know:\n{facts_text}\n\n"
                f"What else is important to know?"
            )
        year = datetime.now(UTC).year
        return f"{label} latest news updates {year}"

    async def _search(self, query: str) -> str | None:
        """Execute a search via SearchTool. Returns the text result or None."""
        assert self._search_tool is not None
        try:
            result = await self._search_tool.execute(
                query=query,
                skip_images=True,
                trigger=PennyConstants.SearchTrigger.PENNY_ENRICHMENT,
            )
            if isinstance(result, SearchResult):
                return result.text
            return str(result) if result else None
        except Exception as e:
            logger.error("Learn search failed: %s", e)
            return None

    async def _extract_and_dedup_facts(
        self,
        entity: Entity,
        existing_facts: list[Fact],
        search_text: str,
    ) -> tuple[list[str], list[int]]:
        """Extract facts from search results and deduplicate against existing facts.

        Returns:
            Tuple of (new_fact_texts, confirmed_existing_fact_ids)
        """
        candidate_facts = await self._extract_raw_facts(entity, existing_facts, search_text)
        if not candidate_facts:
            return [], []

        new_facts, confirmed_ids = self._string_dedup_facts(candidate_facts, existing_facts)
        if not new_facts:
            return [], confirmed_ids

        deduped, embed_confirmed = await self._embedding_dedup_facts(new_facts, existing_facts)
        confirmed_ids.extend(embed_confirmed)
        return deduped, confirmed_ids

    async def _extract_raw_facts(
        self, entity: Entity, existing_facts: list[Fact], search_text: str
    ) -> list[str]:
        """Call LLM to extract facts from search text for the given entity."""
        assert entity.id is not None

        existing_context = ""
        if existing_facts:
            facts_text = "\n".join(f"- {f.content}" for f in existing_facts)
            existing_context = (
                f"\n\nAlready known facts (return only NEW facts not listed here):\n{facts_text}"
            )

        entity_label = f"{entity.name} ({entity.tagline})" if entity.tagline else entity.name
        prompt = (
            f"{Prompt.ENTITY_FACT_EXTRACTION_PROMPT}\n\n"
            f"Entity: {entity_label}\n\n"
            f"Content:\n{search_text}"
            f"{existing_context}"
        )

        try:
            response = await self._background_model_client.generate(
                prompt=prompt,
                tools=None,
                format=ExtractedFacts.model_json_schema(),
            )
            if not response.content or not response.content.strip():
                logger.warning(
                    "Empty LLM response from fact extraction for '%s' — skipping", entity.name
                )
                return []
            extracted = ExtractedFacts.model_validate_json(response.content)
            return extracted.facts
        except Exception as e:
            logger.error("Failed to extract facts for '%s': %s", entity.name, e)
            return []

    def _string_dedup_facts(
        self, candidate_facts: list[str], existing_facts: list[Fact]
    ) -> tuple[list[str], list[int]]:
        """Fast pass: deduplicate candidates against existing facts by normalized string match.

        Returns:
            Tuple of (new_facts not matching existing, confirmed_existing_fact_ids)
        """
        existing_normalized = {normalize_fact(f.content): f for f in existing_facts}
        new_facts: list[str] = []
        confirmed_ids: list[int] = []
        seen_normalized: set[str] = set()

        for fact_text in candidate_facts:
            fact_text = fact_text.strip()
            if not fact_text:
                continue
            normalized = normalize_fact(fact_text)
            if normalized in seen_normalized:
                continue
            seen_normalized.add(normalized)

            if normalized in existing_normalized:
                existing_fact = existing_normalized[normalized]
                if existing_fact.id is not None:
                    confirmed_ids.append(existing_fact.id)
                continue
            new_facts.append(fact_text)

        return new_facts, confirmed_ids

    async def _embedding_dedup_facts(
        self, new_facts: list[str], existing_facts: list[Fact]
    ) -> tuple[list[str], list[int]]:
        """Slow pass: deduplicate via embedding similarity (paraphrase detection).

        Returns:
            Tuple of (deduped_facts, confirmed_existing_fact_ids)
        """
        facts_with_embeddings = [f for f in existing_facts if f.embedding is not None]
        existing_with_emb = [
            (i, f.embedding) for i, f in enumerate(facts_with_embeddings) if f.embedding is not None
        ]
        threshold = self.config.runtime.EXTRACTION_FACT_DEDUP_SIMILARITY_THRESHOLD
        survivors, matched_indices = await dedup_facts_by_embedding(
            self._embedding_model_client, new_facts, existing_with_emb, threshold
        )

        confirmed_ids: list[int] = []
        for idx in matched_indices:
            matched_fact = facts_with_embeddings[idx]
            if matched_fact.id is not None:
                confirmed_ids.append(matched_fact.id)

        return survivors, confirmed_ids

    async def _store_new_facts(
        self, entity: Entity, new_fact_texts: list[str], search_text: str
    ) -> list[str]:
        """Store new facts with embeddings. Returns list of stored fact texts."""
        if not new_fact_texts:
            return []

        assert entity.id is not None

        # Batch-embed new facts
        fact_embeddings: list[bytes | None] = [None] * len(new_fact_texts)
        if self._embedding_model_client:
            try:
                vecs = await self._embedding_model_client.embed(new_fact_texts)
                fact_embeddings = [serialize_embedding(v) for v in vecs]
            except Exception as e:
                logger.warning("Failed to embed new facts for '%s': %s", entity.name, e)

        stored: list[str] = []
        for fact_text, emb in zip(new_fact_texts, fact_embeddings, strict=True):
            fact = self.db.facts.add(
                entity_id=entity.id,
                content=fact_text,
                embedding=emb,
            )
            if fact:
                stored.append(fact_text)
                logger.info("Learn +fact for '%s': %s", entity.name, fact_text)

        return stored

    async def _update_entity_embedding(self, entity: Entity) -> None:
        """Regenerate entity embedding after adding new facts."""
        assert entity.id is not None
        try:
            assert self._embedding_model_client is not None
            facts = self.db.facts.get_for_entity(entity.id)
            text = build_entity_embed_text(entity.name, [f.content for f in facts], entity.tagline)
            vecs = await self._embedding_model_client.embed(text)
            self.db.entities.update_embedding(entity.id, serialize_embedding(vecs[0]))
        except Exception as e:
            logger.warning("Failed to update entity embedding for '%s': %s", entity.name, e)

    # --- Entity discovery during enrichment ---

    async def _discover_related_entities(
        self, entity: Entity, user: str, search_text: str
    ) -> list[Entity]:
        """Discover new entities related to the enriching entity in search results.

        Uses embedding similarity to the enriching entity as a relevance gate,
        plus embedding dedup to prevent creating duplicate entities.
        """
        assert self._embedding_model_client is not None

        enriching_vec = self._load_enriching_embedding(entity)
        if enriching_vec is None:
            return []

        existing_entities = self.db.entities.get_for_user(user)
        existing_names = {e.name for e in existing_entities}

        candidates = await self._identify_entity_candidates(
            entity.name, existing_names, search_text
        )
        if not candidates:
            return []

        scored = await self._score_discovery_candidates(candidates, enriching_vec, existing_names)
        scored.sort(key=lambda s: s[1], reverse=True)

        budget = int(self.config.runtime.ENRICHMENT_MAX_NEW_ENTITIES)
        created: list[Entity] = []
        for candidate, relevance_score, candidate_vec in scored:
            if len(created) >= budget:
                break
            new_entity = await self._create_if_not_duplicate(
                candidate,
                relevance_score,
                candidate_vec,
                existing_entities,
                existing_names,
                user,
                search_text,
            )
            if new_entity:
                created.append(new_entity)
                existing_entities.append(new_entity)
                existing_names.add(new_entity.name)

        return created

    def _load_enriching_embedding(self, entity: Entity) -> list[float] | None:
        """Reload and deserialize the enriching entity's embedding from DB."""
        assert entity.id is not None
        refreshed = self.db.entities.get(entity.id)
        if refreshed is None or refreshed.embedding is None:
            return None
        return deserialize_embedding(refreshed.embedding)

    async def _identify_entity_candidates(
        self, entity_name: str, existing_names: set[str], search_text: str
    ) -> list[_DiscoveredEntity]:
        """Call LLM to identify new entity candidates in search results."""
        known_list = "\n".join(f"- {n}" for n in sorted(existing_names))
        prompt = (
            f"{Prompt.ENRICHMENT_ENTITY_DISCOVERY_PROMPT.format(entity_name=entity_name)}\n\n"
            f"Content:\n{search_text}\n\n"
            f"Known entities (do NOT return these):\n{known_list}"
        )

        try:
            response = await self._background_model_client.generate(
                prompt=prompt,
                tools=None,
                format=_DiscoveredEntities.model_json_schema(),
            )
            if not response.content or not response.content.strip():
                return []
            result = _DiscoveredEntities.model_validate_json(response.content)
            return result.entities
        except Exception as e:
            logger.error("Failed to identify entity candidates: %s", e)
            return []

    async def _score_discovery_candidates(
        self,
        candidates: list[_DiscoveredEntity],
        enriching_vec: list[float],
        existing_names: set[str],
    ) -> list[tuple[_DiscoveredEntity, float, list[float]]]:
        """Validate and score all candidates by relevance.

        Returns list of (candidate_with_cleaned_name, relevance_score, candidate_vec).
        Only candidates that pass name validation and the relevance gate are included.
        """
        scored: list[tuple[_DiscoveredEntity, float, list[float]]] = []
        for candidate in candidates:
            name = candidate.name.lower().strip()
            if not name or not _is_valid_entity_name(name):
                continue
            if name in existing_names:
                continue
            candidate.name = name
            candidate.tagline = self._clean_tagline(candidate.tagline) or ""

            relevance = await self._check_candidate_relevance(name, enriching_vec)
            if relevance is not None:
                scored.append((candidate, relevance[0], relevance[1]))
        return scored

    async def _create_if_not_duplicate(
        self,
        candidate: _DiscoveredEntity,
        relevance_score: float,
        candidate_vec: list[float],
        existing_entities: list[Entity],
        existing_names: set[str],
        user: str,
        search_text: str,
    ) -> Entity | None:
        """Check dedup and create entity if not a duplicate."""
        if self._is_discovery_duplicate(candidate.name, candidate_vec, existing_entities):
            return None
        tagline = candidate.tagline or None
        return await self._create_discovered_entity(
            candidate.name,
            tagline,
            relevance_score,
            user,
            search_text,
        )

    def _clean_tagline(self, raw: str) -> str | None:
        """Normalize tagline: lowercase, strip trailing period, reject if too long."""
        if not raw:
            return None
        cleaned = raw.strip().lower().rstrip(".")
        if len(cleaned.split()) > 10:
            return None
        return cleaned or None

    async def _check_candidate_relevance(
        self, name: str, enriching_vec: list[float]
    ) -> tuple[float, list[float]] | None:
        """Embed candidate and check relevance to enriching entity.

        Returns (similarity_score, candidate_vec) if relevant, None otherwise.
        """
        candidate_vec = await embed_text(self._embedding_model_client, name)
        if candidate_vec is None:
            return None

        threshold = self.config.runtime.ENRICHMENT_DISCOVERY_SIMILARITY_THRESHOLD
        score = check_relevance(candidate_vec, enriching_vec, threshold)
        if score is None:
            logger.info("Discovery: rejected '%s' (relevance < %.2f)", name, threshold)
            return None

        logger.info("Discovery: accepted '%s' (relevance %.2f >= %.2f)", name, score, threshold)
        return score, candidate_vec

    def _is_discovery_duplicate(
        self, name: str, candidate_vec: list[float], existing_entities: list[Entity]
    ) -> bool:
        """Check if candidate is a duplicate of an existing entity by embedding."""
        items = [(e.name, e.embedding) for e in existing_entities]
        threshold = self.config.runtime.EXTRACTION_ENTITY_DEDUP_EMBEDDING_THRESHOLD
        match_idx = is_embedding_duplicate(
            name, candidate_vec, items, DedupStrategy.EMBEDDING_ONLY, threshold
        )
        if match_idx is not None:
            logger.info(
                "Discovery: '%s' is duplicate of '%s'", name, existing_entities[match_idx].name
            )
            return True
        return False

    async def _create_discovered_entity(
        self,
        name: str,
        tagline: str | None,
        relevance_score: float,
        user: str,
        search_text: str,
    ) -> Entity | None:
        """Create a new entity with facts and record discovery engagement."""
        facts = await self._extract_discovery_facts(name, tagline, search_text)
        if not facts:
            logger.info("Discovery: skipping '%s' (no facts extracted)", name)
            return None

        entity = self.db.entities.get_or_create(user, name)
        if entity is None or entity.id is None:
            return None

        if tagline:
            self.db.entities.update_tagline(entity.id, tagline)
            entity.tagline = tagline

        await self._store_new_facts(entity, facts, search_text)
        await self._update_entity_embedding(entity)

        self.db.engagements.add(
            user=user,
            engagement_type=PennyConstants.EngagementType.SEARCH_DISCOVERY,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=relevance_score,
            entity_id=entity.id,
        )
        logger.info(
            "Discovery: created '%s' (relevance=%.2f, %d facts)", name, relevance_score, len(facts)
        )
        return entity

    async def _extract_discovery_facts(
        self, name: str, tagline: str | None, search_text: str
    ) -> list[str]:
        """Extract facts for a newly discovered entity from search text."""
        entity_label = f"{name} ({tagline})" if tagline else name
        prompt = (
            f"{Prompt.ENTITY_FACT_EXTRACTION_PROMPT}\n\n"
            f"Entity: {entity_label}\n\n"
            f"Content:\n{search_text}"
        )

        try:
            response = await self._background_model_client.generate(
                prompt=prompt,
                tools=None,
                format=ExtractedFacts.model_json_schema(),
            )
            if not response.content or not response.content.strip():
                return []
            extracted = ExtractedFacts.model_validate_json(response.content)
            return extracted.facts
        except Exception as e:
            logger.error("Failed to extract discovery facts for '%s': %s", name, e)
            return []

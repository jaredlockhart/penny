"""Adaptive learn agent — background research driven by interest scores."""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from penny.agents.base import Agent
from penny.constants import PennyConstants
from penny.database.models import Engagement, Entity, Fact
from penny.interest import compute_interest_score
from penny.ollama.embeddings import (
    build_entity_embed_text,
    deserialize_embedding,
    find_similar,
    serialize_embedding,
)
from penny.prompts import Prompt
from penny.tools.models import SearchResult

if TYPE_CHECKING:
    from penny.tools import Tool

logger = logging.getLogger(__name__)

# Pattern to collapse whitespace and strip bullet prefixes for fact comparison
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_fact(fact: str) -> str:
    """Normalize a fact string for dedup comparison."""
    text = fact.strip().lstrip("-").strip()
    return _WHITESPACE_RE.sub(" ", text).lower()


class ExtractedFacts(BaseModel):
    """Schema for LLM response: facts extracted from search results."""

    facts: list[str] = Field(
        default_factory=list,
        description="NEW specific, verifiable facts about the entity from the text",
    )


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


class LearnAgent(Agent):
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

    @property
    def name(self) -> str:
        """Task name for logging."""
        return "learn"

    async def execute(self) -> bool:
        """Run one cycle of the learn agent.

        Returns:
            True if work was done, False if nothing to research.
        """
        if not self._search_tool:
            logger.debug("LearnAgent: no search tool configured")
            return False

        # Score all entities and pick the highest-priority one
        candidates = self._score_candidates()
        if not candidates:
            logger.debug("LearnAgent: no candidates to research")
            return False

        candidate = max(candidates, key=lambda c: c.priority)

        entity = candidate.entity
        user = candidate.user
        assert entity.id is not None

        # Determine mode
        is_enrichment = candidate.fact_count < self.config.runtime.LEARN_ENRICHMENT_FACT_THRESHOLD

        mode_label = "enrichment" if is_enrichment else "briefing"
        logger.info(
            "Learn: %s mode for '%s' (user=%s, interest=%.2f, facts=%d, priority=%.3f)",
            mode_label,
            entity.name,
            user,
            candidate.interest,
            candidate.fact_count,
            candidate.priority,
        )

        # Build search query
        query = self._build_query(entity.name, is_enrichment, candidate.facts)
        logger.info("Learn search query: '%s'", query)

        # Execute search
        search_result = await self._search(query)
        if search_result is None:
            return False

        # Extract and deduplicate facts
        new_facts, confirmed_fact_ids = await self._extract_and_dedup_facts(
            entity, candidate.facts, search_result
        )
        logger.info(
            "Learn extracted %d new facts, %d confirmed existing for '%s'",
            len(new_facts),
            len(confirmed_fact_ids),
            entity.name,
        )

        # Store new facts
        stored_facts = await self._store_new_facts(entity, new_facts, search_result)

        # Update entity embedding if we added facts
        if stored_facts and self._embedding_model_client:
            await self._update_entity_embedding(entity)
            logger.info("Updated entity embedding for '%s'", entity.name)

        return True

    def _score_candidates(self) -> list[_ScoredEntity]:
        """Score all entities across all users by behavioral interest.

        Returns candidates above the minimum interest threshold, sorted by priority.
        Semantic interest scores are applied separately in _apply_semantic_scores().
        """
        users = self.db.get_all_senders()
        if not users:
            return []

        candidates: list[_ScoredEntity] = []

        for user in users:
            entities = self.db.get_user_entities(user)
            if not entities:
                continue

            all_engagements = self.db.get_user_engagements(user)
            engagements_by_entity: dict[int, list[Engagement]] = defaultdict(list)
            for eng in all_engagements:
                if eng.entity_id is not None:
                    engagements_by_entity[eng.entity_id].append(eng)

            for entity in entities:
                assert entity.id is not None
                entity_engagements = engagements_by_entity.get(entity.id, [])
                interest = compute_interest_score(
                    entity_engagements,
                    half_life_days=self.config.runtime.INTEREST_SCORE_HALF_LIFE_DAYS,
                )

                if interest < self.config.runtime.LEARN_MIN_INTEREST_SCORE:
                    continue

                facts = self.db.get_entity_facts(entity.id)
                fact_count = len(facts)

                # Log-diminishing returns: high-interest entities stay on top,
                # but gradually yield as facts accumulate, allowing rotation.
                # log2(0+2)=1.0, log2(3+2)=2.3, log2(7+2)=3.2, log2(15+2)=4.1
                priority = interest / math.log2(fact_count + 2)

                candidates.append(
                    _ScoredEntity(
                        entity=entity,
                        user=user,
                        interest=interest,
                        fact_count=fact_count,
                        facts=facts,
                        priority=priority,
                    )
                )

        return candidates

    def _build_query(
        self, entity_name: str, is_enrichment: bool, existing_facts: list[Fact]
    ) -> str:
        """Build a search query based on mode.

        For enrichment, includes existing facts so Perplexity can focus on
        information we don't already have.
        """
        if is_enrichment:
            if not existing_facts:
                return entity_name
            facts_text = "\n".join(f"- {f.content}" for f in existing_facts)
            return (
                f"Tell me more about {entity_name}. "
                f"I already know:\n{facts_text}\n\n"
                f"What else is important to know?"
            )
        year = datetime.now(UTC).year
        return f"{entity_name} latest news updates {year}"

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
        assert entity.id is not None

        # Build extraction prompt
        existing_context = ""
        if existing_facts:
            facts_text = "\n".join(f"- {f.content}" for f in existing_facts)
            existing_context = (
                f"\n\nAlready known facts (return only NEW facts not listed here):\n{facts_text}"
            )

        prompt = (
            f"{Prompt.ENTITY_FACT_EXTRACTION_PROMPT}\n\n"
            f"Entity: {entity.name}\n\n"
            f"Content:\n{search_text}"
            f"{existing_context}"
        )

        try:
            response = await self._background_model_client.generate(
                prompt=prompt,
                tools=None,
                format=ExtractedFacts.model_json_schema(),
            )
            extracted = ExtractedFacts.model_validate_json(response.content)
            candidate_facts = extracted.facts
        except Exception as e:
            logger.error("Failed to extract facts for '%s': %s", entity.name, e)
            return [], []

        if not candidate_facts:
            return [], []

        # Fast pass: normalized string dedup
        existing_normalized = {_normalize_fact(f.content): f for f in existing_facts}
        new_facts: list[str] = []
        confirmed_ids: list[int] = []
        seen_normalized: set[str] = set()

        for fact_text in candidate_facts:
            fact_text = fact_text.strip()
            if not fact_text:
                continue
            normalized = _normalize_fact(fact_text)
            if normalized in seen_normalized:
                continue
            seen_normalized.add(normalized)

            if normalized in existing_normalized:
                # Exact match → confirm existing fact
                existing_fact = existing_normalized[normalized]
                if existing_fact.id is not None:
                    confirmed_ids.append(existing_fact.id)
                continue
            new_facts.append(fact_text)

        if not new_facts:
            return [], confirmed_ids

        # Slow pass: embedding similarity dedup
        if not self._embedding_model_client:
            return new_facts, confirmed_ids

        facts_with_embeddings = [f for f in existing_facts if f.embedding is not None]
        if not facts_with_embeddings:
            return new_facts, confirmed_ids

        try:
            vecs = await self._embedding_model_client.embed(new_facts)
            existing_candidates = [
                (i, deserialize_embedding(f.embedding))
                for i, f in enumerate(facts_with_embeddings)
                if f.embedding is not None
            ]

            deduped: list[str] = []
            for fact_text, query_vec in zip(new_facts, vecs, strict=True):
                matches = find_similar(
                    query_vec,
                    existing_candidates,
                    top_k=1,
                    threshold=self.config.runtime.EXTRACTION_FACT_DEDUP_SIMILARITY_THRESHOLD,
                )
                if matches:
                    # Paraphrase match → confirm the existing fact
                    matched_idx = matches[0][0]
                    matched_fact = facts_with_embeddings[matched_idx]
                    if matched_fact.id is not None:
                        confirmed_ids.append(matched_fact.id)
                    logger.debug("Skipping duplicate fact (embedding match): %s", fact_text[:50])
                    continue
                deduped.append(fact_text)

            return deduped, confirmed_ids
        except Exception as e:
            logger.warning("Embedding dedup failed, keeping all candidates: %s", e)
            return new_facts, confirmed_ids

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
            fact = self.db.add_fact(
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
            facts = self.db.get_entity_facts(entity.id)
            text = build_entity_embed_text(entity.name, [f.content for f in facts])
            vecs = await self._embedding_model_client.embed(text)
            self.db.update_entity_embedding(entity.id, serialize_embedding(vecs[0]))
        except Exception as e:
            logger.warning("Failed to update entity embedding for '%s': %s", entity.name, e)

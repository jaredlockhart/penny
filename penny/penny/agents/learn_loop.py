"""Adaptive learn loop — background research driven by interest scores."""

from __future__ import annotations

import asyncio
import logging
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
    from penny.channels import MessageChannel
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


def _staleness_factor(facts: list[Fact], recent_days: float, staleness_days: float) -> float:
    """Compute staleness factor from a list of facts.

    Returns a multiplier:
    - 2.0 if no facts (needs enrichment)
    - 0.1 if most recent verification < recent_days ago
    - Scales linearly up to 3.0 cap based on days since last verification
    """
    if not facts:
        return 2.0

    now = datetime.now(UTC)
    latest = max(
        (f.last_verified or f.learned_at for f in facts),
        default=now,
    )
    # SQLite returns naive datetimes; treat as UTC
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=UTC)

    days_since = (now - latest).total_seconds() / 86400.0
    if days_since < recent_days:
        return 0.1
    return min(days_since / staleness_days, 3.0)


class LearnLoopAgent(Agent):
    """Background agent that adaptively researches entities based on interest scores.

    Picks the highest-priority entity across all users each cycle:
    - Enrichment mode (few facts): broad search to build knowledge
    - Briefing mode (many facts, stale): targeted search for new developments
    - Skip: if entity was recently verified or has negative interest

    Uses SearchTool for Perplexity searches and OllamaClient for fact extraction
    and message composition.
    """

    def __init__(self, search_tool: Tool | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._channel: MessageChannel | None = None
        self._search_tool = search_tool

    @property
    def name(self) -> str:
        """Task name for logging."""
        return "learn_loop"

    def set_channel(self, channel: MessageChannel) -> None:
        """Set the channel for sending proactive messages."""
        self._channel = channel

    async def execute(self) -> bool:
        """Run one cycle of the learn loop.

        Returns:
            True if work was done, False if nothing to research.
        """
        if not self._channel:
            logger.error("LearnLoopAgent: no channel set")
            return False

        if not self._search_tool:
            logger.debug("LearnLoopAgent: no search tool configured")
            return False

        # Score all entities and pick the highest-priority one
        candidates = self._score_candidates()
        if not candidates:
            logger.debug("LearnLoopAgent: no candidates to research")
            return False

        candidate = max(candidates, key=lambda c: c.priority)

        entity = candidate.entity
        user = candidate.user
        assert entity.id is not None

        # Determine mode
        is_enrichment = candidate.fact_count < self.config.learn_enrichment_fact_threshold

        mode_label = "enrichment" if is_enrichment else "briefing"
        logger.info(
            "Learn loop: %s mode for '%s' (user=%s, interest=%.2f, facts=%d, priority=%.3f)",
            mode_label,
            entity.name,
            user,
            candidate.interest,
            candidate.fact_count,
            candidate.priority,
        )

        # Build search query
        query = self._build_query(entity.name, is_enrichment)
        logger.info("Learn loop search query: '%s'", query)

        # Execute search
        search_result = await self._search(query)
        if search_result is None:
            return False

        # Extract and deduplicate facts
        new_facts, confirmed_fact_ids = await self._extract_and_dedup_facts(
            entity, candidate.facts, search_result
        )
        logger.info(
            "Learn loop extracted %d new facts, %d confirmed existing for '%s'",
            len(new_facts),
            len(confirmed_fact_ids),
            entity.name,
        )

        # Update last_verified on confirmed existing facts
        for fact_id in confirmed_fact_ids:
            self.db.update_fact_last_verified(fact_id)

        # Store new facts
        stored_facts = await self._store_new_facts(entity, new_facts, search_result)

        # Update entity embedding if we added facts
        if stored_facts and self.embedding_model:
            await self._update_entity_embedding(entity)
            logger.info("Updated entity embedding for '%s'", entity.name)

        # Send message if we found novel info
        if stored_facts:
            await self._send_findings_message(user, entity.name, stored_facts, is_enrichment)

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
                interest = compute_interest_score(entity_engagements)

                if interest < self.config.learn_min_interest_score:
                    continue

                facts = self.db.get_entity_facts(entity.id)
                fact_count = len(facts)
                staleness = _staleness_factor(
                    facts,
                    recent_days=self.config.learn_recent_days,
                    staleness_days=self.config.learn_staleness_days,
                )

                priority = interest * (1.0 / max(fact_count, 1)) * staleness

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

    def _build_query(self, entity_name: str, is_enrichment: bool) -> str:
        """Build a search query based on mode."""
        if is_enrichment:
            return entity_name
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
            logger.error("Learn loop search failed: %s", e)
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
            response = await self._ollama_client.generate(
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
        if not self.embedding_model:
            return new_facts, confirmed_ids

        facts_with_embeddings = [f for f in existing_facts if f.embedding is not None]
        if not facts_with_embeddings:
            return new_facts, confirmed_ids

        try:
            vecs = await self._ollama_client.embed(new_facts, model=self.embedding_model)
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
                    threshold=self.config.extraction_fact_dedup_similarity_threshold,
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
        if self.embedding_model:
            try:
                vecs = await self._ollama_client.embed(new_fact_texts, model=self.embedding_model)
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
                logger.info("Learn loop +fact for '%s': %s", entity.name, fact_text)

        return stored

    async def _update_entity_embedding(self, entity: Entity) -> None:
        """Regenerate entity embedding after adding new facts."""
        assert entity.id is not None
        try:
            assert self.embedding_model is not None
            facts = self.db.get_entity_facts(entity.id)
            text = build_entity_embed_text(entity.name, [f.content for f in facts])
            vecs = await self._ollama_client.embed(text, model=self.embedding_model)
            self.db.update_entity_embedding(entity.id, serialize_embedding(vecs[0]))
        except Exception as e:
            logger.warning("Failed to update entity embedding for '%s': %s", entity.name, e)

    async def _send_findings_message(
        self,
        user: str,
        entity_name: str,
        new_facts: list[str],
        is_enrichment: bool,
    ) -> None:
        """Compose and send a proactive message about findings."""
        assert self._channel is not None

        # Build prompt for message composition
        facts_text = "\n".join(f"- {f}" for f in new_facts)

        if is_enrichment:
            prompt_template = Prompt.LEARN_ENRICHMENT_MESSAGE_PROMPT
        else:
            prompt_template = Prompt.LEARN_BRIEFING_MESSAGE_PROMPT

        prompt = f"{prompt_template.format(entity_name=entity_name)}\n\nNew facts:\n{facts_text}"

        # Inject a fake user turn so the model understands it's responding to the
        # user's interest rather than composing a message into the void.
        history = [("user", f"what's new with {entity_name}?")]
        result = await self._compose_user_facing(prompt, history=history, image_query=entity_name)
        if not result.answer:
            return

        attachments = result.attachments or None
        typing_task = asyncio.create_task(self._channel._typing_loop(user))
        try:
            await self._channel.send_response(
                user,
                result.answer,
                parent_id=None,  # Unsolicited, not threaded
                attachments=attachments,
            )
            logger.info(
                "Learn loop sent %s message about '%s' to %s",
                "enrichment" if is_enrichment else "briefing",
                entity_name,
                user,
            )
        finally:
            typing_task.cancel()
            await self._channel.send_typing(user, False)

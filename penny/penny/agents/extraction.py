"""Unified extraction pipeline for entities, facts, and engagements."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from penny.agents.base import Agent
from penny.constants import PennyConstants
from penny.database.models import Entity, Fact
from penny.ollama.embeddings import (
    build_entity_embed_text,
    cosine_similarity,
    deserialize_embedding,
    find_similar,
    serialize_embedding,
    token_containment_ratio,
    tokenize_entity_name,
)
from penny.prompts import Prompt

if TYPE_CHECKING:
    from penny.channels import MessageChannel
    from penny.database.models import MessageLog

logger = logging.getLogger(__name__)

# Pattern to collapse whitespace and strip bullet prefixes for fact comparison
_WHITESPACE_RE = re.compile(r"\s+")

# --- Entity name validation patterns ---
_LLM_ARTIFACT_PATTERNS = (
    "{topic}",
    "{description}",
    "{desccription}",
    "confidence score:",
    "-brief:",
    "fill_me_in",
    "placeholder",
    "<example>",
    "todo",
)
_NUMBERED_LIST_RE = re.compile(r"^\d+\.")
_URL_RE = re.compile(r"https?://")
_MARKDOWN_BOLD_RE = re.compile(r"\*\*")
_ANGLE_BRACKET_RE = re.compile(r"<[^>]+>")
_ENTITY_NAME_MAX_WORDS = 8


def _normalize_fact(fact: str) -> str:
    """Normalize a fact string for dedup comparison.

    Strips leading '- ', lowercases, and collapses whitespace so that
    near-duplicate facts with minor formatting differences are caught.
    """
    text = fact.strip().lstrip("-").strip()
    return _WHITESPACE_RE.sub(" ", text).lower()


def _is_valid_entity_name(name: str) -> bool:
    """Structural validation for entity name candidates.

    Rejects names that are clearly not entity names: too long, contain LLM
    output artifacts, numbered list items, URLs, markdown, or newlines.
    """
    if len(name.split()) > _ENTITY_NAME_MAX_WORDS:
        return False
    name_lower = name.lower()
    if any(artifact in name_lower for artifact in _LLM_ARTIFACT_PATTERNS):
        return False
    if _NUMBERED_LIST_RE.match(name.strip()):
        return False
    return not (
        _URL_RE.search(name)
        or _MARKDOWN_BOLD_RE.search(name)
        or _ANGLE_BRACKET_RE.search(name)
        or "\n" in name
    )


# --- Pydantic schemas for LLM responses ---


class IdentifiedNewEntity(BaseModel):
    """A newly discovered entity from pass 1."""

    name: str = Field(description="Entity name (e.g., 'KEF LS50 Meta', 'NVIDIA Jetson')")


class IdentifiedEntities(BaseModel):
    """Schema for pass 1: which known and new entities appear in the text."""

    known: list[str] = Field(
        default_factory=list,
        description="Names of already-known entities that appear in this text",
    )
    new: list[IdentifiedNewEntity] = Field(
        default_factory=list,
        description="New entities found in this text (not in the known list)",
    )


class IdentifiedKnownEntities(BaseModel):
    """Schema for known-only mode: which known entities appear in the text."""

    known: list[str] = Field(
        default_factory=list,
        description="Names of already-known entities that appear in this text",
    )


class ExtractedFacts(BaseModel):
    """Schema for pass 2: new facts about a single entity."""

    facts: list[str] = Field(
        default_factory=list,
        description="NEW specific, verifiable facts about the entity from the text",
    )


@dataclass
class _EntityFactDiscovery:
    """Facts discovered for a single entity during extraction."""

    entity: Entity
    new_facts: list[str]
    is_new_entity: bool


@dataclass
class _ExtractionResult:
    """Result of _extract_and_store_entities with discovery info for notifications."""

    entities: list[Entity] = field(default_factory=list)
    discoveries: list[_EntityFactDiscovery] = field(default_factory=list)


class _UserBackoff:
    """Per-user backoff state for proactive fact notifications."""

    __slots__ = ("last_proactive_send", "backoff_seconds")

    def __init__(self) -> None:
        self.last_proactive_send: datetime | None = None
        self.backoff_seconds: float = 0.0


class ExtractionPipeline(Agent):
    """Unified background agent that extracts entities, facts, preferences, and engagements.

    Replaces the separate EntityExtractor and PreferenceAgent with a single pipeline
    that processes both SearchLog and MessageLog entries.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._channel: MessageChannel | None = None
        self._backoff_state: dict[str, _UserBackoff] = {}

    @property
    def name(self) -> str:
        """Task name for logging."""
        return "extraction"

    def set_channel(self, channel: MessageChannel) -> None:
        """Set the channel for sending preference notifications."""
        self._channel = channel

    async def execute(self) -> bool:
        """Process unextracted search logs, then unprocessed messages, then backfill.

        Returns:
            True if any work was done.
        """
        work_done = False

        # Phase 1: Extract entities/facts from search results (newest first)
        work_done |= await self._process_search_logs()

        # Phase 2: Extract entities/facts/preferences from messages (newest first)
        work_done |= await self._process_messages()

        # Phase 3: Backfill embeddings for items that don't have them
        if self.embedding_model:
            work_done |= await self._backfill_embeddings()

        return work_done

    # --- Phase 1: Search log processing ---

    async def _process_search_logs(self) -> bool:
        """Process unextracted SearchLog entries for entity/fact extraction."""
        search_logs = self.db.get_unprocessed_search_logs(
            limit=PennyConstants.ENTITY_EXTRACTION_BATCH_LIMIT
        )
        if not search_logs:
            return False

        work_done = False
        logger.info("Processing %d unprocessed search logs", len(search_logs))

        for search_log in search_logs:
            assert search_log.id is not None

            user = self.db.find_sender_for_timestamp(search_log.timestamp)
            if not user:
                self.db.mark_search_extracted(search_log.id)
                continue

            allow_new = search_log.trigger != PennyConstants.SearchTrigger.PENNY_ENRICHMENT
            id_prompt = (
                Prompt.ENTITY_IDENTIFICATION_PROMPT
                if allow_new
                else Prompt.KNOWN_ENTITY_IDENTIFICATION_PROMPT
            )

            # Resolve learn prompt text for semantic relevance gate
            relevance_ref: str | None = None
            if search_log.learn_prompt_id is not None:
                learn_prompt = self.db.get_learn_prompt(search_log.learn_prompt_id)
                if learn_prompt:
                    relevance_ref = learn_prompt.prompt_text

            logger.info(
                "Extracting entities from search: %s (mode=%s)",
                search_log.query,
                "full" if allow_new else "known-only",
            )
            result = await self._extract_and_store_entities(
                user=user,
                identification_prompt=id_prompt,
                fact_prompt=Prompt.ENTITY_FACT_EXTRACTION_PROMPT,
                context_label="Search query",
                context_value=search_log.query,
                content=search_log.response,
                source_search_log_id=search_log.id,
                allow_new_entities=allow_new,
                relevance_reference=relevance_ref,
                record_discovery_score=True,
            )
            if result.entities:
                work_done = True

                # Create SEARCH_INITIATED engagements only for user-triggered searches
                if allow_new:
                    for entity in result.entities:
                        assert entity.id is not None
                        self.db.add_engagement(
                            user=user,
                            engagement_type=PennyConstants.EngagementType.SEARCH_INITIATED,
                            valence=PennyConstants.EngagementValence.POSITIVE,
                            strength=PennyConstants.ENGAGEMENT_STRENGTH_SEARCH_INITIATED,
                            entity_id=entity.id,
                        )

            # Send per-entity discovery notifications (respecting backoff)
            if allow_new and result.discoveries and self._should_send_proactive(user):
                for discovery in result.discoveries:
                    await self._send_fact_notification(user, discovery, relevance_ref)
                self._mark_proactive_sent(user)

            self.db.mark_search_extracted(search_log.id)

        return work_done

    # --- Phase 2: Message processing ---

    async def _process_messages(self) -> bool:
        """Process unprocessed messages for entity/fact extraction."""
        senders = self.db.get_all_senders()
        if not senders:
            return False

        work_done = False

        for sender in senders:
            messages = self.db.get_unprocessed_messages(
                sender, limit=PennyConstants.PREFERENCE_BATCH_LIMIT
            )

            if not messages:
                continue

            logger.info(
                "Processing messages for %s: %d messages",
                sender,
                len(messages),
            )

            # --- Entity/fact extraction from messages ---
            for message in messages:
                if not self._should_process_message(message):
                    continue

                assert message.id is not None
                result = await self._extract_and_store_entities(
                    user=sender,
                    identification_prompt=Prompt.MESSAGE_ENTITY_IDENTIFICATION_PROMPT,
                    fact_prompt=Prompt.MESSAGE_FACT_EXTRACTION_PROMPT,
                    context_label="User message",
                    context_value=message.content,
                    content=message.content,
                    source_message_id=message.id,
                )
                if result.entities:
                    work_done = True

                    # Create MESSAGE_MENTION engagements for identified entities
                    for entity in result.entities:
                        assert entity.id is not None
                        self.db.add_engagement(
                            user=sender,
                            engagement_type=PennyConstants.EngagementType.MESSAGE_MENTION,
                            valence=PennyConstants.EngagementValence.NEUTRAL,
                            strength=PennyConstants.ENGAGEMENT_STRENGTH_MESSAGE_MENTION,
                            entity_id=entity.id,
                            source_message_id=message.id,
                        )

            # --- Follow-up detection (separate from entity extraction, no min-length filter) ---
            for message in messages:
                created = await self._create_follow_up_engagements(sender, message)
                if created:
                    work_done = True

            # Mark messages as processed
            message_ids = [m.id for m in messages if m.id is not None]
            if message_ids:
                self.db.mark_messages_processed(message_ids)

            logger.info("Finished processing messages for %s", sender)

        return work_done

    # --- Entity/fact extraction (shared by search logs and messages) ---

    async def _extract_and_store_entities(
        self,
        user: str,
        identification_prompt: str,
        fact_prompt: str,
        context_label: str,
        context_value: str,
        content: str,
        source_search_log_id: int | None = None,
        source_message_id: int | None = None,
        allow_new_entities: bool = True,
        relevance_reference: str | None = None,
        record_discovery_score: bool = False,
    ) -> _ExtractionResult:
        """Two-pass extraction for a single piece of content.

        Pass 1: Identify entities in the text.
          - Full mode (allow_new_entities=True): known + new entities
          - Known-only mode (allow_new_entities=False): only known entities
        Pass 2: For each entity, extract new facts as individual Fact rows.

        Args:
            relevance_reference: Text to compare entity names against for semantic
                relevance gating. Defaults to context_value when not provided.
                For /learn searches, this should be the original learn prompt text
                rather than the intermediate search query.

        Returns _ExtractionResult with all identified entities and per-entity discoveries.
        """
        existing_entities = self.db.get_user_entities(user)

        # Collect all entities to process facts for
        entities_to_process: list[Entity] = []
        newly_created_names: set[str] = set()
        existing_by_name = {e.name: e for e in existing_entities}

        # Pre-filter entities by embedding similarity to reduce LLM prompt size
        filtered_entities = await self._prefilter_entities_by_similarity(existing_entities, content)

        if allow_new_entities:
            # Full mode: identify known + new entities
            identified = await self._identify_entities(
                filtered_entities, identification_prompt, context_label, context_value, content
            )
            if not identified:
                return _ExtractionResult()

            # Filter out fragment entities that are token-subsets of another
            # candidate in the same batch (e.g., "totem" when "totem loon" is also present)
            candidate_names = [e.name.lower().strip() for e in identified.new if e.name.strip()]
            candidate_token_sets = {n: set(tokenize_entity_name(n)) for n in candidate_names}
            subset_names: set[str] = set()
            for name_a, tokens_a in candidate_token_sets.items():
                if not tokens_a:
                    continue
                for name_b, tokens_b in candidate_token_sets.items():
                    if name_a == name_b:
                        continue
                    if tokens_a < tokens_b:  # strict subset
                        subset_names.add(name_a)
                        break

            # Validate and store new entities
            for new_entity in identified.new:
                name = new_entity.name.lower().strip()
                if not name:
                    continue
                if not _is_valid_entity_name(name):
                    logger.info("Rejected entity '%s' (structural filter)", name)
                    continue
                if name in subset_names:
                    logger.info("Rejected entity '%s' (token-subset of another candidate)", name)
                    continue
                (
                    relevant,
                    candidate_embedding,
                    similarity_score,
                ) = await self._check_semantic_relevance(name, relevance_reference or context_value)
                if not relevant:
                    continue

                # Check for duplicate before creating
                duplicate = self._find_duplicate_entity(
                    name, candidate_embedding, existing_entities
                )
                if duplicate:
                    resolved_entity = duplicate
                else:
                    is_new = name not in existing_by_name
                    resolved_entity = self.db.get_or_create_entity(user, name)
                    if not resolved_entity or resolved_entity.id is None:
                        continue
                    if is_new:
                        newly_created_names.add(resolved_entity.name)
                    logger.info("New entity discovered: '%s'", name)

                entities_to_process.append(resolved_entity)

                if record_discovery_score and similarity_score > 0.0:
                    assert resolved_entity.id is not None
                    self.db.add_engagement(
                        user=user,
                        engagement_type=PennyConstants.EngagementType.SEARCH_DISCOVERY,
                        valence=PennyConstants.EngagementValence.POSITIVE,
                        strength=similarity_score,
                        entity_id=resolved_entity.id,
                    )

            # Look up known entities that were identified
            for known_name in identified.known:
                normalized = known_name.lower().strip()
                if normalized in existing_by_name:
                    entities_to_process.append(existing_by_name[normalized])
                    logger.info("Known entity referenced: '%s'", normalized)
        else:
            # Known-only mode: only match against existing entities
            known_result = await self._identify_known_entities(
                filtered_entities, identification_prompt, context_label, context_value, content
            )
            if not known_result:
                return _ExtractionResult()

            for known_name in known_result.known:
                normalized = known_name.lower().strip()
                if normalized in existing_by_name:
                    entities_to_process.append(existing_by_name[normalized])
                    logger.info("Known entity referenced: '%s'", normalized)

        # Pass 2: extract facts for each identified entity
        entities_with_new_facts: list[int] = []
        discoveries: list[_EntityFactDiscovery] = []
        for entity in entities_to_process:
            assert entity.id is not None
            new_facts = await self._extract_facts(
                entity, fact_prompt, context_label, context_value, content
            )
            if not new_facts:
                continue

            # Dedup against existing Fact rows
            existing_fact_rows = self.db.get_entity_facts(entity.id)
            new_fact_texts = await self._dedup_facts(new_facts, existing_fact_rows)

            if not new_fact_texts:
                continue

            # Batch-embed all new facts at once
            fact_embeddings: list[bytes | None] = [None] * len(new_fact_texts)
            if self.embedding_model:
                try:
                    vecs = await self._ollama_client.embed(
                        new_fact_texts, model=self.embedding_model
                    )
                    fact_embeddings = [serialize_embedding(v) for v in vecs]
                except Exception as e:
                    logger.warning("Failed to embed facts for '%s': %s", entity.name, e)

            # Store facts with embeddings
            for fact_text, emb in zip(new_fact_texts, fact_embeddings, strict=True):
                self.db.add_fact(
                    entity_id=entity.id,
                    content=fact_text,
                    source_search_log_id=source_search_log_id,
                    source_message_id=source_message_id,
                    embedding=emb,
                )
                logger.info("  '%s' +fact: %s", entity.name, fact_text)

            entities_with_new_facts.append(entity.id)
            discovery = _EntityFactDiscovery(
                entity=entity,
                new_facts=new_fact_texts,
                is_new_entity=entity.name in newly_created_names,
            )
            discoveries.append(discovery)

        # Regenerate entity embeddings for entities that got new facts
        if self.embedding_model and entities_with_new_facts:
            await self._update_entity_embeddings(
                [e for e in entities_to_process if e.id in entities_with_new_facts]
            )

        return _ExtractionResult(entities=entities_to_process, discoveries=discoveries)

    async def _identify_entities(
        self,
        existing_entities: list[Entity],
        identification_prompt: str,
        context_label: str,
        context_value: str,
        content: str,
    ) -> IdentifiedEntities | None:
        """Pass 1: Identify which known entities appear in the text and any new entities."""
        known_names = [e.name for e in existing_entities]
        known_context = ""
        if known_names:
            known_context = (
                "\n\nKnown entities (return any that appear in the text):\n"
                + "\n".join(f"- {name}" for name in known_names)
            )

        prompt = (
            f"{identification_prompt}\n\n"
            f"{context_label}: {context_value}\n\n"
            f"Content:\n{content}"
            f"{known_context}"
        )

        try:
            response = await self._ollama_client.generate(
                prompt=prompt,
                tools=None,
                format=IdentifiedEntities.model_json_schema(),
            )
            result = IdentifiedEntities.model_validate_json(response.content)
            if not result.known and not result.new:
                return None
            return result
        except Exception as e:
            logger.error("Failed to identify entities: %s", e)
            return None

    async def _identify_known_entities(
        self,
        existing_entities: list[Entity],
        identification_prompt: str,
        context_label: str,
        context_value: str,
        content: str,
    ) -> IdentifiedKnownEntities | None:
        """Known-only identification: only match against existing entities, never create new."""
        known_names = [e.name for e in existing_entities]
        if not known_names:
            return None

        known_context = "\n\nKnown entities (return any that appear in the text):\n" + "\n".join(
            f"- {name}" for name in known_names
        )

        prompt = (
            f"{identification_prompt}\n\n"
            f"{context_label}: {context_value}\n\n"
            f"Content:\n{content}"
            f"{known_context}"
        )

        try:
            response = await self._ollama_client.generate(
                prompt=prompt,
                tools=None,
                format=IdentifiedKnownEntities.model_json_schema(),
            )
            result = IdentifiedKnownEntities.model_validate_json(response.content)
            if not result.known:
                return None
            return result
        except Exception as e:
            logger.error("Failed to identify known entities: %s", e)
            return None

    async def _prefilter_entities_by_similarity(
        self,
        entities: list[Entity],
        content: str,
    ) -> list[Entity]:
        """Pre-filter entities by embedding similarity to content.

        Reduces the entity list sent to the LLM prompt by keeping only entities
        whose embeddings are similar to the content being analyzed. Skips
        filtering when the entity count is below ENTITY_PREFILTER_MIN_COUNT
        or when no embedding model is configured.
        """
        if not self.embedding_model:
            return entities
        if len(entities) < PennyConstants.ENTITY_PREFILTER_MIN_COUNT:
            return entities

        entities_with_embeddings = [e for e in entities if e.embedding is not None]
        entities_without_embeddings = [e for e in entities if e.embedding is None]

        if not entities_with_embeddings:
            return entities

        try:
            content_vecs = await self._ollama_client.embed(content, model=self.embedding_model)
            content_vec = content_vecs[0]

            candidates = [
                (i, deserialize_embedding(e.embedding))
                for i, e in enumerate(entities_with_embeddings)
                if e.embedding is not None
            ]

            matches = find_similar(
                content_vec,
                candidates,
                top_k=len(candidates),
                threshold=PennyConstants.ENTITY_PREFILTER_SIMILARITY_THRESHOLD,
            )

            matched_indices = {idx for idx, _score in matches}
            filtered = [entities_with_embeddings[i] for i in matched_indices]
            filtered.extend(entities_without_embeddings)

            logger.info(
                "Pre-filtered entities: %d -> %d (from %d with embeddings, %d without)",
                len(entities),
                len(filtered),
                len(entities_with_embeddings),
                len(entities_without_embeddings),
            )

            return filtered
        except Exception as e:
            logger.warning("Entity pre-filter failed, using full list: %s", e)
            return entities

    async def _check_semantic_relevance(
        self, candidate_name: str, trigger_text: str
    ) -> tuple[bool, list[float] | None, float]:
        """Semantic validation: reject entity candidates unrelated to the triggering content.

        Returns (is_relevant, candidate_embedding, similarity_score). The candidate
        embedding is returned so callers can reuse it for dedup without a second embed
        call. The similarity score is returned so callers can record it as engagement
        strength.
        """
        if not self.embedding_model:
            return True, None, 0.0
        try:
            vecs = await self._ollama_client.embed(
                [candidate_name, trigger_text], model=self.embedding_model
            )
            candidate_embedding = vecs[0]
            score = cosine_similarity(candidate_embedding, vecs[1])
            if score < PennyConstants.ENTITY_NAME_SEMANTIC_THRESHOLD:
                logger.info(
                    "Rejected entity '%s' (similarity %.2f < %.2f)",
                    candidate_name,
                    score,
                    PennyConstants.ENTITY_NAME_SEMANTIC_THRESHOLD,
                )
                return False, None, 0.0
            return True, candidate_embedding, score
        except Exception:
            logger.debug("Semantic validation failed, accepting candidate", exc_info=True)
            return True, None, 0.0

    def _find_duplicate_entity(
        self,
        candidate_name: str,
        candidate_embedding: list[float] | None,
        existing_entities: list[Entity],
    ) -> Entity | None:
        """Check if a candidate entity name is a duplicate of an existing entity.

        Uses dual-threshold detection: token containment ratio (TCR) as a fast
        lexical pre-filter, then embedding cosine similarity for confirmation.
        Both signals must pass for a match.
        """
        if candidate_embedding is None:
            return None

        for entity in existing_entities:
            if entity.embedding is None:
                continue
            tcr = token_containment_ratio(candidate_name, entity.name)
            if tcr < PennyConstants.ENTITY_DEDUP_TCR_THRESHOLD:
                continue
            entity_vec = deserialize_embedding(entity.embedding)
            sim = cosine_similarity(candidate_embedding, entity_vec)
            if sim >= PennyConstants.ENTITY_DEDUP_EMBEDDING_THRESHOLD:
                logger.info(
                    "Dedup: '%s' matches existing '%s' (TCR=%.2f, sim=%.2f)",
                    candidate_name,
                    entity.name,
                    tcr,
                    sim,
                )
                return entity
        return None

    async def _extract_facts(
        self,
        entity: Entity,
        fact_prompt: str,
        context_label: str,
        context_value: str,
        content: str,
    ) -> list[str]:
        """Pass 2: Extract new facts about a specific entity from content."""
        assert entity.id is not None
        existing_fact_rows = self.db.get_entity_facts(entity.id)
        existing_context = ""
        if existing_fact_rows:
            facts_text = "\n".join(f"- {f.content}" for f in existing_fact_rows)
            existing_context = (
                f"\n\nAlready known facts (return only NEW facts not listed here):\n{facts_text}"
            )

        prompt = (
            f"{fact_prompt}\n\n"
            f"Entity: {entity.name}\n\n"
            f"{context_label}: {context_value}\n\n"
            f"Content:\n{content}"
            f"{existing_context}"
        )

        try:
            response = await self._ollama_client.generate(
                prompt=prompt,
                tools=None,
                format=ExtractedFacts.model_json_schema(),
            )
            result = ExtractedFacts.model_validate_json(response.content)
            return result.facts
        except Exception as e:
            logger.error("Failed to extract facts for '%s': %s", entity.name, e)
            return []

    async def _dedup_facts(self, new_facts: list[str], existing_facts: list[Fact]) -> list[str]:
        """Deduplicate new facts against existing ones.

        Uses normalized string match as a fast first pass, then embedding
        similarity for paraphrase detection.
        """
        existing_normalized = {_normalize_fact(f.content) for f in existing_facts}

        # Fast pass: normalized string dedup
        candidates: list[str] = []
        for fact_text in new_facts:
            fact_text = fact_text.strip()
            if not fact_text:
                continue
            normalized = _normalize_fact(fact_text)
            if normalized in existing_normalized:
                continue
            candidates.append(fact_text)
            existing_normalized.add(normalized)

        if not candidates:
            return []

        # Slow pass: embedding similarity dedup
        if not self.embedding_model:
            return candidates

        facts_with_embeddings = [f for f in existing_facts if f.embedding is not None]
        if not facts_with_embeddings:
            return candidates

        try:
            vecs = await self._ollama_client.embed(candidates, model=self.embedding_model)
            existing_candidates = [
                (i, deserialize_embedding(f.embedding))
                for i, f in enumerate(facts_with_embeddings)
                if f.embedding is not None
            ]

            deduped: list[str] = []
            for fact_text, query_vec in zip(candidates, vecs, strict=True):
                matches = find_similar(
                    query_vec,
                    existing_candidates,
                    top_k=1,
                    threshold=PennyConstants.FACT_DEDUP_SIMILARITY_THRESHOLD,
                )
                if matches:
                    logger.debug("Skipping duplicate fact (embedding match): %s", fact_text[:50])
                    continue
                deduped.append(fact_text)

            return deduped
        except Exception as e:
            logger.warning("Embedding dedup failed, keeping all candidates: %s", e)
            return candidates

    # --- Message filtering ---

    @staticmethod
    def _should_process_message(message: MessageLog) -> bool:
        """Lightweight pre-filter to skip low-signal messages before LLM calls."""
        content = message.content.strip()
        if len(content) < PennyConstants.MIN_EXTRACTION_MESSAGE_LENGTH:
            return False
        return not content.startswith("/")

    # --- Follow-up detection ---

    async def _create_follow_up_engagements(self, sender: str, message: MessageLog) -> bool:
        """Create FOLLOW_UP_QUESTION engagements when user replies to Penny's message.

        Embeds the parent message content and finds similar entities to determine
        which entities the user is following up on.

        Returns:
            True if any engagements were created.
        """
        if not message.parent_id or not self.embedding_model:
            return False

        parent_msg = self.db.get_message_by_id(message.parent_id)
        if not parent_msg:
            return False
        if parent_msg.direction != PennyConstants.MessageDirection.OUTGOING:
            return False

        entities = self.db.get_user_entities_with_embeddings(sender)
        if not entities:
            return False

        candidates = [
            (e.id, deserialize_embedding(e.embedding)) for e in entities if e.id and e.embedding
        ]
        if not candidates:
            return False

        try:
            vecs = await self._ollama_client.embed(parent_msg.content, model=self.embedding_model)
            query_vec = vecs[0]
            matches = find_similar(
                query_vec,
                candidates,
                top_k=PennyConstants.ENTITY_CONTEXT_TOP_K,
                threshold=PennyConstants.ENTITY_CONTEXT_THRESHOLD,
            )

            for entity_id, _score in matches:
                self.db.add_engagement(
                    user=sender,
                    engagement_type=PennyConstants.EngagementType.FOLLOW_UP_QUESTION,
                    valence=PennyConstants.EngagementValence.POSITIVE,
                    strength=PennyConstants.ENGAGEMENT_STRENGTH_FOLLOW_UP_QUESTION,
                    entity_id=entity_id,
                    source_message_id=message.id,
                )

            return len(matches) > 0
        except Exception:
            logger.debug("Follow-up engagement extraction failed", exc_info=True)
            return False

    # --- Fact discovery notifications ---

    def _should_send_proactive(self, user: str) -> bool:
        """Check if we should send proactive notifications to this user (backoff check)."""
        state = self._backoff_state.get(user)
        if state is None:
            return True

        # Check if user has sent a message since our last proactive send
        if state.last_proactive_send is not None:
            latest_incoming = self.db.get_latest_incoming_message_time(user)
            if latest_incoming is not None:
                incoming_time = latest_incoming
                if incoming_time.tzinfo is None:
                    incoming_time = incoming_time.replace(tzinfo=UTC)
                last_send = state.last_proactive_send
                if last_send.tzinfo is None:
                    last_send = last_send.replace(tzinfo=UTC)
                if incoming_time > last_send:
                    state.backoff_seconds = 0.0

        if state.backoff_seconds <= 0:
            return True

        if state.last_proactive_send is None:
            return True

        now = datetime.now(UTC)
        last_send = state.last_proactive_send
        if last_send.tzinfo is None:
            last_send = last_send.replace(tzinfo=UTC)
        elapsed = (now - last_send).total_seconds()
        return elapsed >= state.backoff_seconds

    def _mark_proactive_sent(self, user: str) -> None:
        """Record that we sent proactive notifications and increase backoff."""
        state = self._backoff_state.get(user)
        if state is None:
            state = _UserBackoff()
            self._backoff_state[user] = state

        state.last_proactive_send = datetime.now(UTC)
        if state.backoff_seconds <= 0:
            state.backoff_seconds = PennyConstants.FACT_NOTIFICATION_INITIAL_BACKOFF
        else:
            state.backoff_seconds = min(
                state.backoff_seconds * 2,
                PennyConstants.FACT_NOTIFICATION_MAX_BACKOFF,
            )

    async def _send_fact_notification(
        self,
        user: str,
        discovery: _EntityFactDiscovery,
        learn_prompt_text: str | None = None,
    ) -> None:
        """Compose and send a notification for a single entity's newly discovered facts."""
        if not self._channel:
            return

        facts_text = "\n".join(f"- {fact}" for fact in discovery.new_facts)
        if discovery.is_new_entity:
            prompt_template = Prompt.FACT_DISCOVERY_NEW_ENTITY_PROMPT
        else:
            prompt_template = Prompt.FACT_DISCOVERY_KNOWN_ENTITY_PROMPT

        prompt = (
            f"{prompt_template.format(entity_name=discovery.entity.name)}"
            f"\n\nNew facts:\n{facts_text}"
        )

        # Inject the learn prompt as a prior user turn so the model understands
        # it's responding to the user's request rather than writing into the void.
        history = [("user", learn_prompt_text)] if learn_prompt_text else None
        result = await self._compose_user_facing(
            prompt, history=history, image_query=discovery.entity.name
        )
        if not result.answer:
            return
        if len(result.answer) < PennyConstants.FACT_NOTIFICATION_MIN_LENGTH:
            logger.debug(
                "Skipping near-empty notification (%d chars): %r",
                len(result.answer),
                result.answer,
            )
            return

        await self._send_notification(user, result.answer, attachments=result.attachments)

    # --- General notifications ---

    async def _send_notification(
        self, recipient: str, message: str, attachments: list[str] | None = None
    ) -> None:
        """Send a notification message to the user."""
        if not self._channel:
            return

        typing_task = asyncio.create_task(self._channel._typing_loop(recipient))
        try:
            await self._channel.send_response(
                recipient,
                message,
                parent_id=None,
                attachments=attachments,
            )
        finally:
            typing_task.cancel()
            await self._channel.send_typing(recipient, False)

    # --- Entity embedding updates ---

    async def _update_entity_embeddings(self, entities: list[Entity]) -> None:
        """Regenerate embeddings for entities by composing name + facts."""
        assert self.embedding_model is not None
        if not entities:
            return

        embed_texts: list[str] = []
        for entity in entities:
            assert entity.id is not None
            facts = self.db.get_entity_facts(entity.id)
            embed_texts.append(build_entity_embed_text(entity.name, [f.content for f in facts]))

        try:
            vecs = await self._ollama_client.embed(embed_texts, model=self.embedding_model)
            for entity, vec in zip(entities, vecs, strict=True):
                assert entity.id is not None
                self.db.update_entity_embedding(entity.id, serialize_embedding(vec))
                logger.debug("Updated entity embedding for '%s'", entity.name)
        except Exception as e:
            logger.warning("Failed to update entity embeddings: %s", e)

    # --- Embedding backfill ---

    async def _backfill_embeddings(self) -> bool:
        """Backfill embeddings for entities, facts, and preferences that don't have them.

        Returns:
            True if any embeddings were generated.
        """
        assert self.embedding_model is not None
        work_done = False
        batch_limit = PennyConstants.EMBEDDING_BACKFILL_BATCH_LIMIT

        # Backfill fact embeddings
        facts = self.db.get_facts_without_embeddings(limit=batch_limit)
        if facts:
            fact_texts = [f.content for f in facts]
            try:
                vecs = await self._ollama_client.embed(fact_texts, model=self.embedding_model)
                for fact, vec in zip(facts, vecs, strict=True):
                    assert fact.id is not None
                    self.db.update_fact_embedding(fact.id, serialize_embedding(vec))
                logger.info("Backfilled embeddings for %d facts", len(facts))
                work_done = True
            except Exception as e:
                logger.warning("Failed to backfill fact embeddings: %s", e)

        # Backfill entity embeddings
        entities = self.db.get_entities_without_embeddings(limit=batch_limit)
        if entities:
            embed_texts: list[str] = []
            for entity in entities:
                assert entity.id is not None
                facts_for_entity = self.db.get_entity_facts(entity.id)
                embed_texts.append(
                    build_entity_embed_text(entity.name, [f.content for f in facts_for_entity])
                )
            try:
                vecs = await self._ollama_client.embed(embed_texts, model=self.embedding_model)
                for entity, vec in zip(entities, vecs, strict=True):
                    assert entity.id is not None
                    self.db.update_entity_embedding(entity.id, serialize_embedding(vec))
                logger.info("Backfilled embeddings for %d entities", len(entities))
                work_done = True
            except Exception as e:
                logger.warning("Failed to backfill entity embeddings: %s", e)

        # Backfill preference embeddings
        prefs = self.db.get_preferences_without_embeddings(limit=batch_limit)
        if prefs:
            topics = [p.topic for p in prefs]
            try:
                vecs = await self._ollama_client.embed(topics, model=self.embedding_model)
                for pref, vec in zip(prefs, vecs, strict=True):
                    assert pref.id is not None
                    self.db.update_preference_embedding(pref.id, serialize_embedding(vec))
                logger.info("Backfilled embeddings for %d preferences", len(prefs))
                work_done = True
            except Exception as e:
                logger.warning("Failed to backfill preference embeddings: %s", e)

        return work_done

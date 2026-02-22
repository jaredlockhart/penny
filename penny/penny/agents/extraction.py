"""Unified extraction pipeline for entities, facts, and engagements."""

from __future__ import annotations

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
    from penny.agents.learn import LearnAgent
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
_BARE_NUMBER_RE = re.compile(r"\d[\d.,]*")
_DATE_RE = re.compile(
    r"(?:"
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"  # 2024-01-15, 2024/1/15
    r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"  # 01/15/2024, 1-15-24
    r"|Q[1-4]\s*\d{4}"  # Q1 2025
    r")$",
    re.IGNORECASE,
)
_MONTH_NAMES = frozenset(
    {
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    }
)
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
    if _BARE_NUMBER_RE.fullmatch(name.strip()):
        return False
    stripped = name.strip()
    if _DATE_RE.fullmatch(stripped):
        return False
    name_lower = name.lower()
    # Standalone month names (e.g. "January", "Feb")
    if name_lower.strip() in _MONTH_NAMES:
        return False
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
    tagline: str = Field(
        default="",
        description="Short 3-8 word summary describing what the entity is "
        "(e.g., 'bookshelf speaker by kef', 'british progressive rock band')",
    )


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


class EntitySentiment(BaseModel):
    """A single entity's sentiment from a user message."""

    entity_name: str = Field(description="Name of the entity")
    sentiment: str = Field(description="'positive' or 'negative'")


class MessageSentiments(BaseModel):
    """Schema for sentiment extraction from user messages."""

    sentiments: list[EntitySentiment] = Field(
        default_factory=list,
        description="Entities with non-neutral sentiment expressed by the user",
    )


@dataclass
class _ExtractionResult:
    """Result of _extract_and_store_entities."""

    entities: list[Entity] = field(default_factory=list)


class GeneratedTagline(BaseModel):
    """Schema for LLM response: a short disambiguating tagline."""

    tagline: str = Field(description="Short 3-8 word summary describing what the entity is")


@dataclass
class _EntityCandidate:
    """New entity held in memory until post-fact semantic pruning."""

    name: str
    tagline: str | None = None
    facts: list[str] = field(default_factory=list)


class ExtractionPipeline(Agent):
    """Unified background agent that extracts entities, facts, and engagements.

    Processes SearchLog and MessageLog entries to discover entities and facts.
    Does not send notifications — the NotificationAgent handles that separately.
    """

    def __init__(self, learn_agent: LearnAgent | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._learn_agent = learn_agent

    @property
    def name(self) -> str:
        """Task name for logging."""
        return "extraction"

    async def execute(self) -> bool:
        """Run the knowledge pipeline in strict priority order.

        Phase 1: User messages (highest priority — freshest signals)
        Phase 2: Search logs (drain backlog before enrichment)
        Phase 3: Enrichment (only when 1 & 2 had nothing to do)
        Phase 4: Embedding backfill

        Returns:
            True if any work was done.
        """
        work_done = False

        # Phase 1: Extract entities/facts from user messages (highest priority)
        work_done |= await self._process_messages()

        # Phase 2: Extract entities/facts from search results (newest first)
        work_done |= await self._process_search_logs()

        # Phase 3: Enrichment — only when phases 1 & 2 are fully drained.
        # Enrichment creates new SearchLog entries (trigger=penny_enrichment)
        # that feed back into phase 2 on the next cycle.
        if not work_done and self._learn_agent:
            work_done |= await self._learn_agent.execute()

        # Phase 4: Backfill embeddings for items that don't have them
        if self._embedding_model_client:
            work_done |= await self._backfill_embeddings()

        return work_done

    # --- Phase 1: Search log processing ---

    async def _process_search_logs(self) -> bool:
        """Process unextracted SearchLog entries for entity/fact extraction."""
        search_logs = self.db.get_unprocessed_search_logs(
            limit=int(self.config.runtime.ENTITY_EXTRACTION_BATCH_LIMIT)
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
            # Pre-mark user-sourced facts as notified (user already knows about them)
            silent = search_log.trigger == PennyConstants.SearchTrigger.USER_MESSAGE
            notified_at = datetime.now(UTC) if silent else None

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
                notified_at=notified_at,
            )
            if result.entities:
                work_done = True

                # Create USER_SEARCH engagements for user-triggered searches
                if allow_new:
                    for entity in result.entities:
                        assert entity.id is not None
                        self.db.add_engagement(
                            user=user,
                            engagement_type=PennyConstants.EngagementType.USER_SEARCH,
                            valence=PennyConstants.EngagementValence.POSITIVE,
                            strength=self.config.runtime.ENGAGEMENT_STRENGTH_USER_SEARCH,
                            entity_id=entity.id,
                        )

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
                sender, limit=int(self.config.runtime.PREFERENCE_BATCH_LIMIT)
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
                    notified_at=datetime.now(UTC),  # User-sourced — don't notify
                )
                if result.entities:
                    work_done = True

                    # Create MESSAGE_MENTION engagements for identified entities
                    entity_by_name: dict[str, Entity] = {
                        e.name: e for e in result.entities if e.id is not None
                    }
                    for entity in result.entities:
                        assert entity.id is not None
                        self.db.add_engagement(
                            user=sender,
                            engagement_type=PennyConstants.EngagementType.MESSAGE_MENTION,
                            valence=PennyConstants.EngagementValence.POSITIVE,
                            strength=self.config.runtime.ENGAGEMENT_STRENGTH_MESSAGE_MENTION,
                            entity_id=entity.id,
                            source_message_id=message.id,
                        )

                    # Extract sentiment and create EXPLICIT_STATEMENT engagements
                    sentiments = await self._extract_message_sentiments(
                        message.content, list(entity_by_name.keys())
                    )
                    for s in sentiments:
                        matched = entity_by_name.get(s.entity_name)
                        if not matched or matched.id is None:
                            continue
                        self.db.add_engagement(
                            user=sender,
                            engagement_type=PennyConstants.EngagementType.EXPLICIT_STATEMENT,
                            valence=s.sentiment,
                            strength=self.config.runtime.ENGAGEMENT_STRENGTH_EXPLICIT_STATEMENT,
                            entity_id=matched.id,
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
        notified_at: datetime | None = None,
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
            notified_at: Pre-mark stored facts as notified (e.g. user-sourced facts).

        Returns _ExtractionResult with all identified entities.
        """
        existing_entities = self.db.get_user_entities(user)

        # Collect all entities to process facts for
        entities_to_process: list[Entity] = []
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

            # Filter and route new entities: dedup matches → entities_to_process,
            # genuinely new → candidates (held in memory for post-fact pruning)
            candidates: list[_EntityCandidate] = []
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

                # Check for duplicate against existing entities
                duplicate: Entity | None = None
                if existing_entities and self._embedding_model_client:
                    try:
                        vecs = await self._embedding_model_client.embed([name])
                        duplicate = self._find_duplicate_entity(name, vecs[0], existing_entities)
                    except Exception:
                        logger.debug("Failed to embed candidate '%s'", name, exc_info=True)

                if duplicate:
                    entities_to_process.append(duplicate)
                else:
                    tagline = (
                        new_entity.tagline.strip().lower().rstrip(".")
                        if new_entity.tagline
                        else None
                    )
                    if tagline and len(tagline.split()) > 10:
                        tagline = None  # Reject overly long taglines
                    candidates.append(_EntityCandidate(name=name, tagline=tagline or None))

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

        # Pass 2a: extract facts for existing entities (already in DB)
        entities_with_new_facts: list[int] = []
        for entity in entities_to_process:
            assert entity.id is not None
            existing_fact_rows = self.db.get_entity_facts(entity.id)
            new_facts = await self._extract_facts(
                entity.name,
                existing_fact_rows,
                fact_prompt,
                context_label,
                context_value,
                content,
            )
            if not new_facts:
                continue

            new_fact_texts = await self._dedup_facts(new_facts, existing_fact_rows)
            if not new_fact_texts:
                continue

            # Batch-embed all new facts at once
            fact_embeddings: list[bytes | None] = [None] * len(new_fact_texts)
            if self._embedding_model_client:
                try:
                    vecs = await self._embedding_model_client.embed(new_fact_texts)
                    fact_embeddings = [serialize_embedding(v) for v in vecs]
                except Exception as e:
                    logger.warning("Failed to embed facts for '%s': %s", entity.name, e)

            for fact_text, emb in zip(new_fact_texts, fact_embeddings, strict=True):
                self.db.add_fact(
                    entity_id=entity.id,
                    content=fact_text,
                    source_search_log_id=source_search_log_id,
                    source_message_id=source_message_id,
                    embedding=emb,
                    notified_at=notified_at,
                )
                logger.info("  '%s' +fact: %s", entity.name, fact_text)

            entities_with_new_facts.append(entity.id)

        # Regenerate entity embeddings for existing entities that got new facts
        if self._embedding_model_client and entities_with_new_facts:
            await self._update_entity_embeddings(
                [e for e in entities_to_process if e.id in entities_with_new_facts]
            )

        # Pass 2b: extract facts for new candidates (in memory)
        if allow_new_entities:
            for candidate in candidates:
                candidate.facts = await self._extract_facts(
                    candidate.name,
                    [],
                    fact_prompt,
                    context_label,
                    context_value,
                    content,
                )

            # Drop candidates with no facts (nothing to validate or store)
            candidates = [c for c in candidates if c.facts]

        # Pass 3: semantic pruning + commit for new candidates
        if allow_new_entities and candidates:
            survivors = await self._prune_and_commit_candidates(
                candidates=candidates,
                trigger_text=relevance_reference or context_value,
                user=user,
                fact_prompt=fact_prompt,
                source_search_log_id=source_search_log_id,
                source_message_id=source_message_id,
                notified_at=notified_at,
                record_discovery_score=record_discovery_score,
            )
            entities_to_process.extend(survivors)

        return _ExtractionResult(entities=entities_to_process)

    async def _identify_entities(
        self,
        existing_entities: list[Entity],
        identification_prompt: str,
        context_label: str,
        context_value: str,
        content: str,
    ) -> IdentifiedEntities | None:
        """Pass 1: Identify which known entities appear in the text and any new entities."""
        if existing_entities:
            known_lines = []
            for e in existing_entities:
                if e.tagline:
                    known_lines.append(f"- {e.name} ({e.tagline})")
                else:
                    known_lines.append(f"- {e.name}")
            known_context = (
                "\n\nKnown entities (return any that appear in the text):\n"
                + "\n".join(known_lines)
            )
        else:
            known_context = (
                "\n\nKnown entities: none. Put all discovered entities in the 'new' list."
            )

        prompt = (
            f"{identification_prompt}\n\n"
            f"{context_label}: {context_value}\n\n"
            f"Content:\n{content}"
            f"{known_context}"
        )

        try:
            response = await self._background_model_client.generate(
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
        if not existing_entities:
            return None

        known_lines = []
        for e in existing_entities:
            if e.tagline:
                known_lines.append(f"- {e.name} ({e.tagline})")
            else:
                known_lines.append(f"- {e.name}")
        known_context = "\n\nKnown entities (return any that appear in the text):\n" + "\n".join(
            known_lines
        )

        prompt = (
            f"{identification_prompt}\n\n"
            f"{context_label}: {context_value}\n\n"
            f"Content:\n{content}"
            f"{known_context}"
        )

        try:
            response = await self._background_model_client.generate(
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
        if not self._embedding_model_client:
            return entities
        if len(entities) < self.config.runtime.EXTRACTION_PREFILTER_MIN_COUNT:
            return entities

        entities_with_embeddings = [e for e in entities if e.embedding is not None]
        entities_without_embeddings = [e for e in entities if e.embedding is None]

        if not entities_with_embeddings:
            return entities

        try:
            content_vecs = await self._embedding_model_client.embed(content)
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
                threshold=self.config.runtime.EXTRACTION_PREFILTER_SIMILARITY_THRESHOLD,
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

    async def _prune_and_commit_candidates(
        self,
        candidates: list[_EntityCandidate],
        trigger_text: str,
        user: str,
        fact_prompt: str,
        source_search_log_id: int | None,
        source_message_id: int | None,
        notified_at: datetime | None,
        record_discovery_score: bool,
    ) -> list[Entity]:
        """Semantic pruning using entity+facts embeddings, then commit survivors to DB.

        Each candidate has already had facts extracted in memory. This method:
        1. Builds entity+facts embed text for each candidate
        2. Compares against the trigger text via cosine similarity
        3. Creates DB records only for candidates that pass the threshold

        Returns the list of Entity objects that were committed.
        """
        if not candidates:
            return []

        # Without an embedding model, accept all candidates
        if not self._embedding_model_client:
            survivors_with_scores = [(c, 0.0) for c in candidates]
        else:
            # Batch-embed all entity+facts texts + trigger text in one call
            embed_texts = [build_entity_embed_text(c.name, c.facts, c.tagline) for c in candidates]
            embed_texts.append(trigger_text)

            try:
                vecs = await self._embedding_model_client.embed(embed_texts)
            except Exception:
                logger.warning(
                    "Post-fact embedding failed, accepting all candidates", exc_info=True
                )
                survivors_with_scores = [(c, 0.0) for c in candidates]
                vecs = None

            if vecs is not None:
                trigger_vec = vecs[-1]
                threshold = self.config.runtime.EXTRACTION_ENTITY_SEMANTIC_THRESHOLD
                survivors_with_scores = []

                for candidate, entity_vec in zip(candidates, vecs[:-1], strict=True):
                    score = cosine_similarity(entity_vec, trigger_vec)
                    if score >= threshold:
                        logger.info(
                            "Accepted entity '%s' (post-fact similarity %.2f >= %.2f)",
                            candidate.name,
                            score,
                            threshold,
                        )
                        survivors_with_scores.append((candidate, score))
                    else:
                        logger.info(
                            "Rejected entity '%s' (post-fact similarity %.2f < %.2f)",
                            candidate.name,
                            score,
                            threshold,
                        )

        # Commit survivors: create entity, store facts, record engagement
        committed: list[Entity] = []
        for candidate, score in survivors_with_scores:
            entity = self.db.get_or_create_entity(user, candidate.name)
            if not entity or entity.id is None:
                continue
            logger.info("New entity discovered: '%s'", candidate.name)

            # Store tagline if available
            if candidate.tagline and entity.tagline is None:
                self.db.update_entity_tagline(entity.id, candidate.tagline)
                entity.tagline = candidate.tagline
                logger.info("Tagline for '%s': '%s'", candidate.name, candidate.tagline)

            # Batch-embed and store facts
            fact_embeddings: list[bytes | None] = [None] * len(candidate.facts)
            if self._embedding_model_client:
                try:
                    fact_vecs = await self._embedding_model_client.embed(candidate.facts)
                    fact_embeddings = [serialize_embedding(v) for v in fact_vecs]
                except Exception as e:
                    logger.warning("Failed to embed facts for '%s': %s", candidate.name, e)

            for fact_text, emb in zip(candidate.facts, fact_embeddings, strict=True):
                self.db.add_fact(
                    entity_id=entity.id,
                    content=fact_text,
                    source_search_log_id=source_search_log_id,
                    source_message_id=source_message_id,
                    embedding=emb,
                    notified_at=notified_at,
                )
                logger.info("  '%s' +fact: %s", candidate.name, fact_text)

            committed.append(entity)

            if record_discovery_score and score > 0.0:
                self.db.add_engagement(
                    user=user,
                    engagement_type=PennyConstants.EngagementType.SEARCH_DISCOVERY,
                    valence=PennyConstants.EngagementValence.POSITIVE,
                    strength=score,
                    entity_id=entity.id,
                )

        # Regenerate entity embeddings for all committed entities
        if self._embedding_model_client and committed:
            await self._update_entity_embeddings(committed)

        return committed

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
            if tcr < self.config.runtime.EXTRACTION_ENTITY_DEDUP_TCR_THRESHOLD:
                continue
            entity_vec = deserialize_embedding(entity.embedding)
            sim = cosine_similarity(candidate_embedding, entity_vec)
            if sim >= self.config.runtime.EXTRACTION_ENTITY_DEDUP_EMBEDDING_THRESHOLD:
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
        entity_name: str,
        existing_facts: list[Fact],
        fact_prompt: str,
        context_label: str,
        context_value: str,
        content: str,
    ) -> list[str]:
        """Pass 2: Extract new facts about a specific entity from content."""
        existing_context = ""
        if existing_facts:
            facts_text = "\n".join(f"- {f.content}" for f in existing_facts)
            existing_context = (
                f"\n\nAlready known facts (return only NEW facts not listed here):\n{facts_text}"
            )

        prompt = (
            f"{fact_prompt}\n\n"
            f"Entity: {entity_name}\n\n"
            f"{context_label}: {context_value}\n\n"
            f"Content:\n{content}"
            f"{existing_context}"
        )

        try:
            response = await self._background_model_client.generate(
                prompt=prompt,
                tools=None,
                format=ExtractedFacts.model_json_schema(),
            )
            result = ExtractedFacts.model_validate_json(response.content)
            return result.facts
        except Exception as e:
            logger.error("Failed to extract facts for '%s': %s", entity_name, e)
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
        if not self._embedding_model_client:
            return candidates

        facts_with_embeddings = [f for f in existing_facts if f.embedding is not None]
        if not facts_with_embeddings:
            return candidates

        try:
            vecs = await self._embedding_model_client.embed(candidates)
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
                    threshold=self.config.runtime.EXTRACTION_FACT_DEDUP_SIMILARITY_THRESHOLD,
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

    def _should_process_message(self, message: MessageLog) -> bool:
        """Lightweight pre-filter to skip low-signal messages before LLM calls."""
        content = message.content.strip()
        if len(content) < self.config.runtime.EXTRACTION_MIN_MESSAGE_LENGTH:
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
        if not message.parent_id or not self._embedding_model_client:
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
            vecs = await self._embedding_model_client.embed(parent_msg.content)
            query_vec = vecs[0]
            matches = find_similar(
                query_vec,
                candidates,
                top_k=int(self.config.runtime.ENTITY_CONTEXT_TOP_K),
                threshold=self.config.runtime.ENTITY_CONTEXT_THRESHOLD,
            )

            for entity_id, _score in matches:
                self.db.add_engagement(
                    user=sender,
                    engagement_type=PennyConstants.EngagementType.FOLLOW_UP_QUESTION,
                    valence=PennyConstants.EngagementValence.POSITIVE,
                    strength=self.config.runtime.ENGAGEMENT_STRENGTH_FOLLOW_UP_QUESTION,
                    entity_id=entity_id,
                    source_message_id=message.id,
                )

            return len(matches) > 0
        except Exception:
            logger.debug("Follow-up engagement extraction failed", exc_info=True)
            return False

    # --- Sentiment extraction ---

    async def _extract_message_sentiments(
        self, message_content: str, entity_names: list[str]
    ) -> list[EntitySentiment]:
        """Extract user sentiment toward entities mentioned in their message.

        Returns only non-neutral sentiments. Gracefully returns empty on failure.
        """
        entities_context = "\n".join(f"- {name}" for name in entity_names)
        prompt = (
            f"{Prompt.MESSAGE_SENTIMENT_EXTRACTION_PROMPT}\n\n"
            f"Entities found in this message:\n{entities_context}\n\n"
            f"User message:\n{message_content}"
        )

        try:
            response = await self._background_model_client.generate(
                prompt=prompt,
                tools=None,
                format=MessageSentiments.model_json_schema(),
            )
            result = MessageSentiments.model_validate_json(response.content)
            return [
                s
                for s in result.sentiments
                if s.sentiment
                in (
                    PennyConstants.EngagementValence.POSITIVE,
                    PennyConstants.EngagementValence.NEGATIVE,
                )
            ]
        except Exception:
            logger.debug("Sentiment extraction failed", exc_info=True)
            return []

    # --- Tagline generation ---

    async def _generate_tagline(self, entity_name: str, facts: list[str]) -> str | None:
        """Generate a short disambiguating tagline for an entity using LLM.

        Used for backfilling existing entities that don't have taglines.
        New entities get taglines from the identification step instead.
        """
        facts_text = "\n".join(f"- {f}" for f in facts[:10])
        prompt = (
            f"{Prompt.TAGLINE_GENERATION_PROMPT}\n\n"
            f"Entity: {entity_name}\n\n"
            f"Known facts:\n{facts_text}"
        )

        try:
            response = await self._background_model_client.generate(
                prompt=prompt,
                tools=None,
                format=GeneratedTagline.model_json_schema(),
            )
            result = GeneratedTagline.model_validate_json(response.content)
            tagline = result.tagline.strip().lower().rstrip(".")
            if tagline and len(tagline.split()) <= 10:
                return tagline
            logger.warning("Rejected tagline for '%s': too long or empty", entity_name)
            return None
        except Exception as e:
            logger.error("Failed to generate tagline for '%s': %s", entity_name, e)
            return None

    # --- Entity embedding updates ---

    async def _update_entity_embeddings(self, entities: list[Entity]) -> None:
        """Regenerate embeddings for entities by composing name + facts."""
        assert self._embedding_model_client is not None
        if not entities:
            return

        embed_texts: list[str] = []
        for entity in entities:
            assert entity.id is not None
            facts = self.db.get_entity_facts(entity.id)
            embed_texts.append(
                build_entity_embed_text(entity.name, [f.content for f in facts], entity.tagline)
            )

        try:
            vecs = await self._embedding_model_client.embed(embed_texts)
            for entity, vec in zip(entities, vecs, strict=True):
                assert entity.id is not None
                self.db.update_entity_embedding(entity.id, serialize_embedding(vec))
                logger.debug("Updated entity embedding for '%s'", entity.name)
        except Exception as e:
            logger.warning("Failed to update entity embeddings: %s", e)

    # --- Embedding backfill ---

    async def _backfill_embeddings(self) -> bool:
        """Backfill embeddings for entities and facts that don't have them.

        Returns:
            True if any embeddings were generated.
        """
        assert self._embedding_model_client is not None
        work_done = False
        batch_limit = int(self.config.runtime.EMBEDDING_BACKFILL_BATCH_LIMIT)

        # Backfill fact embeddings
        facts = self.db.get_facts_without_embeddings(limit=batch_limit)
        if facts:
            fact_texts = [f.content for f in facts]
            try:
                vecs = await self._embedding_model_client.embed(fact_texts)
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
                    build_entity_embed_text(
                        entity.name, [f.content for f in facts_for_entity], entity.tagline
                    )
                )
            try:
                vecs = await self._embedding_model_client.embed(embed_texts)
                for entity, vec in zip(entities, vecs, strict=True):
                    assert entity.id is not None
                    self.db.update_entity_embedding(entity.id, serialize_embedding(vec))
                logger.info("Backfilled embeddings for %d entities", len(entities))
                work_done = True
            except Exception as e:
                logger.warning("Failed to backfill entity embeddings: %s", e)

        # Backfill taglines for entities that have facts but no tagline (1 per cycle)
        entities_needing_taglines = self.db.get_entities_without_taglines(limit=1)
        for entity in entities_needing_taglines:
            assert entity.id is not None
            entity_facts = self.db.get_entity_facts(entity.id)
            if entity_facts:
                tagline = await self._generate_tagline(
                    entity.name, [f.content for f in entity_facts]
                )
                if tagline:
                    self.db.update_entity_tagline(entity.id, tagline)
                    logger.info("Backfilled tagline for '%s': '%s'", entity.name, tagline)
                    work_done = True

        return work_done

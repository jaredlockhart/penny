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
    deserialize_embedding,
    find_similar,
    serialize_embedding,
    tokenize_entity_name,
)
from penny.ollama.similarity import (
    DedupStrategy,
    check_relevance,
    dedup_facts_by_embedding,
    is_embedding_duplicate,
    normalize_fact,
)
from penny.prompts import Prompt

if TYPE_CHECKING:
    from penny.database.models import MessageLog

logger = logging.getLogger(__name__)


def _strip_name_from_text(text: str, name: str) -> str:
    """Remove all occurrences of an entity name from text (case-insensitive).

    Uses word-boundary matching to avoid partial-word replacements, then
    collapses any resulting extra whitespace/punctuation artifacts.
    """
    # Escape for regex, match case-insensitively at word boundaries
    pattern = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
    result = pattern.sub("", text)
    # Collapse leftover whitespace and strip leading punctuation/space
    result = re.sub(r"\s+", " ", result).strip()
    result = re.sub(r"^[;,.\s]+", "", result).strip()
    return result


def _build_relevance_text(tagline: str | None, facts: list[str], name: str | None = None) -> str:
    """Build embedding text for relevance comparison (tagline + facts, no name).

    Entity names are omitted because ambiguous names (e.g. "genesis", "focus",
    "renaissance") pull the embedding away from the domain context.  If *name*
    is provided it is also stripped from the fact text, since the LLM often
    embeds the name inside each fact sentence.
    """
    cleaned_facts = list(facts or [])
    if name:
        cleaned_facts = [_strip_name_from_text(f, name) for f in cleaned_facts]
        cleaned_facts = [f for f in cleaned_facts if f]  # drop empty after stripping
    parts = [p for p in [tagline, *cleaned_facts] if p]
    return "; ".join(parts) if parts else ""


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

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]

    @property
    def name(self) -> str:
        """Task name for logging."""
        return "extraction"

    async def execute(self) -> bool:
        """Run the knowledge pipeline in strict priority order.

        Phase 1: User messages (highest priority — freshest signals)
        Phase 2: Search logs (drain backlog)
        Phase 3: Embedding backfill

        Returns:
            True if any work was done.
        """
        work_done = False

        # Phase 1: Extract entities/facts from user messages (highest priority)
        work_done |= await self._process_messages()

        # Phase 2: Extract entities/facts from search results (newest first)
        work_done |= await self._process_search_logs()

        # Phase 3: Backfill embeddings for items that don't have them
        if self._embedding_model_client:
            work_done |= await self._backfill_embeddings()

        return work_done

    # --- Phase 1: Search log processing ---

    async def _process_search_logs(self) -> bool:
        """Process unextracted SearchLog entries for entity/fact extraction."""
        search_logs = self.db.searches.get_unprocessed(
            limit=int(self.config.runtime.ENTITY_EXTRACTION_BATCH_LIMIT)
        )
        if not search_logs:
            return False

        work_done = False
        logger.info("Processing %d unprocessed search logs", len(search_logs))

        for search_log in search_logs:
            assert search_log.id is not None

            user = self.db.users.find_sender_for_timestamp(search_log.timestamp)
            if not user:
                self.db.searches.mark_extracted(search_log.id)
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
                learn_prompt = self.db.learn_prompts.get(search_log.learn_prompt_id)
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
                        self.db.engagements.add(
                            user=user,
                            engagement_type=PennyConstants.EngagementType.USER_SEARCH,
                            valence=PennyConstants.EngagementValence.POSITIVE,
                            strength=self.config.runtime.ENGAGEMENT_STRENGTH_USER_SEARCH,
                            entity_id=entity.id,
                        )

            self.db.searches.mark_extracted(search_log.id)

        return work_done

    # --- Phase 2: Message processing ---

    async def _process_messages(self) -> bool:
        """Process unprocessed messages for entity/fact extraction."""
        senders = self.db.users.get_all_senders()
        if not senders:
            return False

        work_done = False

        for sender in senders:
            messages = self.db.messages.get_unprocessed(
                sender, limit=int(self.config.runtime.MESSAGE_EXTRACTION_BATCH_LIMIT)
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
                        self.db.engagements.add(
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
                        self.db.engagements.add(
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
                self.db.messages.mark_processed(message_ids)

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
        existing_entities = self.db.entities.get_for_user(user)
        existing_by_name = {e.name: e for e in existing_entities}
        filtered_entities = await self._prefilter_entities_by_similarity(existing_entities, content)

        # Pass 1: identify and route entities
        result = await self._identify_and_route(
            filtered_entities,
            existing_entities,
            existing_by_name,
            identification_prompt,
            context_label,
            context_value,
            content,
            allow_new_entities,
        )
        if result is None:
            return _ExtractionResult()
        entities_to_process, candidates = result

        # Pass 2a: extract facts for existing entities (already in DB)
        entities_with_new_facts = await self._extract_facts_for_existing(
            entities_to_process,
            fact_prompt,
            context_label,
            context_value,
            content,
            source_search_log_id,
            source_message_id,
            notified_at,
        )

        # Regenerate entity embeddings for existing entities that got new facts
        if self._embedding_model_client and entities_with_new_facts:
            await self._update_entity_embeddings(
                [e for e in entities_to_process if e.id in entities_with_new_facts]
            )

        # Pass 2b: extract facts for new candidates (in memory)
        if allow_new_entities:
            candidates = await self._extract_candidate_facts(
                candidates,
                fact_prompt,
                context_label,
                context_value,
                content,
            )

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

    async def _identify_and_route(
        self,
        filtered_entities: list[Entity],
        existing_entities: list[Entity],
        existing_by_name: dict[str, Entity],
        identification_prompt: str,
        context_label: str,
        context_value: str,
        content: str,
        allow_new_entities: bool,
    ) -> tuple[list[Entity], list[_EntityCandidate]] | None:
        """Pass 1: identify entities and route to existing or candidate lists.

        Returns (entities_to_process, candidates) or None if nothing identified.
        """
        if allow_new_entities:
            identified = await self._identify_entities(
                filtered_entities, identification_prompt, context_label, context_value, content
            )
            if not identified:
                return None

            subset_names = self._filter_subset_candidates(identified)
            entities_to_process, candidates = await self._route_new_entities(
                identified,
                existing_entities,
                existing_by_name,
                subset_names,
            )
            self._route_known_entities(identified.known, existing_by_name, entities_to_process)
            return entities_to_process, candidates

        # Known-only mode
        known_result = await self._identify_known_entities(
            filtered_entities, identification_prompt, context_label, context_value, content
        )
        if not known_result:
            return None

        entities_to_process: list[Entity] = []
        self._route_known_entities(known_result.known, existing_by_name, entities_to_process)
        return entities_to_process, []

    def _filter_subset_candidates(self, identified: IdentifiedEntities) -> set[str]:
        """Find candidate names that are token-subsets of another candidate.

        Filters out fragment entities (e.g. "totem" when "totem loon" is also present).
        """
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
        return subset_names

    async def _route_new_entities(
        self,
        identified: IdentifiedEntities,
        existing_entities: list[Entity],
        existing_by_name: dict[str, Entity],
        subset_names: set[str],
    ) -> tuple[list[Entity], list[_EntityCandidate]]:
        """Route new entities: dedup matches go to entities_to_process, rest to candidates."""
        entities_to_process: list[Entity] = []
        candidates: list[_EntityCandidate] = []
        for new_entity in identified.new:
            name = new_entity.name.lower().strip()
            if not name or not _is_valid_entity_name(name):
                if name:
                    logger.info("Rejected entity '%s' (structural filter)", name)
                continue
            if name in subset_names:
                logger.info("Rejected entity '%s' (token-subset of another candidate)", name)
                continue

            duplicate = await self._check_duplicate(name, existing_entities)
            if duplicate:
                entities_to_process.append(duplicate)
            else:
                candidate = self._build_candidate(name, new_entity.tagline)
                candidates.append(candidate)
        return entities_to_process, candidates

    async def _check_duplicate(self, name: str, existing_entities: list[Entity]) -> Entity | None:
        """Check if a candidate name is a duplicate of an existing entity via embedding."""
        if not existing_entities or not self._embedding_model_client:
            return None
        try:
            vecs = await self._embedding_model_client.embed([name])
            return self._find_duplicate_entity(name, vecs[0], existing_entities)
        except Exception:
            logger.debug("Failed to embed candidate '%s'", name, exc_info=True)
            return None

    def _build_candidate(self, name: str, raw_tagline: str | None) -> _EntityCandidate:
        """Build an _EntityCandidate with validated tagline."""
        tagline = raw_tagline.strip().lower().rstrip(".") if raw_tagline else None
        if tagline and len(tagline.split()) > 10:
            tagline = None  # Reject overly long taglines
        return _EntityCandidate(name=name, tagline=tagline or None)

    def _route_known_entities(
        self,
        known_names: list[str],
        existing_by_name: dict[str, Entity],
        entities_to_process: list[Entity],
    ) -> None:
        """Look up known entity names and append matches to entities_to_process."""
        for known_name in known_names:
            normalized = known_name.lower().strip()
            if normalized in existing_by_name:
                entities_to_process.append(existing_by_name[normalized])
                logger.info("Known entity referenced: '%s'", normalized)

    async def _extract_facts_for_existing(
        self,
        entities_to_process: list[Entity],
        fact_prompt: str,
        context_label: str,
        context_value: str,
        content: str,
        source_search_log_id: int | None,
        source_message_id: int | None,
        notified_at: datetime | None,
    ) -> list[int]:
        """Pass 2a: extract and store facts for existing entities. Returns IDs with new facts."""
        entities_with_new_facts: list[int] = []
        for entity in entities_to_process:
            assert entity.id is not None
            existing_fact_rows = self.db.facts.get_for_entity(entity.id)
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

            await self._embed_and_store_facts(
                entity.id,
                entity.name,
                new_fact_texts,
                source_search_log_id,
                source_message_id,
                notified_at,
            )
            entities_with_new_facts.append(entity.id)
        return entities_with_new_facts

    async def _embed_and_store_facts(
        self,
        entity_id: int,
        entity_name: str,
        fact_texts: list[str],
        source_search_log_id: int | None,
        source_message_id: int | None,
        notified_at: datetime | None,
    ) -> None:
        """Batch-embed facts and store them in the database for one entity."""
        fact_embeddings: list[bytes | None] = [None] * len(fact_texts)
        if self._embedding_model_client:
            try:
                vecs = await self._embedding_model_client.embed(fact_texts)
                fact_embeddings = [serialize_embedding(v) for v in vecs]
            except Exception as e:
                logger.warning("Failed to embed facts for '%s': %s", entity_name, e)

        for fact_text, emb in zip(fact_texts, fact_embeddings, strict=True):
            self.db.facts.add(
                entity_id=entity_id,
                content=fact_text,
                source_search_log_id=source_search_log_id,
                source_message_id=source_message_id,
                embedding=emb,
                notified_at=notified_at,
            )
            logger.info("  '%s' +fact: %s", entity_name, fact_text)

    async def _extract_candidate_facts(
        self,
        candidates: list[_EntityCandidate],
        fact_prompt: str,
        context_label: str,
        context_value: str,
        content: str,
    ) -> list[_EntityCandidate]:
        """Pass 2b: extract facts for new candidates (in memory). Returns candidates with facts."""
        for candidate in candidates:
            candidate.facts = await self._extract_facts(
                candidate.name,
                [],
                fact_prompt,
                context_label,
                context_value,
                content,
            )
        return [c for c in candidates if c.facts]

    def _build_known_entities_context(self, existing_entities: list[Entity]) -> str:
        """Build the known-entities context block shared by identification prompts."""
        if not existing_entities:
            return "\n\nKnown entities: none. Put all discovered entities in the 'new' list."

        known_lines = []
        for e in existing_entities:
            if e.tagline:
                known_lines.append(f"- {e.name} ({e.tagline})")
            else:
                known_lines.append(f"- {e.name}")
        return "\n\nKnown entities (return any that appear in the text):\n" + "\n".join(known_lines)

    async def _identify_entities(
        self,
        existing_entities: list[Entity],
        identification_prompt: str,
        context_label: str,
        context_value: str,
        content: str,
    ) -> IdentifiedEntities | None:
        """Pass 1: Identify which known entities appear in the text and any new entities."""
        known_context = self._build_known_entities_context(existing_entities)

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
            if not response.content or not response.content.strip():
                logger.warning("Empty LLM response from entity identification — skipping")
                return None
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

        known_context = self._build_known_entities_context(existing_entities)

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
            if not response.content or not response.content.strip():
                logger.warning("Empty LLM response from known entity identification — skipping")
                return None
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

        survivors_with_scores = await self._score_candidates_by_relevance(candidates, trigger_text)

        # Commit survivors sequentially: for each candidate, build its full
        # embedding (name + tagline + facts), check dedup against all existing
        # entities in the DB (including earlier candidates committed this batch),
        # then either merge into the dupe or create a new entity.
        committed: list[Entity] = []
        for candidate, score in survivors_with_scores:
            entity = await self._commit_candidate(
                candidate,
                score,
                user,
                source_search_log_id,
                source_message_id,
                notified_at,
                record_discovery_score,
            )
            if entity:
                committed.append(entity)
        return committed

    async def _score_candidates_by_relevance(
        self,
        candidates: list[_EntityCandidate],
        trigger_text: str,
    ) -> list[tuple[_EntityCandidate, float]]:
        """Embed candidates and score against trigger text. Returns (candidate, score) pairs.

        Computes two scores per candidate — name-stripped and name-included —
        and takes the max.  Stripping helps ambiguous names ("genesis", "yes")
        that pull embeddings away from their domain; including helps when the
        name IS the relevance signal ("star trek: voyager" vs trigger "star trek").
        """
        if not self._embedding_model_client:
            return [(c, 0.0) for c in candidates]

        n = len(candidates)
        stripped_texts = [_build_relevance_text(c.tagline, c.facts, c.name) for c in candidates]
        included_texts = [_build_relevance_text(c.tagline, c.facts) for c in candidates]
        all_texts = stripped_texts + included_texts + [trigger_text]

        try:
            vecs = await self._embedding_model_client.embed(all_texts)
        except Exception:
            logger.warning("Post-fact embedding failed, accepting all candidates", exc_info=True)
            return [(c, 0.0) for c in candidates]

        stripped_vecs = vecs[:n]
        included_vecs = vecs[n : 2 * n]
        trigger_vec = vecs[-1]
        threshold = self.config.runtime.EXTRACTION_ENTITY_SEMANTIC_THRESHOLD
        survivors: list[tuple[_EntityCandidate, float]] = []

        for i, candidate in enumerate(candidates):
            s_result = check_relevance(stripped_vecs[i], trigger_vec, threshold)
            i_result = check_relevance(included_vecs[i], trigger_vec, threshold)
            s_score = s_result if s_result is not None else 0.0
            i_score = i_result if i_result is not None else 0.0
            best = "included" if i_score > s_score else "stripped"
            score = round(max(s_score, i_score), 2)
            if score >= threshold:
                logger.info(
                    "Accepted entity '%s' (post-fact similarity %.2f >= %.2f, best=%s)",
                    candidate.name,
                    score,
                    threshold,
                    best,
                )
                survivors.append((candidate, score))
            else:
                logger.info(
                    "Rejected entity '%s' (post-fact similarity %.2f < %.2f, best=%s)",
                    candidate.name,
                    score,
                    threshold,
                    best,
                )
        return survivors

    async def _commit_candidate(
        self,
        candidate: _EntityCandidate,
        score: float,
        user: str,
        source_search_log_id: int | None,
        source_message_id: int | None,
        notified_at: datetime | None,
        record_discovery_score: bool,
    ) -> Entity | None:
        """Commit a single candidate: dedup check, then merge or create.

        Returns the committed entity, or None if creation failed.
        """
        candidate_embed_text = build_entity_embed_text(
            candidate.name, candidate.facts, candidate.tagline
        )
        candidate_embedding: list[float] | None = None
        if self._embedding_model_client:
            try:
                vecs = await self._embedding_model_client.embed([candidate_embed_text])
                candidate_embedding = vecs[0]
            except Exception:
                logger.debug(
                    "Failed to embed candidate '%s' for dedup", candidate.name, exc_info=True
                )

        # Check dedup against all existing entities (refreshed each iteration
        # to include entities committed earlier in this same loop)
        existing_entities = self.db.entities.get_for_user(user)
        duplicate = self._find_duplicate_entity(
            candidate.name, candidate_embedding, existing_entities
        )

        if duplicate:
            entity = await self._merge_into_existing(
                candidate,
                duplicate,
                source_search_log_id,
                source_message_id,
                notified_at,
            )
        else:
            entity = await self._create_new_entity(
                candidate,
                candidate_embedding,
                user,
                source_search_log_id,
                source_message_id,
                notified_at,
            )

        if entity and record_discovery_score and score > 0.0:
            assert entity.id is not None
            self.db.engagements.add(
                user=user,
                engagement_type=PennyConstants.EngagementType.SEARCH_DISCOVERY,
                valence=PennyConstants.EngagementValence.POSITIVE,
                strength=score,
                entity_id=entity.id,
            )
        return entity

    async def _merge_into_existing(
        self,
        candidate: _EntityCandidate,
        duplicate: Entity,
        source_search_log_id: int | None,
        source_message_id: int | None,
        notified_at: datetime | None,
    ) -> Entity:
        """Merge a candidate's facts into an existing duplicate entity."""
        assert duplicate.id is not None
        logger.info(
            "Post-fact dedup: '%s' merges into existing '%s'",
            candidate.name,
            duplicate.name,
        )
        existing_facts = self.db.facts.get_for_entity(duplicate.id)
        new_fact_texts = await self._dedup_facts(candidate.facts, existing_facts)

        if new_fact_texts:
            await self._embed_and_store_facts(
                duplicate.id,
                duplicate.name,
                new_fact_texts,
                source_search_log_id,
                source_message_id,
                notified_at,
            )
            if self._embedding_model_client:
                await self._update_entity_embeddings([duplicate])

        return duplicate

    async def _create_new_entity(
        self,
        candidate: _EntityCandidate,
        candidate_embedding: list[float] | None,
        user: str,
        source_search_log_id: int | None,
        source_message_id: int | None,
        notified_at: datetime | None,
    ) -> Entity | None:
        """Create a new entity from a candidate and store its facts."""
        entity = self.db.entities.get_or_create(user, candidate.name)
        if not entity or entity.id is None:
            return None
        logger.info("New entity discovered: '%s'", candidate.name)

        if candidate.tagline and entity.tagline is None:
            self.db.entities.update_tagline(entity.id, candidate.tagline)
            entity.tagline = candidate.tagline
            logger.info("Tagline for '%s': '%s'", candidate.name, candidate.tagline)

        await self._embed_and_store_facts(
            entity.id,
            candidate.name,
            candidate.facts,
            source_search_log_id,
            source_message_id,
            notified_at,
        )

        # Compute and store entity embedding immediately so subsequent
        # candidates in this batch can dedup against it
        if self._embedding_model_client and candidate_embedding is not None:
            self.db.entities.update_embedding(entity.id, serialize_embedding(candidate_embedding))
        return entity

    def _find_duplicate_entity(
        self,
        candidate_name: str,
        candidate_embedding: list[float] | None,
        existing_entities: list[Entity],
    ) -> Entity | None:
        """Check if a candidate entity name is a duplicate of an existing entity.

        Delegates to is_embedding_duplicate with TCR_AND_EMBEDDING strategy.
        """
        items = [(e.name, e.embedding) for e in existing_entities]
        match_idx = is_embedding_duplicate(
            candidate_name,
            candidate_embedding,
            items,
            DedupStrategy.TCR_AND_EMBEDDING,
            embedding_threshold=self.config.runtime.EXTRACTION_ENTITY_DEDUP_EMBEDDING_THRESHOLD,
            tcr_threshold=self.config.runtime.EXTRACTION_ENTITY_DEDUP_TCR_THRESHOLD,
        )
        if match_idx is not None:
            matched = existing_entities[match_idx]
            logger.info("Dedup: '%s' matches existing '%s'", candidate_name, matched.name)
            return matched
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
            if not response.content or not response.content.strip():
                logger.warning(
                    "Empty LLM response from fact extraction for '%s' — skipping", entity_name
                )
                return []
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
        candidates = self._string_dedup_facts(new_facts, existing_facts)
        if not candidates:
            return []

        # Slow pass: embedding similarity dedup
        existing_with_emb = [
            (i, f.embedding) for i, f in enumerate(existing_facts) if f.embedding is not None
        ]
        threshold = self.config.runtime.EXTRACTION_FACT_DEDUP_SIMILARITY_THRESHOLD
        survivors, _ = await dedup_facts_by_embedding(
            self._embedding_model_client, candidates, existing_with_emb, threshold
        )
        return survivors

    def _string_dedup_facts(self, new_facts: list[str], existing_facts: list[Fact]) -> list[str]:
        """Fast pass: deduplicate by normalized string match."""
        existing_normalized = {normalize_fact(f.content) for f in existing_facts}
        candidates: list[str] = []
        for fact_text in new_facts:
            fact_text = fact_text.strip()
            if not fact_text:
                continue
            normalized = normalize_fact(fact_text)
            if normalized in existing_normalized:
                continue
            candidates.append(fact_text)
            existing_normalized.add(normalized)
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

        parent_msg = self.db.messages.get_by_id(message.parent_id)
        if not parent_msg:
            return False
        if parent_msg.direction != PennyConstants.MessageDirection.OUTGOING:
            return False

        entities = self.db.entities.get_with_embeddings(sender)
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
                self.db.engagements.add(
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
            if not response.content or not response.content.strip():
                return []
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

    # --- Entity embedding updates ---

    async def _update_entity_embeddings(self, entities: list[Entity]) -> None:
        """Regenerate embeddings for entities by composing name + facts."""
        assert self._embedding_model_client is not None
        if not entities:
            return

        embed_texts: list[str] = []
        for entity in entities:
            assert entity.id is not None
            facts = self.db.facts.get_for_entity(entity.id)
            embed_texts.append(
                build_entity_embed_text(entity.name, [f.content for f in facts], entity.tagline)
            )

        try:
            vecs = await self._embedding_model_client.embed(embed_texts)
            for entity, vec in zip(entities, vecs, strict=True):
                assert entity.id is not None
                self.db.entities.update_embedding(entity.id, serialize_embedding(vec))
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
        batch_limit = int(self.config.runtime.EMBEDDING_BACKFILL_BATCH_LIMIT)

        facts_done = await self._backfill_fact_embeddings(batch_limit)
        entities_done = await self._backfill_entity_embeddings(batch_limit)
        return facts_done or entities_done

    async def _backfill_fact_embeddings(self, batch_limit: int) -> bool:
        """Backfill embeddings for facts that don't have them."""
        assert self._embedding_model_client is not None
        facts = self.db.facts.get_without_embeddings(limit=batch_limit)
        if not facts:
            return False

        fact_texts = [f.content for f in facts]
        try:
            vecs = await self._embedding_model_client.embed(fact_texts)
            for fact, vec in zip(facts, vecs, strict=True):
                assert fact.id is not None
                self.db.facts.update_embedding(fact.id, serialize_embedding(vec))
            logger.info("Backfilled embeddings for %d facts", len(facts))
            return True
        except Exception as e:
            logger.warning("Failed to backfill fact embeddings: %s", e)
            return False

    async def _backfill_entity_embeddings(self, batch_limit: int) -> bool:
        """Backfill embeddings for entities that don't have them."""
        assert self._embedding_model_client is not None
        entities = self.db.entities.get_without_embeddings(limit=batch_limit)
        if not entities:
            return False

        embed_texts: list[str] = []
        for entity in entities:
            assert entity.id is not None
            facts_for_entity = self.db.facts.get_for_entity(entity.id)
            embed_texts.append(
                build_entity_embed_text(
                    entity.name, [f.content for f in facts_for_entity], entity.tagline
                )
            )
        try:
            vecs = await self._embedding_model_client.embed(embed_texts)
            for entity, vec in zip(entities, vecs, strict=True):
                assert entity.id is not None
                self.db.entities.update_embedding(entity.id, serialize_embedding(vec))
            logger.info("Backfilled embeddings for %d entities", len(entities))
            return True
        except Exception as e:
            logger.warning("Failed to backfill entity embeddings: %s", e)
            return False

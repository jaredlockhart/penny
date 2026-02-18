"""Unified extraction pipeline for entities, facts, preferences, and signals."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from penny.agents.base import Agent
from penny.constants import PennyConstants
from penny.database.models import Entity, Fact, Preference
from penny.ollama.embeddings import (
    build_entity_embed_text,
    deserialize_embedding,
    find_similar,
    serialize_embedding,
)
from penny.prompts import Prompt

if TYPE_CHECKING:
    from penny.channels import MessageChannel
    from penny.database.models import MessageLog

logger = logging.getLogger(__name__)

# Pattern to collapse whitespace and strip bullet prefixes for fact comparison
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_fact(fact: str) -> str:
    """Normalize a fact string for dedup comparison.

    Strips leading '- ', lowercases, and collapses whitespace so that
    near-duplicate facts with minor formatting differences are caught.
    """
    text = fact.strip().lstrip("-").strip()
    return _WHITESPACE_RE.sub(" ", text).lower()


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


class ExtractedFacts(BaseModel):
    """Schema for pass 2: new facts about a single entity."""

    facts: list[str] = Field(
        default_factory=list,
        description="NEW specific, verifiable facts about the entity from the text",
    )


class ExtractedTopics(BaseModel):
    """Schema for LLM response: list of newly discovered preference topics."""

    topics: list[str] = Field(
        default_factory=list,
        description="List of new topics found (short phrases, 1-4 words each)",
    )


class ExtractionPipeline(Agent):
    """Unified background agent that extracts entities, facts, preferences, and signals.

    Replaces the separate EntityExtractor and PreferenceAgent with a single pipeline
    that processes both SearchLog and MessageLog entries.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._channel: MessageChannel | None = None

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

            logger.info("Extracting entities from search: %s", search_log.query)
            result = await self._extract_and_store_entities(
                user=user,
                identification_prompt=Prompt.ENTITY_IDENTIFICATION_PROMPT,
                fact_prompt=Prompt.ENTITY_FACT_EXTRACTION_PROMPT,
                context_label="Search query",
                context_value=search_log.query,
                content=search_log.response,
                source_search_log_id=search_log.id,
            )
            if result:
                work_done = True

            self.db.mark_search_extracted(search_log.id)

        return work_done

    # --- Phase 2: Message processing ---

    async def _process_messages(self) -> bool:
        """Process unprocessed messages for entity/fact/preference extraction."""
        senders = self.db.get_all_senders()
        if not senders:
            return False

        work_done = False

        for sender in senders:
            reactions = self.db.get_user_reactions(
                sender, limit=PennyConstants.PREFERENCE_BATCH_LIMIT
            )
            messages = self.db.get_unprocessed_messages(
                sender, limit=PennyConstants.PREFERENCE_BATCH_LIMIT
            )

            if not reactions and not messages:
                continue

            # --- Entity/fact extraction from messages ---
            for message in messages:
                if not self._should_process_message(message):
                    continue

                assert message.id is not None
                entities = await self._extract_and_store_entities(
                    user=sender,
                    identification_prompt=Prompt.MESSAGE_ENTITY_IDENTIFICATION_PROMPT,
                    fact_prompt=Prompt.MESSAGE_FACT_EXTRACTION_PROMPT,
                    context_label="User message",
                    context_value=message.content,
                    content=message.content,
                    source_message_id=message.id,
                )
                if entities:
                    work_done = True

                    # Create MESSAGE_MENTION engagements for identified entities
                    for entity in entities:
                        assert entity.id is not None
                        self.db.add_engagement(
                            user=sender,
                            engagement_type=PennyConstants.EngagementType.MESSAGE_MENTION,
                            valence=PennyConstants.EngagementValence.NEUTRAL,
                            strength=PennyConstants.ENGAGEMENT_STRENGTH_MESSAGE_MENTION,
                            entity_id=entity.id,
                            source_message_id=message.id,
                        )

            # --- Preference extraction from reactions + messages ---
            like_reaction_texts: list[str] = []
            dislike_reaction_texts: list[str] = []
            for reaction in reactions:
                emoji = reaction.content
                if (
                    emoji not in PennyConstants.LIKE_REACTIONS
                    and emoji not in PennyConstants.DISLIKE_REACTIONS
                ):
                    continue
                if not reaction.parent_id:
                    continue
                parent_msg = self.db.get_message_by_id(reaction.parent_id)
                if not parent_msg:
                    continue
                if emoji in PennyConstants.LIKE_REACTIONS:
                    like_reaction_texts.append(parent_msg.content)
                else:
                    dislike_reaction_texts.append(parent_msg.content)

            user_message_texts = [msg.content for msg in messages]

            for pref_type, reaction_texts in [
                (PennyConstants.PreferenceType.LIKE, like_reaction_texts),
                (PennyConstants.PreferenceType.DISLIKE, dislike_reaction_texts),
            ]:
                if not reaction_texts and not user_message_texts:
                    continue

                updated = await self._extract_and_store_preferences(
                    sender, pref_type, reaction_texts, user_message_texts
                )
                if updated:
                    work_done = True

            # Mark all reactions and messages as processed
            reaction_ids = [r.id for r in reactions if r.id is not None]
            message_ids = [m.id for m in messages if m.id is not None]
            if reaction_ids:
                self.db.mark_messages_processed(reaction_ids)
            if message_ids:
                self.db.mark_messages_processed(message_ids)

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
    ) -> list[Entity]:
        """Two-pass extraction for a single piece of content.

        Pass 1: Identify known + new entities in the text.
        Pass 2: For each entity, extract new facts as individual Fact rows.

        Returns list of entities that were created or updated (empty if none).
        """
        existing_entities = self.db.get_user_entities(user)

        # Pass 1: identify entities
        identified = await self._identify_entities(
            existing_entities, identification_prompt, context_label, context_value, content
        )
        if not identified:
            return []

        # Collect all entities to process facts for
        entities_to_process: list[Entity] = []

        # Store new entities
        for new_entity in identified.new:
            name = new_entity.name.lower().strip()
            if not name:
                continue
            entity = self.db.get_or_create_entity(user, name)
            if entity and entity.id is not None:
                entities_to_process.append(entity)
                logger.info("New entity discovered: '%s'", name)

        # Look up known entities that were identified
        existing_by_name = {e.name: e for e in existing_entities}
        for known_name in identified.known:
            normalized = known_name.lower().strip()
            if normalized in existing_by_name:
                entities_to_process.append(existing_by_name[normalized])
                logger.info("Known entity referenced: '%s'", normalized)

        # Pass 2: extract facts for each identified entity
        entities_with_new_facts: list[int] = []
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

        # Regenerate entity embeddings for entities that got new facts
        if self.embedding_model and entities_with_new_facts:
            await self._update_entity_embeddings(
                [e for e in entities_to_process if e.id in entities_with_new_facts]
            )

        return entities_to_process

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

    # --- Preference extraction ---

    async def _extract_and_store_preferences(
        self,
        sender: str,
        pref_type: str,
        reaction_texts: list[str],
        user_message_texts: list[str],
    ) -> bool:
        """Run a single LLM pass for one preference type (like or dislike).

        Returns True if any new preferences were added.
        """
        existing = self.db.get_preferences(sender, pref_type)
        existing_topics = [p.topic for p in existing]

        sentiment_desc = (
            "enjoys or is enthusiastic about"
            if pref_type == PennyConstants.PreferenceType.LIKE
            else "dislikes or expresses negativity toward"
        )

        prompt_parts = [
            f"Find any NEW topics the user {pref_type}s from the messages below.",
            f"Only extract clear {pref_type}s — things the user explicitly {sentiment_desc}.",
            "Do NOT extract every noun — only genuine preferences.",
            "Return short phrases (1-4 words each).\n",
        ]

        if existing_topics:
            prompt_parts.append(f"Already known {pref_type}s: {', '.join(existing_topics)}")
            prompt_parts.append("Do NOT include topics already known above.\n")

        if reaction_texts:
            prompt_parts.append(f"Messages the user reacted to with a {pref_type} emoji:")
            for text in reaction_texts:
                prompt_parts.append(f'- "{text}"')
            prompt_parts.append("")

        if user_message_texts:
            prompt_parts.append("Messages from the user:")
            for text in user_message_texts:
                prompt_parts.append(f'- "{text}"')

        prompt = "\n".join(prompt_parts)

        try:
            response = await self._ollama_client.generate(
                prompt=prompt,
                tools=None,
                format=ExtractedTopics.model_json_schema(),
            )
            result = ExtractedTopics.model_validate_json(response.content)

            # Collect new topics (deduped against existing)
            new_topics: list[str] = []
            for raw_topic in result.topics:
                topic = raw_topic.lower().strip()
                if not topic:
                    continue
                if any(p.topic == topic for p in existing):
                    continue
                new_topics.append(topic)

            # Batch-embed all new topics at once
            topic_embeddings: list[bytes | None] = [None] * len(new_topics)
            if new_topics and self.embedding_model:
                try:
                    vecs = await self._ollama_client.embed(new_topics, model=self.embedding_model)
                    topic_embeddings = [serialize_embedding(v) for v in vecs]
                except Exception as e:
                    logger.warning("Failed to embed preference topics: %s", e)

            added_preferences: list[Preference] = []
            for topic, emb in zip(new_topics, topic_embeddings, strict=True):
                pref = self.db.add_preference(sender, topic, pref_type, embedding=emb)
                logger.info("Added %s preference for %s: %s", pref_type, sender, topic)
                if pref:
                    added_preferences.append(pref)

            # Link new preferences to existing entities via embedding similarity
            for pref in added_preferences:
                await self._link_preference_to_entities(sender, pref)

            # Send a single batched notification for all new preferences
            if added_preferences:
                topics = [p.topic for p in added_preferences]
                if len(topics) == 1:
                    message = f"I added {topics[0]} to your {pref_type}s"
                else:
                    bullet_list = "\n".join(f"• {topic}" for topic in topics)
                    message = f"I added these to your {pref_type}s:\n{bullet_list}"
                await self._send_notification(sender, message)

            return len(added_preferences) > 0

        except Exception as e:
            logger.error("Failed to extract %s preferences for %s: %s", pref_type, sender, e)
            return False

    async def _link_preference_to_entities(self, sender: str, preference: Preference) -> None:
        """Find entities similar to a new preference and create engagements."""
        if not self.embedding_model or not preference.embedding:
            return

        entities = self.db.get_user_entities_with_embeddings(sender)
        if not entities:
            return

        query_vec = deserialize_embedding(preference.embedding)
        candidates = [
            (e.id, deserialize_embedding(e.embedding))
            for e in entities
            if e.id is not None and e.embedding is not None
        ]
        if not candidates:
            return

        matches = find_similar(
            query_vec,
            candidates,
            top_k=PennyConstants.ENTITY_CONTEXT_TOP_K,
            threshold=PennyConstants.PREFERENCE_ENTITY_LINK_THRESHOLD,
        )

        valence = (
            PennyConstants.EngagementValence.POSITIVE
            if preference.type == PennyConstants.PreferenceType.LIKE
            else PennyConstants.EngagementValence.NEGATIVE
        )

        for entity_id, _score in matches:
            self.db.add_engagement(
                user=sender,
                engagement_type=PennyConstants.EngagementType.EXPLICIT_STATEMENT,
                valence=valence,
                strength=PennyConstants.ENGAGEMENT_STRENGTH_EXPLICIT_STATEMENT,
                entity_id=entity_id,
                preference_id=preference.id,
            )
            logger.info(
                "Linked preference '%s' to entity %d (valence=%s)",
                preference.topic,
                entity_id,
                valence,
            )

    # --- Message filtering ---

    @staticmethod
    def _should_process_message(message: MessageLog) -> bool:
        """Lightweight pre-filter to skip low-signal messages before LLM calls."""
        content = message.content.strip()
        if len(content) < PennyConstants.MIN_EXTRACTION_MESSAGE_LENGTH:
            return False
        return not content.startswith("/")

    # --- Notifications ---

    async def _send_notification(self, recipient: str, message: str) -> None:
        """Send a notification message to the user."""
        if not self._channel:
            return

        typing_task = asyncio.create_task(self._channel._typing_loop(recipient))
        try:
            await self._channel.send_response(
                recipient,
                message,
                parent_id=None,
                attachments=None,
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

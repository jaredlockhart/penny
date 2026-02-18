"""EntityExtractor agent for building entity knowledge base from search results."""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

from penny.agents.base import Agent
from penny.constants import PennyConstants
from penny.database.models import Entity
from penny.ollama.embeddings import build_entity_embed_text, serialize_embedding
from penny.prompts import Prompt

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


class IdentifiedNewEntity(BaseModel):
    """A newly discovered entity from pass 1."""

    name: str = Field(description="Entity name (e.g., 'KEF LS50 Meta', 'NVIDIA Jetson')")


# Rebuild IdentifiedEntities now that IdentifiedNewEntity is defined
IdentifiedEntities.model_rebuild()


class ExtractedFacts(BaseModel):
    """Schema for pass 2: new facts about a single entity."""

    facts: list[str] = Field(
        default_factory=list,
        description="NEW specific, verifiable facts about the entity from the text",
    )


class EntityExtractor(Agent):
    """Background agent that extracts entities and facts from search results."""

    @property
    def name(self) -> str:
        """Task name for logging."""
        return "entity_extractor"

    async def execute(self) -> bool:
        """
        Process unprocessed SearchLog entries to extract entities and facts.

        Two-pass extraction per entry:
        Pass 1 (identification): Given the list of known entity names, identify
        which known entities appear in the text and any new entities.
        Pass 2 (facts): For each identified entity, given its existing facts,
        extract only new facts from the text.

        Note: ResearchIteration entries are NOT processed because their findings
        are LLM-synthesized reports of the same search results already in SearchLog.

        Returns:
            True if any work was done.
        """
        work_done = False

        search_logs = self.db.get_unprocessed_search_logs(
            limit=PennyConstants.ENTITY_EXTRACTION_BATCH_LIMIT
        )
        if search_logs:
            logger.info("Processing %d unprocessed search logs", len(search_logs))
            for search_log in search_logs:
                assert search_log.id is not None

                user = self.db.find_sender_for_timestamp(search_log.timestamp)
                if not user:
                    self.db.mark_search_extracted(search_log.id)
                    continue

                logger.info("Extracting entities from search: %s", search_log.query)
                result = await self._extract_and_store(
                    user=user,
                    query=search_log.query,
                    content=search_log.response,
                    search_log_id=search_log.id,
                )
                if result:
                    work_done = True

        # Backfill embeddings for existing entities/facts without them
        if self.embedding_model:
            backfilled = await self._backfill_embeddings()
            if backfilled:
                work_done = True

        return work_done

    async def _extract_and_store(
        self, user: str, query: str, content: str, search_log_id: int
    ) -> bool:
        """
        Two-pass extraction for a single piece of content.

        Pass 1: Identify known + new entities in the text.
        Pass 2: For each entity, extract new facts as individual Fact rows.

        Returns True if any new entities or facts were stored.
        """
        existing_entities = self.db.get_user_entities(user)

        # Pass 1: identify entities
        identified = await self._identify_entities(existing_entities, query, content)
        if not identified:
            self.db.mark_search_extracted(search_log_id)
            return False

        work_done = False

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
                work_done = True
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
            new_facts = await self._extract_facts(entity, query, content)
            if not new_facts:
                continue

            # Dedup against existing Fact rows
            existing_fact_rows = self.db.get_entity_facts(entity.id)
            existing_normalized = {_normalize_fact(f.content) for f in existing_fact_rows}

            # Collect new (non-duplicate) fact texts
            new_fact_texts: list[str] = []
            for fact_text in new_facts:
                fact_text = fact_text.strip()
                if not fact_text:
                    continue
                if _normalize_fact(fact_text) in existing_normalized:
                    continue
                new_fact_texts.append(fact_text)
                existing_normalized.add(_normalize_fact(fact_text))

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
                    source_search_log_id=search_log_id,
                    embedding=emb,
                )
                logger.info("  '%s' +fact: %s", entity.name, fact_text)
                work_done = True

            entities_with_new_facts.append(entity.id)

        # Regenerate entity embeddings for entities that got new facts
        if self.embedding_model and entities_with_new_facts:
            await self._update_entity_embeddings(
                [e for e in entities_to_process if e.id in entities_with_new_facts]
            )

        # Mark search as processed
        self.db.mark_search_extracted(search_log_id)

        return work_done

    async def _identify_entities(
        self,
        existing_entities: list[Entity],
        query: str,
        content: str,
    ) -> IdentifiedEntities | None:
        """
        Pass 1: Identify which known entities appear in the text and any new entities.

        Injects only entity names (not facts) to keep the prompt small.
        """
        known_names = [e.name for e in existing_entities]
        known_context = ""
        if known_names:
            known_context = (
                "\n\nKnown entities (return any that appear in the text):\n"
                + "\n".join(f"- {name}" for name in known_names)
            )

        prompt = (
            f"{Prompt.ENTITY_IDENTIFICATION_PROMPT}\n\n"
            f"Search query: {query}\n\n"
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

    async def _extract_facts(self, entity: Entity, query: str, content: str) -> list[str]:
        """
        Pass 2: Extract new facts about a specific entity from content.

        Injects the entity's existing facts so the LLM only returns new ones.
        """
        assert entity.id is not None
        existing_fact_rows = self.db.get_entity_facts(entity.id)
        existing_context = ""
        if existing_fact_rows:
            facts_text = "\n".join(f"- {f.content}" for f in existing_fact_rows)
            existing_context = (
                f"\n\nAlready known facts (return only NEW facts not listed here):\n{facts_text}"
            )

        prompt = (
            f"{Prompt.ENTITY_FACT_EXTRACTION_PROMPT}\n\n"
            f"Entity: {entity.name}\n\n"
            f"Search query: {query}\n\n"
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

    async def _backfill_embeddings(self) -> bool:
        """Backfill embeddings for entities and facts that don't have them.

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

        return work_done

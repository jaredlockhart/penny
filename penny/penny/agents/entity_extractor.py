"""EntityExtractor agent for building entity knowledge base from search results."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from penny.agents.base import Agent
from penny.constants import PennyConstants
from penny.database.models import Entity
from penny.prompts import Prompt

logger = logging.getLogger(__name__)


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
        search_logs = self.db.get_unprocessed_search_logs(
            limit=PennyConstants.ENTITY_EXTRACTION_BATCH_LIMIT
        )
        if not search_logs:
            return False

        logger.info("Processing %d unprocessed search logs", len(search_logs))
        work_done = False
        for search_log in search_logs:
            assert search_log.id is not None

            user = self.db.find_sender_for_timestamp(search_log.timestamp)
            if not user:
                self.db.link_entity_to_search_log(None, search_log.id)
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

        return work_done

    async def _extract_and_store(
        self, user: str, query: str, content: str, search_log_id: int
    ) -> bool:
        """
        Two-pass extraction for a single piece of content.

        Pass 1: Identify known + new entities in the text.
        Pass 2: For each entity, extract new facts.

        Returns True if any new entities or facts were stored.
        """
        existing_entities = self.db.get_user_entities(user)

        # Pass 1: identify entities
        identified = await self._identify_entities(existing_entities, query, content)
        if not identified:
            self.db.link_entity_to_search_log(None, search_log_id)
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
        for entity in entities_to_process:
            assert entity.id is not None
            self.db.link_entity_to_search_log(entity.id, search_log_id)
            new_facts = await self._extract_facts(entity, query, content)
            if not new_facts:
                continue

            # Merge new facts (dedup in Python-space)
            existing_facts = {
                line.strip() for line in entity.facts.strip().split("\n") if line.strip()
            }
            genuinely_new = [
                f"- {fact.strip()}"
                for fact in new_facts
                if fact.strip() and f"- {fact.strip()}" not in existing_facts
            ]

            if genuinely_new:
                all_facts = list(existing_facts) + genuinely_new
                self.db.update_entity_facts(entity.id, "\n".join(all_facts))
                for fact in genuinely_new:
                    logger.info("  '%s' +fact: %s", entity.name, fact)
                work_done = True

        # If no entities were linked (all names were empty/invalid), insert sentinel
        if not entities_to_process:
            self.db.link_entity_to_search_log(None, search_log_id)

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
        existing_context = ""
        if entity.facts.strip():
            existing_context = (
                f"\n\nAlready known facts (return only NEW facts not listed here):\n{entity.facts}"
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

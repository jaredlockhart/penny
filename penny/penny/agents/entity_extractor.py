"""EntityExtractor agent for building entity knowledge base from search results."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from penny.agents.base import Agent
from penny.constants import PennyConstants
from penny.prompts import Prompt

logger = logging.getLogger(__name__)


class ExtractedEntity(BaseModel):
    """A single entity extracted from text."""

    name: str = Field(description="Entity name (e.g., 'KEF LS50 Meta', 'NVIDIA Jetson')")
    entity_type: str = Field(
        description="Type: product, person, place, concept, organization, event"
    )
    facts: list[str] = Field(
        default_factory=list,
        description="Key facts about this entity from the text (specific, verifiable statements)",
    )


class ExtractedEntities(BaseModel):
    """Schema for LLM response: entities and their facts from search results."""

    entities: list[ExtractedEntity] = Field(
        default_factory=list,
        description="Entities found in the text with their key facts",
    )


class EntityExtractor(Agent):
    """Background agent that extracts entities and facts from search results."""

    @property
    def name(self) -> str:
        """Task name for logging."""
        return "entity_extractor"

    async def execute(self) -> bool:
        """
        Process unprocessed SearchLog and ResearchIteration entries
        to extract entities and facts.

        Returns:
            True if any work was done.
        """
        search_work = await self._process_search_logs()
        research_work = await self._process_research_iterations()
        return search_work or research_work

    async def _process_search_logs(self) -> bool:
        """Process unprocessed SearchLog entries."""
        search_logs = self.db.get_unprocessed_search_logs(
            limit=PennyConstants.ENTITY_EXTRACTION_BATCH_LIMIT
        )
        if not search_logs:
            return False

        work_done = False
        for search_log in search_logs:
            assert search_log.id is not None

            user = self.db.find_sender_for_timestamp(search_log.timestamp)
            if not user:
                self.db.update_extraction_cursor("search", search_log.id)
                continue

            extracted = await self._extract_entities(
                user=user,
                query=search_log.query,
                content=search_log.response,
            )

            if extracted:
                self._store_entities(user, extracted)
                work_done = True

            self.db.update_extraction_cursor("search", search_log.id)

        return work_done

    async def _process_research_iterations(self) -> bool:
        """Process unprocessed ResearchIteration entries."""
        iterations = self.db.get_unprocessed_research_iterations(
            limit=PennyConstants.ENTITY_EXTRACTION_BATCH_LIMIT
        )
        if not iterations:
            return False

        work_done = False
        for iteration, user in iterations:
            assert iteration.id is not None

            extracted = await self._extract_entities(
                user=user,
                query=f"Research iteration {iteration.iteration_num}",
                content=iteration.findings,
            )

            if extracted:
                self._store_entities(user, extracted)
                work_done = True

            self.db.update_extraction_cursor("research", iteration.id)

        return work_done

    async def _extract_entities(self, user: str, query: str, content: str) -> list[ExtractedEntity]:
        """
        Use LLM structured output to extract entities from content.

        Includes existing entity facts in the prompt so the LLM can
        avoid extracting duplicate information.
        """
        # Build context of existing entities for dedup
        existing_entities = self.db.get_user_entities(user)
        existing_context = ""
        if existing_entities:
            entity_summaries = []
            for entity in existing_entities:
                if entity.facts.strip():
                    entity_summaries.append(
                        f"{entity.name} ({entity.entity_type}):\n{entity.facts}"
                    )
                else:
                    entity_summaries.append(f"{entity.name} ({entity.entity_type}): (no facts yet)")
            existing_context = (
                "\n\nAlready known entities and facts (extract only NEW information):\n"
                + "\n\n".join(entity_summaries)
            )

        prompt = (
            f"{Prompt.ENTITY_EXTRACTION_PROMPT}\n\n"
            f"Search query: {query}\n\n"
            f"Content:\n{content}"
            f"{existing_context}"
        )

        try:
            response = await self._ollama_client.generate(
                prompt=prompt,
                tools=None,
                format=ExtractedEntities.model_json_schema(),
            )
            result = ExtractedEntities.model_validate_json(response.content)
            return result.entities
        except Exception as e:
            logger.error("Failed to extract entities: %s", e)
            return []

    def _store_entities(self, user: str, entities: list[ExtractedEntity]) -> None:
        """Store extracted entities and merge new facts into existing ones."""
        valid_types = {t.value for t in PennyConstants.EntityType}

        for extracted in entities:
            name = extracted.name.lower().strip()
            if not name:
                continue

            entity_type = extracted.entity_type.lower().strip()
            if entity_type not in valid_types:
                entity_type = PennyConstants.EntityType.CONCEPT

            entity = self.db.get_or_create_entity(user, name, entity_type)
            if not entity or entity.id is None:
                continue

            # Parse existing facts and merge new ones
            existing_facts = {
                line.strip() for line in entity.facts.strip().split("\n") if line.strip()
            }
            new_facts = [
                f"- {fact.strip()}"
                for fact in extracted.facts
                if fact.strip() and f"- {fact.strip()}" not in existing_facts
            ]

            if new_facts:
                all_facts = list(existing_facts) + new_facts
                self.db.update_entity_facts(entity.id, "\n".join(all_facts))
                logger.info(
                    "Added %d facts to entity '%s' for user %s",
                    len(new_facts),
                    name,
                    user,
                )

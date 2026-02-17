"""EntityCleaner agent for merging duplicate entities in the knowledge base."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, Field

from penny.agents.base import Agent
from penny.agents.entity_extractor import _normalize_fact
from penny.constants import PennyConstants
from penny.prompts import Prompt

logger = logging.getLogger(__name__)


class MergeGroup(BaseModel):
    """A group of duplicate entities to merge."""

    canonical_name: str = Field(description="The name to keep (short, canonical form)")
    duplicates: list[str] = Field(description="Entity names to merge into the canonical name")


class MergeGroups(BaseModel):
    """LLM response: groups of entities that should be merged."""

    groups: list[MergeGroup] = Field(
        default_factory=list,
        description="Groups of duplicate entities to merge",
    )


class EntityCleaner(Agent):
    """Background agent that periodically merges duplicate entities."""

    @property
    def name(self) -> str:
        """Task name for logging."""
        return "entity_cleaner"

    async def execute(self) -> bool:
        """
        Merge duplicate entities in the knowledge base.

        Checks a DB-stored timestamp to enforce a minimum interval between
        cleaning runs. If enough time has passed, sends entity names to the
        LLM to identify merge candidates, then merges them in Python-space.

        Returns:
            True if any merges were performed.
        """
        # Check if enough time has passed since last cleaning
        last_cleaned = self.db.get_entity_cleaning_timestamp()
        if last_cleaned:
            interval = timedelta(seconds=PennyConstants.ENTITY_CLEANING_INTERVAL_SECONDS)
            if datetime.now(UTC) - last_cleaned < interval:
                return False

        senders = self.db.get_all_senders()
        if not senders:
            return False

        work_done = False
        for sender in senders:
            entities = self.db.get_user_entities(sender)
            if len(entities) < 2:
                continue

            merged = await self._clean_user_entities(sender, entities)
            if merged:
                work_done = True

        # Always update timestamp, even if no merges were needed
        self.db.set_entity_cleaning_timestamp(datetime.now(UTC))
        return work_done

    async def _clean_user_entities(self, user: str, entities: list) -> bool:
        """
        Identify and merge duplicate entities for a single user.

        Args:
            user: User identifier
            entities: All entities for the user

        Returns:
            True if any merges were performed
        """
        # Build name list for LLM (respect batch limit)
        names = [e.name for e in entities[: PennyConstants.ENTITY_CLEANING_BATCH_LIMIT]]

        merge_groups = await self._identify_merge_groups(names)
        if not merge_groups:
            return False

        # Build lookup by name
        entities_by_name = {e.name: e for e in entities}

        work_done = False
        for group in merge_groups:
            canonical = group.canonical_name.lower().strip()
            duplicates = [d.lower().strip() for d in group.duplicates]

            # Resolve entity objects
            primary = entities_by_name.get(canonical)
            dup_entities = [
                entities_by_name[d] for d in duplicates if d in entities_by_name and d != canonical
            ]

            if not primary or not dup_entities:
                # If canonical doesn't exist, try to use the first duplicate as primary
                if not primary and dup_entities:
                    primary = dup_entities.pop(0)
                if not primary or not dup_entities:
                    continue

            assert primary.id is not None

            # Combine facts from all entities (dedup with normalization)
            all_fact_lines: list[str] = []
            seen_normalized: set[str] = set()
            for entity in [primary, *dup_entities]:
                for line in entity.facts.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    normalized = _normalize_fact(line)
                    if normalized not in seen_normalized:
                        seen_normalized.add(normalized)
                        all_fact_lines.append(line)

            merged_facts = "\n".join(all_fact_lines)
            dup_ids = [e.id for e in dup_entities if e.id is not None]

            logger.info(
                "Merging entities %s into '%s' (id=%d) for user %s",
                [e.name for e in dup_entities],
                primary.name,
                primary.id,
                user,
            )

            self.db.merge_entities(primary.id, dup_ids, merged_facts)
            work_done = True

        return work_done

    async def _identify_merge_groups(self, names: list[str]) -> list[MergeGroup]:
        """
        Ask the LLM to identify groups of duplicate entity names.

        Args:
            names: List of entity names

        Returns:
            List of merge groups, or empty list if none found
        """
        name_list = "\n".join(f"- {name}" for name in names)
        prompt = f"{Prompt.ENTITY_MERGE_PROMPT}\n\nEntity names:\n{name_list}"

        try:
            response = await self._ollama_client.generate(
                prompt=prompt,
                tools=None,
                format=MergeGroups.model_json_schema(),
            )
            result = MergeGroups.model_validate_json(response.content)
            return result.groups
        except Exception as e:
            logger.error("Failed to identify merge groups: %s", e)
            return []

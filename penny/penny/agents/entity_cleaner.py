"""EntityCleaner agent for merging duplicate entities in the knowledge base."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, Field

from penny.agents.base import Agent
from penny.agents.entity_extractor import _normalize_fact
from penny.constants import PennyConstants
from penny.database.models import Fact
from penny.ollama.embeddings import build_entity_embed_text, serialize_embedding
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

            # Collect all facts from primary and duplicates, dedup by content
            all_facts: list[Fact] = self.db.get_entity_facts(primary.id)
            for dup in dup_entities:
                assert dup.id is not None
                all_facts.extend(self.db.get_entity_facts(dup.id))

            keep_fact_ids: list[int] = []
            seen_normalized: set[str] = set()
            for fact in all_facts:
                assert fact.id is not None
                normalized = _normalize_fact(fact.content)
                if normalized not in seen_normalized:
                    seen_normalized.add(normalized)
                    keep_fact_ids.append(fact.id)

            dup_ids = [e.id for e in dup_entities if e.id is not None]

            logger.info(
                "Merging entities %s into '%s' (id=%d) for user %s",
                [e.name for e in dup_entities],
                primary.name,
                primary.id,
                user,
            )

            self.db.merge_entities(primary.id, dup_ids, keep_fact_ids)
            work_done = True

            # Regenerate embedding for merged entity (facts changed)
            if self.embedding_model:
                await self._regenerate_entity_embedding(primary.id, primary.name)

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

    async def _regenerate_entity_embedding(self, entity_id: int, entity_name: str) -> None:
        """Regenerate the embedding for an entity after its facts changed."""
        assert self.embedding_model is not None
        try:
            facts = self.db.get_entity_facts(entity_id)
            embed_text = build_entity_embed_text(entity_name, [f.content for f in facts])
            vecs = await self._ollama_client.embed(embed_text, model=self.embedding_model)
            self.db.update_entity_embedding(entity_id, serialize_embedding(vecs[0]))
            logger.debug("Regenerated entity embedding for '%s' after merge", entity_name)
        except Exception as e:
            logger.warning("Failed to regenerate entity %d embedding: %s", entity_id, e)

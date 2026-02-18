"""/learn command — search-based entity discovery for background research."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from penny.agents.extraction import IdentifiedEntities
from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.constants import PennyConstants
from penny.database.models import Engagement
from penny.interest import compute_interest_score
from penny.prompts import Prompt
from penny.responses import PennyResponse
from penny.tools.models import SearchResult

if TYPE_CHECKING:
    from penny.tools import Tool

logger = logging.getLogger(__name__)


class LearnCommand(Command):
    """Search for a topic, discover entities, and track them for background research."""

    name = "learn"
    description = "Start learning about a topic"
    help_text = (
        "Express interest in a topic so Penny researches it in the background.\n\n"
        "**Usage**:\n"
        "- `/learn` — List what's being actively tracked\n"
        "- `/learn <topic>` — Start learning about a topic\n\n"
        "**Examples**:\n"
        "- `/learn kef ls50 meta`\n"
        "- `/learn travel in china 2026`"
    )

    def __init__(self, search_tool: Tool | None = None) -> None:
        self._search_tool = search_tool

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute learn command."""
        topic = args.strip()

        if not topic:
            return self._list_tracked(context)

        return await self._learn_topic(topic, context)

    def _list_tracked(self, context: CommandContext) -> CommandResult:
        """List entities currently being tracked (positive interest score)."""
        entities = context.db.get_user_entities(context.user)
        if not entities:
            return CommandResult(text=PennyResponse.LEARN_EMPTY)

        all_engagements = context.db.get_user_engagements(context.user)
        engagements_by_entity: dict[int, list[Engagement]] = defaultdict(list)
        for eng in all_engagements:
            if eng.entity_id is not None:
                engagements_by_entity[eng.entity_id].append(eng)

        scored: list[tuple[str, float, int]] = []
        for entity in entities:
            assert entity.id is not None
            entity_engagements = engagements_by_entity.get(entity.id, [])
            score = compute_interest_score(entity_engagements)
            if score <= 0:
                continue
            fact_count = len(context.db.get_entity_facts(entity.id))
            scored.append((entity.name, score, fact_count))

        if not scored:
            return CommandResult(text=PennyResponse.LEARN_EMPTY)

        scored.sort(key=lambda x: x[1], reverse=True)

        lines = [PennyResponse.LEARN_LIST_HEADER, ""]
        for i, (name, score, fact_count) in enumerate(scored, 1):
            facts_label = f"{fact_count} fact{'s' if fact_count != 1 else ''}"
            lines.append(f"{i}. **{name}** (+{score:.2f}) — {facts_label}")

        return CommandResult(text="\n".join(lines))

    async def _learn_topic(self, topic: str, context: CommandContext) -> CommandResult:
        """Search for topic, discover entities, record learn engagements."""
        # Without search tool, fall back to creating entity from topic text
        if not self._search_tool:
            logger.warning("LearnCommand: no search tool, creating entity directly")
            return await self._fallback_create_entity(topic, context)

        # Search for the topic
        search_text = await self._search(topic)
        if not search_text:
            return await self._fallback_create_entity(topic, context)

        # Identify entities from search results
        entity_names = await self._identify_entities(search_text, topic, context)
        if not entity_names:
            return CommandResult(text=PennyResponse.LEARN_NO_ENTITIES_FOUND)

        # Create entities and record engagements
        created_names = self._create_entities_with_engagements(entity_names, context)

        if not created_names:
            return CommandResult(text=PennyResponse.LEARN_NO_ENTITIES_FOUND)

        # Build response with entity list
        lines = [PennyResponse.LEARN_DISCOVERED, ""]
        for i, name in enumerate(created_names, 1):
            lines.append(f"{i}. **{name}**")

        return CommandResult(text="\n".join(lines))

    async def _search(self, query: str) -> str | None:
        """Execute search via SearchTool. Returns text or None."""
        assert self._search_tool is not None
        try:
            original_skip = getattr(self._search_tool, "skip_images", False)
            self._search_tool.skip_images = True  # type: ignore[attr-defined]
            try:
                result = await self._search_tool.execute(query=query)
            finally:
                self._search_tool.skip_images = original_skip  # type: ignore[attr-defined]

            if isinstance(result, SearchResult):
                return result.text
            return str(result) if result else None
        except Exception as e:
            logger.error("Learn command search failed: %s", e)
            return None

    async def _identify_entities(
        self, search_text: str, topic: str, context: CommandContext
    ) -> list[str]:
        """Extract entity names from search results via LLM."""
        existing_entities = context.db.get_user_entities(context.user)
        known_names = [e.name for e in existing_entities]
        known_context = ""
        if known_names:
            known_context = (
                "\n\nKnown entities (return any that appear in the text):\n"
                + "\n".join(f"- {name}" for name in known_names)
            )

        prompt = (
            f"{Prompt.ENTITY_IDENTIFICATION_PROMPT}\n\n"
            f"Search query: {topic}\n\n"
            f"Content:\n{search_text}"
            f"{known_context}"
        )

        try:
            response = await context.ollama_client.generate(
                prompt=prompt,
                tools=None,
                format=IdentifiedEntities.model_json_schema(),
            )
            result = IdentifiedEntities.model_validate_json(response.content)

            names: list[str] = []
            for name in result.known:
                if name not in names:
                    names.append(name)
            for new_entity in result.new:
                if new_entity.name not in names:
                    names.append(new_entity.name)
            return names
        except Exception as e:
            logger.error("Failed to identify entities for learn '%s': %s", topic, e)
            return []

    def _create_entities_with_engagements(
        self, entity_names: list[str], context: CommandContext
    ) -> list[str]:
        """Create entities and record LEARN_COMMAND engagements. Returns created names."""
        created: list[str] = []
        for name in entity_names:
            entity = context.db.get_or_create_entity(context.user, name)
            if entity is None or entity.id is None:
                continue

            context.db.add_engagement(
                user=context.user,
                engagement_type=PennyConstants.EngagementType.LEARN_COMMAND,
                valence=PennyConstants.EngagementValence.POSITIVE,
                strength=PennyConstants.ENGAGEMENT_STRENGTH_LEARN_COMMAND,
                entity_id=entity.id,
            )
            created.append(entity.name)

        return created

    async def _fallback_create_entity(self, topic: str, context: CommandContext) -> CommandResult:
        """Fallback: create entity directly from topic text (no search)."""
        entity = context.db.get_or_create_entity(context.user, topic)
        if entity is None or entity.id is None:
            return CommandResult(text="Sorry, I couldn't create that topic.")

        context.db.add_engagement(
            user=context.user,
            engagement_type=PennyConstants.EngagementType.LEARN_COMMAND,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=PennyConstants.ENGAGEMENT_STRENGTH_LEARN_COMMAND,
            entity_id=entity.id,
        )

        if self._search_tool:
            return CommandResult(
                text=PennyResponse.LEARN_SEARCH_FAILED.format(entity_name=entity.name)
            )
        return CommandResult(text=PennyResponse.LEARN_NO_SEARCH.format(entity_name=entity.name))

"""Memory command — /memory."""

from __future__ import annotations

import logging

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.database.models import Entity
from penny.responses import PennyResponse

logger = logging.getLogger(__name__)


class MemoryCommand(Command):
    """View Penny's knowledge base."""

    name = "memory"
    description = "View what Penny has remembered"
    help_text = (
        "View what Penny remembers from conversations and searches.\n\n"
        "**Usage**:\n"
        "• `/memory` — List all remembered entities\n"
        "• `/memory <number>` — Show details for an entity\n\n"
        "To delete a memory, use `/forget <number>`.\n\n"
        "**Examples**:\n"
        "• `/memory`\n"
        "• `/memory 1`"
    )

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute memory command."""
        args = args.strip()
        parts = args.split() if args else []

        # No args — list all entities sorted by recency
        if not parts:
            entities = self._sorted_entities(context)
            if not entities:
                return CommandResult(text=PennyResponse.MEMORY_EMPTY)

            lines = [PennyResponse.MEMORY_LIST_HEADER, ""]
            for i, entity in enumerate(entities, 1):
                assert entity.id is not None
                facts = context.db.facts.get_for_entity(entity.id)
                facts_label = f"{len(facts)} fact{'s' if len(facts) != 1 else ''}"
                tagline_suffix = f" — {entity.tagline}" if entity.tagline else ""
                name_part = f"{i}. **{entity.name}**{tagline_suffix}"
                lines.append(f"{name_part} ({facts_label})")
            return CommandResult(text="\n".join(lines))

        # First arg must be a number
        if not parts[0].isdigit():
            return CommandResult(text=PennyResponse.MEMORY_ENTITY_NOT_FOUND.format(number=parts[0]))

        position = int(parts[0])
        entities = self._sorted_entities(context)

        if position < 1 or position > len(entities):
            return CommandResult(text=PennyResponse.MEMORY_ENTITY_NOT_FOUND.format(number=position))

        entity = entities[position - 1]
        assert entity.id is not None

        # Number only — show entity details
        facts = context.db.facts.get_for_entity(entity.id)
        if not facts:
            return CommandResult(text=PennyResponse.MEMORY_NO_FACTS.format(name=entity.name))

        updated = entity.updated_at.strftime("%Y-%m-%d %H:%M")
        origin = self._get_origin(facts, context)
        facts_text = "\n\n".join(f"• {f.content}" for f in facts)
        lines = [
            f"**{entity.name}**",
        ]
        if entity.tagline:
            lines.append(f"*{entity.tagline}*")
        lines.append(f"**Updated**: {updated}")
        if origin:
            lines.append(f"**Origin**: {origin}")
        lines += ["", facts_text]
        return CommandResult(text="\n".join(lines))

    @staticmethod
    def _get_origin(facts: list, context: CommandContext) -> str | None:
        """Trace facts back to their originating search query."""
        for fact in facts:
            if fact.source_search_log_id is None:
                continue
            search_log = context.db.searches.get(fact.source_search_log_id)
            if search_log is None:
                continue
            return f"search: {search_log.query[:80]}"
        return None

    @staticmethod
    def _sorted_entities(context: CommandContext) -> list[Entity]:
        """Return entities sorted by created_at descending."""
        entities = context.db.entities.get_for_user(context.user)
        return sorted(entities, key=lambda e: e.created_at, reverse=True)

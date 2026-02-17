"""Memory command — /memory."""

from __future__ import annotations

import logging

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.responses import PennyResponse

logger = logging.getLogger(__name__)


def _count_facts(facts: str) -> int:
    """Count the number of bullet-point facts in an entity's facts string."""
    if not facts.strip():
        return 0
    return sum(1 for line in facts.splitlines() if line.strip().startswith("- "))


class MemoryCommand(Command):
    """View or manage Penny's knowledge base."""

    name = "memory"
    description = "View or manage Penny's knowledge base"
    help_text = (
        "View what Penny has learned from searches, or manage stored knowledge.\n\n"
        "**Usage**:\n"
        "- `/memory` — List all remembered entities\n"
        "- `/memory <number>` — Show details for an entity\n"
        "- `/memory <number> delete` — Delete an entity and its facts\n\n"
        "**Examples**:\n"
        "- `/memory`\n"
        "- `/memory 1`\n"
        "- `/memory 3 delete`"
    )

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute memory command."""
        args = args.strip()
        parts = args.split() if args else []

        # No args — list all entities
        if not parts:
            entities = context.db.get_user_entities(context.user)
            if not entities:
                return CommandResult(text=PennyResponse.MEMORY_EMPTY)

            lines = [PennyResponse.MEMORY_LIST_HEADER, ""]
            for i, entity in enumerate(entities, 1):
                count = _count_facts(entity.facts)
                lines.append(f"{i}. {entity.name} ({count} fact{'s' if count != 1 else ''})")
            return CommandResult(text="\n".join(lines))

        # First arg must be a number
        if not parts[0].isdigit():
            return CommandResult(text=PennyResponse.MEMORY_ENTITY_NOT_FOUND.format(number=parts[0]))

        position = int(parts[0])
        entities = context.db.get_user_entities(context.user)

        if position < 1 or position > len(entities):
            return CommandResult(text=PennyResponse.MEMORY_ENTITY_NOT_FOUND.format(number=position))

        entity = entities[position - 1]
        assert entity.id is not None

        # Number + "delete" — delete entity
        if len(parts) >= 2 and parts[1].lower() == "delete":
            count = _count_facts(entity.facts)
            context.db.delete_entity(entity.id)
            return CommandResult(
                text=PennyResponse.MEMORY_DELETED.format(name=entity.name, count=count)
            )

        # Number only — show entity details
        fact_count = _count_facts(entity.facts)
        if fact_count == 0:
            return CommandResult(text=PennyResponse.MEMORY_NO_FACTS.format(name=entity.name))

        updated = entity.updated_at.strftime("%Y-%m-%d %H:%M")
        lines = [
            f"**{entity.name}**",
            f"Updated: {updated}",
            "",
            entity.facts,
        ]
        return CommandResult(text="\n".join(lines))

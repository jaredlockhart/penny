"""Memory command — /memory."""

from __future__ import annotations

import logging

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.responses import PennyResponse

logger = logging.getLogger(__name__)


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
                assert entity.id is not None
                facts = context.db.get_entity_facts(entity.id)
                count = len(facts)
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
            facts = context.db.get_entity_facts(entity.id)
            context.db.delete_entity(entity.id)
            return CommandResult(
                text=PennyResponse.MEMORY_DELETED.format(name=entity.name, count=len(facts))
            )

        # Number only — show entity details
        facts = context.db.get_entity_facts(entity.id)
        if not facts:
            return CommandResult(text=PennyResponse.MEMORY_NO_FACTS.format(name=entity.name))

        updated = entity.updated_at.strftime("%Y-%m-%d %H:%M")
        facts_text = "\n".join(f"- {f.content}" for f in facts)
        lines = [
            f"**{entity.name}**",
            f"Updated: {updated}",
            "",
            facts_text,
        ]
        return CommandResult(text="\n".join(lines))

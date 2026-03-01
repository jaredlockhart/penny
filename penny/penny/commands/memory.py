"""Memory command — /memory."""

from __future__ import annotations

import logging

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.database.models import Entity
from penny.interest import scored_entities_for_user
from penny.responses import PennyResponse

logger = logging.getLogger(__name__)


class MemoryCommand(Command):
    """View Penny's knowledge base."""

    name = "memory"
    description = "View what Penny has learned and remembered"
    help_text = (
        "View what Penny has learned from searches.\n\n"
        "**Usage**:\n"
        "• `/memory` — List all remembered entities ranked by heat\n"
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

        # No args — list all entities ranked by heat
        if not parts:
            scored = self._scored_entities(context)
            if not scored:
                return CommandResult(text=PennyResponse.MEMORY_EMPTY)

            lines = [PennyResponse.MEMORY_LIST_HEADER, ""]
            for i, (score, entity) in enumerate(scored, 1):
                assert entity.id is not None
                facts = context.db.facts.get_for_entity(entity.id)
                facts_label = f"{len(facts)} fact{'s' if len(facts) != 1 else ''}"
                tagline_suffix = f" — {entity.tagline}" if entity.tagline else ""
                name_part = f"{i}. **{entity.name}**{tagline_suffix}"
                lines.append(f"{name_part} ({facts_label}, heat: {score:.2f})")
            return CommandResult(text="\n".join(lines))

        # First arg must be a number
        if not parts[0].isdigit():
            return CommandResult(text=PennyResponse.MEMORY_ENTITY_NOT_FOUND.format(number=parts[0]))

        position = int(parts[0])
        scored = self._scored_entities(context)

        if position < 1 or position > len(scored):
            return CommandResult(text=PennyResponse.MEMORY_ENTITY_NOT_FOUND.format(number=position))

        _score, entity = scored[position - 1]
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
        """Trace facts back to their originating learn topic or search query."""
        for fact in facts:
            if fact.source_search_log_id is None:
                continue
            search_log = context.db.searches.get(fact.source_search_log_id)
            if search_log is None:
                continue
            if search_log.learn_prompt_id is not None:
                learn_prompt = context.db.learn_prompts.get(search_log.learn_prompt_id)
                if learn_prompt is not None:
                    return f"/learn {learn_prompt.prompt_text}"
            return f"search: {search_log.query[:80]}"
        return None

    @staticmethod
    def _scored_entities(context: CommandContext) -> list[tuple[float, Entity]]:
        """Return (heat, entity) pairs sorted by heat descending."""
        entities = context.db.entities.get_for_user(context.user)
        if not entities:
            return []
        return scored_entities_for_user(entities)

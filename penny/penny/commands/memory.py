"""Memory command — /memory."""

from __future__ import annotations

import logging
from collections import defaultdict

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.database.models import Engagement, Entity
from penny.interest import compute_interest_score
from penny.responses import PennyResponse

logger = logging.getLogger(__name__)


class MemoryCommand(Command):
    """View or manage Penny's knowledge base."""

    name = "memory"
    description = "View or manage Penny's knowledge base"
    help_text = (
        "View what Penny has learned from searches, or manage stored knowledge.\n\n"
        "**Usage**:\n"
        "- `/memory` — List all remembered entities ranked by interest\n"
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

        # No args — list all entities ranked by interest score
        if not parts:
            scored = self._scored_entities(context)
            if not scored:
                return CommandResult(text=PennyResponse.MEMORY_EMPTY)

            lines = [PennyResponse.MEMORY_LIST_HEADER, ""]
            for i, (score, entity) in enumerate(scored, 1):
                assert entity.id is not None
                facts = context.db.get_entity_facts(entity.id)
                sign = "+" if score > 0 else ""
                facts_label = f"{len(facts)} fact{'s' if len(facts) != 1 else ''}"
                lines.append(f"{i}. **{entity.name}** ({facts_label}, interest: {sign}{score:.2f})")
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
        facts_text = "\n\n".join(f"• {f.content}" for f in facts)
        lines = [
            f"**{entity.name}**",
            f"**Updated**: {updated}",
            "",
            facts_text,
        ]
        return CommandResult(text="\n".join(lines))

    @staticmethod
    def _scored_entities(context: CommandContext) -> list[tuple[float, Entity]]:
        """Return (score, entity) pairs sorted by absolute interest score descending."""
        entities = context.db.get_user_entities(context.user)
        if not entities:
            return []

        all_engagements = context.db.get_user_engagements(context.user)
        engagements_by_entity: dict[int, list[Engagement]] = defaultdict(list)
        for eng in all_engagements:
            if eng.entity_id is not None:
                engagements_by_entity[eng.entity_id].append(eng)

        scored: list[tuple[float, Entity]] = []
        for entity in entities:
            assert entity.id is not None
            entity_engagements = engagements_by_entity.get(entity.id, [])
            score = compute_interest_score(
                entity_engagements,
                half_life_days=context.config.runtime.INTEREST_SCORE_HALF_LIFE_DAYS,
            )
            scored.append((score, entity))

        scored.sort(key=lambda x: abs(x[0]), reverse=True)
        return scored

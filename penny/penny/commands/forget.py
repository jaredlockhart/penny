"""The /forget command — delete a remembered entity by number."""

from __future__ import annotations

import logging
from collections import defaultdict

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.database.models import Engagement, Entity
from penny.interest import compute_interest_score
from penny.responses import PennyResponse

logger = logging.getLogger(__name__)


class ForgetCommand(Command):
    """Delete a remembered entity by its /memory list number."""

    name = "forget"
    description = "Delete a memory entry by number"
    help_text = (
        "Delete an entity and all its stored facts from Penny's memory.\n\n"
        "**Usage**:\n"
        "• `/forget <number>` — Delete the memory at that position\n\n"
        "Run `/memory` first to see the numbered list.\n\n"
        "**Examples**:\n"
        "• `/forget 1`\n"
        "• `/forget 3`"
    )

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute forget command."""
        args = args.strip()

        if not args:
            return CommandResult(text=PennyResponse.FORGET_USAGE)

        if not args.isdigit():
            return CommandResult(text=PennyResponse.FORGET_NOT_FOUND.format(number=args))

        position = int(args)
        scored = self._scored_entities(context)

        if position < 1 or position > len(scored):
            return CommandResult(text=PennyResponse.FORGET_NOT_FOUND.format(number=position))

        _score, entity = scored[position - 1]
        assert entity.id is not None

        facts = context.db.facts.get_for_entity(entity.id)
        context.db.entities.delete(entity.id)
        return CommandResult(
            text=PennyResponse.FORGET_DELETED.format(name=entity.name, count=len(facts))
        )

    @staticmethod
    def _scored_entities(context: CommandContext) -> list[tuple[float, Entity]]:
        """Return (score, entity) pairs sorted by absolute interest score descending."""
        entities = context.db.entities.get_for_user(context.user)
        if not entities:
            return []

        all_engagements = context.db.engagements.get_for_user(context.user)
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

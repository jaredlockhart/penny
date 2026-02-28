"""The /forget command — delete a remembered entity by number."""

from __future__ import annotations

import logging

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.database.models import Entity
from penny.interest import scored_entities_for_user
from penny.ollama.embeddings import deserialize_embedding
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
        """Return (score, entity) pairs sorted by notification priority descending."""
        entities = context.db.entities.get_for_user(context.user)
        if not entities:
            return []
        all_engagements = context.db.engagements.get_for_user(context.user)
        facts_by_entity = {
            e.id: context.db.facts.get_for_entity(e.id) for e in entities if e.id is not None
        }
        notified_counts = {eid: context.db.facts.count_notified(eid) for eid in facts_by_entity}
        embedding_candidates = [
            (e.id, deserialize_embedding(e.embedding))
            for e in context.db.entities.get_with_embeddings(context.user)
            if e.id is not None and e.embedding is not None
        ]
        rt = context.config.runtime
        return scored_entities_for_user(
            entities,
            all_engagements,
            facts_by_entity,
            notified_counts,
            embedding_candidates,
            half_life_days=rt.INTEREST_SCORE_HALF_LIFE_DAYS,
            neighbor_k=int(rt.NOTIFICATION_NEIGHBOR_K),
            neighbor_min_similarity=rt.NOTIFICATION_NEIGHBOR_MIN_SIMILARITY,
            neighbor_factor=rt.NOTIFICATION_NEIGHBOR_FACTOR,
        )

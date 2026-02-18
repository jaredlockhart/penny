"""Interest graph visibility command — /interests."""

from __future__ import annotations

import logging
from collections import defaultdict

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.constants import PennyConstants
from penny.database.models import Engagement
from penny.interest import compute_interest_score
from penny.responses import PennyResponse

logger = logging.getLogger(__name__)


class InterestsCommand(Command):
    """View what Penny thinks you care about."""

    name = "interests"
    description = "View your interest graph"
    help_text = (
        "See what Penny thinks you're interested in, ranked by score.\n\n"
        "**Usage**:\n"
        "- `/interests` — Show ranked entities by interest score\n\n"
        "Scores are computed from your interactions: likes, dislikes, "
        "searches, reactions, and mentions. Higher scores mean stronger interest."
    )

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute interests command."""
        entities = context.db.get_user_entities(context.user)
        if not entities:
            return CommandResult(text=PennyResponse.INTERESTS_EMPTY)

        # Get all engagements for this user and group by entity_id
        all_engagements = context.db.get_user_engagements(context.user)
        engagements_by_entity: dict[int, list[Engagement]] = defaultdict(list)
        for eng in all_engagements:
            if eng.entity_id is not None:
                engagements_by_entity[eng.entity_id].append(eng)

        # Compute interest scores and collect display data
        scored: list[tuple[str, float, int, str]] = []  # (name, score, fact_count, last_activity)
        for entity in entities:
            assert entity.id is not None
            entity_engagements = engagements_by_entity.get(entity.id, [])
            score = compute_interest_score(entity_engagements)

            if score == 0.0 and not entity_engagements:
                continue

            facts = context.db.get_entity_facts(entity.id)

            # Last activity: most recent engagement or entity update
            if entity_engagements:
                last_activity = entity_engagements[0].created_at.strftime("%Y-%m-%d")
            else:
                last_activity = entity.updated_at.strftime("%Y-%m-%d")

            scored.append((entity.name, score, len(facts), last_activity))

        if not scored:
            return CommandResult(text=PennyResponse.INTERESTS_EMPTY)

        # Sort by absolute score descending (strongest signals first), cap display
        scored.sort(key=lambda x: abs(x[1]), reverse=True)
        display_limit = PennyConstants.INTERESTS_DISPLAY_LIMIT
        total = len(scored)
        scored = scored[:display_limit]

        lines = [PennyResponse.INTERESTS_HEADER, ""]
        for i, (name, score, fact_count, last_activity) in enumerate(scored, 1):
            sign = "+" if score > 0 else ""
            facts_label = f"{fact_count} fact{'s' if fact_count != 1 else ''}"
            lines.append(
                f"{i}. **{name}** ({sign}{score:.2f}) — {facts_label}, last {last_activity}"
            )

        if total > display_limit:
            lines.append(f"\n({total - display_limit} more not shown)")

        return CommandResult(text="\n".join(lines))

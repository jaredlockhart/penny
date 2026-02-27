"""/learn command — research a topic with full provenance tracking."""

from __future__ import annotations

import logging
from collections import defaultdict

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.constants import PennyConstants
from penny.database.models import Entity
from penny.responses import PennyResponse

logger = logging.getLogger(__name__)


class LearnCommand(Command):
    """Research a topic via multi-query search with provenance tracking."""

    name = "learn"
    description = "Start learning about a topic"
    help_text = (
        "Express interest in a topic so Penny researches it in the background.\n\n"
        "**Usage**:\n"
        "• `/learn` — Show learning status and discoveries\n"
        "• `/learn <topic>` — Start learning about a topic\n\n"
        "**Examples**:\n"
        "• `/learn kef ls50 meta`\n"
        "• `/learn travel in china 2026`"
    )

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute learn command."""
        topic = args.strip()

        if not topic:
            return self._show_status(context)

        return await self._learn_topic(topic, context)

    def _show_status(self, context: CommandContext) -> CommandResult:
        """Show LearnPrompt status with provenance chain."""
        learn_prompts = context.db.learn_prompts.get_for_user(context.user)
        # Hide announced learn prompts — their completion summary was already sent
        learn_prompts = [lp for lp in learn_prompts if lp.announced_at is None]
        if not learn_prompts:
            return CommandResult(text=PennyResponse.LEARN_EMPTY)

        # Limit display
        display_prompts = learn_prompts[: int(context.config.runtime.LEARN_STATUS_DISPLAY_LIMIT)]

        lines = [PennyResponse.LEARN_STATUS_HEADER, ""]
        for i, lp in enumerate(display_prompts, 1):
            assert lp.id is not None

            # Fetch search logs first — used for both status and provenance
            search_logs = context.db.searches.get_by_learn_prompt(lp.id)

            if lp.status != PennyConstants.LearnPromptStatus.COMPLETED:
                status_text = f"(searching, {lp.searches_remaining} left)"
            elif search_logs and any(not sl.extracted for sl in search_logs):
                extracted = sum(1 for sl in search_logs if sl.extracted)
                total = len(search_logs)
                status_text = f"(reading, {extracted} of {total} processed)"
            else:
                status_text = "\u2713"
            lines.append(f"{i}. **{lp.prompt_text}** {status_text}")
            if not search_logs:
                lines.append("")
                continue

            search_log_ids = [sl.id for sl in search_logs if sl.id is not None]
            facts = context.db.facts.get_by_search_log_ids(search_log_ids)
            if not facts:
                lines.append("")
                continue

            # Group facts by entity
            entity_fact_counts: dict[int, int] = defaultdict(int)
            for fact in facts:
                entity_fact_counts[fact.entity_id] += 1

            # Resolve entity names
            entity_lines = _build_entity_lines(entity_fact_counts, context)
            for entity_line in entity_lines:
                lines.append(f"   {entity_line}")
            lines.append("")

        return CommandResult(text="\n".join(lines).rstrip())

    async def _learn_topic(self, topic: str, context: CommandContext) -> CommandResult:
        """Create LearnPrompt and acknowledge. Research happens via scheduled worker."""
        context.db.learn_prompts.create(
            user=context.user,
            prompt_text=topic,
            searches_remaining=int(context.config.runtime.LEARN_PROMPT_DEFAULT_SEARCHES),
        )
        return CommandResult(text=PennyResponse.LEARN_ACKNOWLEDGED.format(topic=topic))


def _build_entity_lines(entity_fact_counts: dict[int, int], context: CommandContext) -> list[str]:
    """Build display lines for entities with fact counts."""
    lines: list[str] = []
    # Sort by fact count descending
    sorted_entities = sorted(entity_fact_counts.items(), key=lambda x: x[1], reverse=True)
    for entity_id, fact_count in sorted_entities:
        entity = _get_entity_cached(entity_id, context)
        if entity is None:
            continue
        facts_label = f"{fact_count} fact{'s' if fact_count != 1 else ''}"
        tagline_suffix = f" — {entity.tagline}" if entity.tagline else ""
        lines.append(f"• **{entity.name}**{tagline_suffix} ({facts_label})")
    return lines


def _get_entity_cached(entity_id: int, context: CommandContext) -> Entity | None:
    """Get an entity by ID. Uses the database directly."""
    return context.db.entities.get(entity_id)

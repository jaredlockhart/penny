"""/learn command — research a topic with full provenance tracking."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.constants import PennyConstants
from penny.database.models import Entity
from penny.prompts import Prompt
from penny.responses import PennyResponse
from penny.tools.models import SearchResult

if TYPE_CHECKING:
    from penny.tools import Tool

logger = logging.getLogger(__name__)


class GeneratedQuery(BaseModel):
    """Schema for LLM response: a single search query."""

    query: str = Field(
        default="",
        description="Search query to execute, or empty string if research is complete",
    )


class LearnCommand(Command):
    """Research a topic via multi-query search with provenance tracking."""

    name = "learn"
    description = "Start learning about a topic"
    help_text = (
        "Express interest in a topic so Penny researches it in the background.\n\n"
        "**Usage**:\n"
        "- `/learn` — Show learning status and discoveries\n"
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
            return self._show_status(context)

        return await self._learn_topic(topic, context)

    def _show_status(self, context: CommandContext) -> CommandResult:
        """Show LearnPrompt status with provenance chain."""
        learn_prompts = context.db.get_user_learn_prompts(context.user)
        # Hide announced learn prompts — their completion summary was already sent
        learn_prompts = [lp for lp in learn_prompts if lp.announced_at is None]
        if not learn_prompts:
            return CommandResult(text=PennyResponse.LEARN_EMPTY)

        # Limit display
        display_prompts = learn_prompts[: PennyConstants.LEARN_STATUS_DISPLAY_LIMIT]

        lines = [PennyResponse.LEARN_STATUS_HEADER, ""]
        for i, lp in enumerate(display_prompts, 1):
            assert lp.id is not None

            # Fetch search logs first — used for both status and provenance
            search_logs = context.db.get_search_logs_by_learn_prompt(lp.id)

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
                continue

            search_log_ids = [sl.id for sl in search_logs if sl.id is not None]
            facts = context.db.get_facts_by_search_log_ids(search_log_ids)
            if not facts:
                continue

            # Group facts by entity
            entity_fact_counts: dict[int, int] = defaultdict(int)
            for fact in facts:
                entity_fact_counts[fact.entity_id] += 1

            # Resolve entity names
            entity_lines = _build_entity_lines(entity_fact_counts, context)
            for entity_line in entity_lines:
                lines.append(f"   {entity_line}")

        return CommandResult(text="\n".join(lines))

    async def _learn_topic(self, topic: str, context: CommandContext) -> CommandResult:
        """Create LearnPrompt, acknowledge, and start background research."""
        # Create LearnPrompt record
        learn_prompt = context.db.create_learn_prompt(
            user=context.user,
            prompt_text=topic,
            searches_remaining=PennyConstants.LEARN_PROMPT_DEFAULT_SEARCHES,
        )

        if self._search_tool and learn_prompt and learn_prompt.id is not None:
            asyncio.create_task(self._research_background(topic, learn_prompt.id, context))

        return CommandResult(text=PennyResponse.LEARN_ACKNOWLEDGED.format(topic=topic))

    async def _generate_initial_query(self, topic: str, context: CommandContext) -> str:
        """Generate the first search query for a topic via LLM."""
        prompt = f"{Prompt.LEARN_INITIAL_QUERY_PROMPT}\n\nTopic: {topic}"

        try:
            response = await context.foreground_model_client.generate(
                prompt=prompt,
                tools=None,
                format=GeneratedQuery.model_json_schema(),
            )
            result = GeneratedQuery.model_validate_json(response.content)
            if result.query.strip():
                return result.query.strip()
        except Exception as e:
            logger.error("Failed to generate initial query for '%s': %s", topic, e)

        # Fallback: use the topic as-is
        return topic

    async def _generate_followup_query(
        self, topic: str, previous_results: list[str], context: CommandContext
    ) -> str | None:
        """Generate the next search query based on previous results.

        Returns the query string, or None if research is complete.
        """
        results_text = "\n\n---\n\n".join(
            f"Search {i + 1}:\n{text[:1000]}" for i, text in enumerate(previous_results)
        )
        prompt = Prompt.LEARN_FOLLOWUP_QUERY_PROMPT.format(
            topic=topic, previous_results=results_text
        )

        try:
            response = await context.foreground_model_client.generate(
                prompt=prompt,
                tools=None,
                format=GeneratedQuery.model_json_schema(),
            )
            result = GeneratedQuery.model_validate_json(response.content)
            query = result.query.strip()
            return query if query else None
        except Exception as e:
            logger.error("Failed to generate followup query for '%s': %s", topic, e)
            return None

    async def _search(self, query: str, learn_prompt_id: int) -> str | None:
        """Execute search via SearchTool with provenance. Returns text or None."""
        assert self._search_tool is not None
        try:
            result = await self._search_tool.execute(
                query=query,
                skip_images=True,
                trigger=PennyConstants.SearchTrigger.LEARN_COMMAND,
                learn_prompt_id=learn_prompt_id,
            )
            if isinstance(result, SearchResult):
                return result.text
            return str(result) if result else None
        except Exception as e:
            logger.error("Learn command search failed: %s", e)
            return None

    async def _research_background(
        self, topic: str, learn_prompt_id: int, context: CommandContext
    ) -> None:
        """Iteratively research a topic: each query builds on previous results."""
        try:
            max_searches = PennyConstants.LEARN_PROMPT_DEFAULT_SEARCHES
            previous_results: list[str] = []
            search_count = 0

            # First query: broad overview
            query = await self._generate_initial_query(topic, context)
            result = await self._search(query, learn_prompt_id)
            search_count += 1
            context.db.decrement_learn_prompt_searches(learn_prompt_id)

            if result:
                previous_results.append(result)

            # Followup queries: each informed by previous results
            while search_count < max_searches and previous_results:
                query = await self._generate_followup_query(topic, previous_results, context)
                if query is None:
                    # LLM decided research is complete
                    break

                result = await self._search(query, learn_prompt_id)
                search_count += 1
                context.db.decrement_learn_prompt_searches(learn_prompt_id)

                if result:
                    previous_results.append(result)

            # Mark as completed
            context.db.update_learn_prompt_status(
                learn_prompt_id, PennyConstants.LearnPromptStatus.COMPLETED
            )
            logger.info("Learn command completed for '%s': %d searches", topic, search_count)
        except Exception:
            logger.exception("Background research failed for '%s'", topic)


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
        lines.append(f"• **{entity.name}** ({facts_label})")
    return lines


def _get_entity_cached(entity_id: int, context: CommandContext) -> Entity | None:
    """Get an entity by ID. Uses the database directly."""
    return context.db.get_entity(entity_id)

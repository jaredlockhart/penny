"""/follow command — monitor a topic for ongoing events via news."""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.constants import PennyConstants
from penny.prompts import Prompt
from penny.responses import PennyResponse

logger = logging.getLogger(__name__)

_CADENCE_VALUES = frozenset(c.value for c in PennyConstants.FollowCadence)


class QueryTermsResult(BaseModel):
    """LLM-generated search query terms for a topic."""

    query_terms: list[str] = Field(description="Search phrases for news monitoring")


class FollowCommand(Command):
    """Monitor a topic for ongoing events via the news API."""

    name = "follow"
    description = "Follow a topic for ongoing event monitoring"
    help_text = (
        "Start monitoring a topic for news and events.\n\n"
        "**Usage**:\n"
        "• `/follow` — List your active subscriptions\n"
        "• `/follow <topic>` — Start following a topic (daily updates by default)\n"
        "• `/follow hourly|daily|weekly <topic>` — Follow with a specific cadence\n\n"
        "**Examples**:\n"
        "• `/follow artificial intelligence`\n"
        "• `/follow hourly spacex launches`\n"
        "• `/follow weekly climate policy`"
    )

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Route to list or create based on args."""
        if not context.config.news_api_key:
            return CommandResult(text=PennyResponse.NEWS_NOT_CONFIGURED)

        topic = args.strip()

        if not topic:
            return self._list_follows(context)

        cadence, topic = self._parse_cadence(topic)
        return await self._follow_topic(topic, cadence, context)

    def _parse_cadence(self, text: str) -> tuple[str, str]:
        """Extract optional cadence prefix from topic text.

        Returns (cadence, remaining_topic). Defaults to FOLLOW_DEFAULT_CADENCE
        if no cadence keyword is found.
        """
        parts = text.split(None, 1)
        if parts and parts[0].lower() in _CADENCE_VALUES:
            cadence = parts[0].lower()
            topic = parts[1] if len(parts) > 1 else ""
            return cadence, topic.strip()
        return PennyConstants.FOLLOW_DEFAULT_CADENCE, text

    def _list_follows(self, context: CommandContext) -> CommandResult:
        """List active follow subscriptions."""
        follows = context.db.follow_prompts.get_active(context.user)
        if not follows:
            return CommandResult(text=PennyResponse.FOLLOW_EMPTY)

        lines = [PennyResponse.FOLLOW_LIST_HEADER, ""]
        for i, fp in enumerate(follows, 1):
            date = fp.created_at.strftime("%Y-%m-%d")
            lines.append(f"{i}. **{fp.prompt_text}** ({fp.cadence}) — since {date}")

        return CommandResult(text="\n".join(lines))

    async def _follow_topic(
        self, topic: str, cadence: str, context: CommandContext
    ) -> CommandResult:
        """Generate query terms via LLM and create a FollowPrompt."""
        query_terms = await self._generate_query_terms(topic, context)
        if query_terms is None:
            return CommandResult(text=PennyResponse.FOLLOW_QUERY_TERMS_ERROR)

        context.db.follow_prompts.create(
            user=context.user,
            prompt_text=topic,
            query_terms=json.dumps(query_terms),
            cadence=cadence,
        )
        return CommandResult(
            text=PennyResponse.FOLLOW_ACKNOWLEDGED.format(topic=topic, cadence=cadence)
        )

    async def _generate_query_terms(self, topic: str, context: CommandContext) -> list[str] | None:
        """Use the foreground model to generate search query terms for a topic."""
        prompt = Prompt.FOLLOW_QUERY_TERMS_PROMPT.format(topic=topic)
        try:
            response = await context.foreground_model_client.generate(
                prompt=prompt,
                format="json",
            )
            result = QueryTermsResult.model_validate_json(response.message.content)
            return result.query_terms
        except Exception as e:
            logger.warning("Failed to generate query terms for '%s': %s", topic, e)
            return None

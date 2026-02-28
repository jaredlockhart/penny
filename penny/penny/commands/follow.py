"""/follow command — monitor a topic for ongoing events via news."""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field
from sqlmodel import Session, select

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.database.models import UserInfo
from penny.prompts import Prompt
from penny.responses import PennyResponse

logger = logging.getLogger(__name__)


class FollowParseResult(BaseModel):
    """Parsed follow command — timing + topic extracted by LLM."""

    timing_description: str = Field(description="Natural language timing description")
    topic_text: str = Field(description="Topic to monitor")
    cron_expression: str = Field(description="Cron expression (5 fields)")


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
        "• `/follow <timing> <topic>` — Follow with specific timing\n\n"
        "**Examples**:\n"
        "• `/follow artificial intelligence`\n"
        "• `/follow daily 9:30am usa news`\n"
        "• `/follow hourly spacex launches`\n"
        "• `/follow every monday morning tech`"
    )

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Route to list or create based on args."""
        if not context.config.news_api_key:
            return CommandResult(text=PennyResponse.NEWS_NOT_CONFIGURED)

        topic = args.strip()

        if not topic:
            return self._list_follows(context)

        return await self._follow_topic(topic, context)

    def _list_follows(self, context: CommandContext) -> CommandResult:
        """List active follow subscriptions."""
        follows = context.db.follow_prompts.get_active(context.user)
        if not follows:
            return CommandResult(text=PennyResponse.FOLLOW_EMPTY)

        lines = [PennyResponse.FOLLOW_LIST_HEADER, ""]
        for i, fp in enumerate(follows, 1):
            date = fp.created_at.strftime("%Y-%m-%d")
            lines.append(f"{i}. **{fp.prompt_text}** ({fp.timing_description}) — since {date}")

        return CommandResult(text="\n".join(lines))

    async def _follow_topic(self, command: str, context: CommandContext) -> CommandResult:
        """Parse timing + topic via LLM, generate query terms, create FollowPrompt."""
        user_timezone = self._get_user_timezone(context)
        if user_timezone is None:
            return CommandResult(text=PennyResponse.FOLLOW_NEED_TIMEZONE)

        parsed = await self._parse_follow_input(command, user_timezone, context)
        if parsed is None:
            return CommandResult(text=PennyResponse.FOLLOW_PARSE_ERROR)

        cron_parts = parsed.cron_expression.split()
        if len(cron_parts) != 5:
            logger.warning("Invalid cron expression from LLM: %s", parsed.cron_expression)
            return CommandResult(text=PennyResponse.FOLLOW_PARSE_ERROR)

        query_terms = await self._generate_query_terms(parsed.topic_text, context)
        if query_terms is None:
            return CommandResult(text=PennyResponse.FOLLOW_QUERY_TERMS_ERROR)

        context.db.follow_prompts.create(
            user=context.user,
            prompt_text=parsed.topic_text,
            query_terms=json.dumps(query_terms),
            cron_expression=parsed.cron_expression,
            timing_description=parsed.timing_description,
            user_timezone=user_timezone,
        )
        return CommandResult(
            text=PennyResponse.FOLLOW_ACKNOWLEDGED.format(
                topic=parsed.topic_text, timing=parsed.timing_description
            )
        )

    def _get_user_timezone(self, context: CommandContext) -> str | None:
        """Look up the user's timezone from their profile."""
        with Session(context.db.engine) as session:
            user_info = session.exec(
                select(UserInfo).where(UserInfo.sender == context.user)
            ).first()
            if user_info and user_info.timezone:
                return user_info.timezone
        return None

    async def _parse_follow_input(
        self, command: str, timezone: str, context: CommandContext
    ) -> FollowParseResult | None:
        """Use the foreground model to parse timing + topic from user input."""
        prompt = Prompt.FOLLOW_PARSE_PROMPT.format(timezone=timezone, command=command)
        try:
            response = await context.foreground_model_client.generate(
                prompt=prompt,
                format="json",
            )
            return FollowParseResult.model_validate_json(response.message.content)
        except Exception as e:
            logger.warning("Failed to parse follow command: %s", e)
            return None

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

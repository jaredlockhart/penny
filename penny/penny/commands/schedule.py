"""The /schedule command — create, list, and delete recurring background tasks."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from pydantic import BaseModel, Field
from sqlmodel import Session, select

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.constants import (
    SCHEDULE_ADDED,
    SCHEDULE_DELETED_NO_REMAINING,
    SCHEDULE_DELETED_PREFIX,
    SCHEDULE_INVALID_CRON,
    SCHEDULE_INVALID_NUMBER,
    SCHEDULE_NEED_TIMEZONE,
    SCHEDULE_NO_SCHEDULE_WITH_NUMBER,
    SCHEDULE_NO_TASKS,
    SCHEDULE_PARSE_ERROR,
    SCHEDULE_STILL_SCHEDULED,
)
from penny.database.models import Schedule, UserInfo

logger = logging.getLogger(__name__)

# Prompt for parsing schedule commands
SCHEDULE_PARSE_PROMPT = """Parse this schedule command into structured components.

Extract:
1. The timing description (e.g., "daily 9am", "every monday", "hourly")
2. The prompt text (the task to execute when the schedule fires)
3. A cron expression representing the timing (use standard cron format)
   Format: minute hour day month weekday

User timezone: {timezone}

Command: {command}

Return JSON with:
- timing_description: the natural language timing description you extracted
- prompt_text: the prompt to execute
- cron_expression: cron expression (5 fields: minute hour day month weekday, use * for "any")

Examples:
- "daily 9am check the news"
  → timing="daily 9am", prompt="check the news", cron="0 9 * * *"
- "every monday morning meal ideas"
  → timing="every monday morning", prompt="meal ideas", cron="0 9 * * 1"
- "hourly sports scores"
  → timing="hourly", prompt="sports scores", cron="0 * * * *"
"""


class ScheduleParseResult(BaseModel):
    """Parsed schedule command."""

    timing_description: str = Field(description="Natural language timing description")
    prompt_text: str = Field(description="Prompt to execute")
    cron_expression: str = Field(description="Cron expression (5 fields)")


class ScheduleCommand(Command):
    """Create, list, and delete recurring background tasks."""

    name = "schedule"
    description = "Create, list, and delete recurring background tasks"
    help_text = (
        "Create recurring background tasks that run prompts automatically.\n\n"
        "**Usage**:\n"
        "- `/schedule` — List all your active schedules\n"
        "- `/schedule <timing> <prompt>` — Create a new schedule\n"
        "  (e.g., `/schedule daily 9am what's the news?`)\n"
        "- `/schedule delete <number>` — Delete a schedule by its list number"
    )

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute schedule command."""
        args = args.strip()

        # Case 1: List schedules
        if not args:
            return await self._list_schedules(context)

        # Case 2: Delete schedule
        if args.startswith("delete "):
            index_str = args[7:].strip()
            return await self._delete_schedule(index_str, context)

        # Case 3: Create schedule
        return await self._create_schedule(args, context)

    async def _list_schedules(self, context: CommandContext) -> CommandResult:
        """List all schedules for the user."""
        with Session(context.db.engine) as session:
            schedules = list(
                session.exec(
                    select(Schedule).where(Schedule.user_id == context.user).order_by(Schedule.id)  # type: ignore[arg-type]
                )
            )

            if not schedules:
                return CommandResult(text=SCHEDULE_NO_TASKS)

            lines = []
            for idx, sched in enumerate(schedules, start=1):
                lines.append(f"{idx}. {sched.timing_description} '{sched.prompt_text}'")

            return CommandResult(text="\n".join(lines))

    async def _delete_schedule(self, index_str: str, context: CommandContext) -> CommandResult:
        """Delete a schedule by index."""
        try:
            index = int(index_str)
        except ValueError:
            return CommandResult(text=SCHEDULE_INVALID_NUMBER.format(number=index_str))

        with Session(context.db.engine) as session:
            schedules = list(
                session.exec(
                    select(Schedule).where(Schedule.user_id == context.user).order_by(Schedule.id)  # type: ignore[arg-type]
                )
            )

            if index < 1 or index > len(schedules):
                return CommandResult(text=SCHEDULE_NO_SCHEDULE_WITH_NUMBER.format(number=index))

            to_delete = schedules[index - 1]
            session.delete(to_delete)
            session.commit()

            # Show remaining schedules
            remaining = [s for s in schedules if s.id != to_delete.id]
            if not remaining:
                deleted_msg = SCHEDULE_DELETED_PREFIX.format(
                    timing=to_delete.timing_description, prompt=to_delete.prompt_text
                )
                return CommandResult(text=f"{deleted_msg}\n\n{SCHEDULE_DELETED_NO_REMAINING}")

            deleted_msg = SCHEDULE_DELETED_PREFIX.format(
                timing=to_delete.timing_description, prompt=to_delete.prompt_text
            )
            lines = [
                f"{deleted_msg}\n",
                SCHEDULE_STILL_SCHEDULED,
            ]
            for idx, sched in enumerate(remaining, start=1):
                lines.append(f"{idx}. {sched.timing_description} '{sched.prompt_text}'")

            return CommandResult(text="\n".join(lines))

    async def _create_schedule(self, command: str, context: CommandContext) -> CommandResult:
        """Create a new schedule."""
        # Get user timezone
        with Session(context.db.engine) as session:
            user_info = session.exec(
                select(UserInfo).where(UserInfo.sender == context.user)
            ).first()

            if not user_info or not user_info.timezone:
                return CommandResult(text=SCHEDULE_NEED_TIMEZONE)

            user_timezone = user_info.timezone

        # Parse command using LLM
        prompt = SCHEDULE_PARSE_PROMPT.format(timezone=user_timezone, command=command)

        try:
            response = await context.ollama_client.generate(
                prompt=prompt,
                format="json",
            )

            # Parse JSON from response
            result = ScheduleParseResult.model_validate_json(response.message.content)

        except Exception as e:
            logger.warning("Failed to parse schedule command: %s", e)
            return CommandResult(text=SCHEDULE_PARSE_ERROR)

        # Validate cron expression format (5 fields)
        cron_parts = result.cron_expression.split()
        if len(cron_parts) != 5:
            logger.warning("Invalid cron expression: %s", result.cron_expression)
            return CommandResult(text=SCHEDULE_INVALID_CRON)

        # Create schedule in database
        with Session(context.db.engine) as session:
            new_schedule = Schedule(
                user_id=context.user,
                user_timezone=user_timezone,
                cron_expression=result.cron_expression,
                prompt_text=result.prompt_text,
                timing_description=result.timing_description,
                created_at=datetime.now(UTC),
            )
            session.add(new_schedule)
            session.commit()

        return CommandResult(
            text=SCHEDULE_ADDED.format(timing=result.timing_description, prompt=result.prompt_text)
        )

"""ScheduleExecutor — agent that executes user-created scheduled tasks."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from croniter import croniter
from sqlmodel import Session, select

from penny.agents.base import Agent
from penny.database.models import Schedule

if TYPE_CHECKING:
    from penny.channels import MessageChannel

logger = logging.getLogger(__name__)


class ScheduleExecutor(Agent):
    """Agent that executes user-created scheduled tasks."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._channel: MessageChannel | None = None

    @property
    def name(self) -> str:
        """Task name for logging."""
        return "schedule"

    def set_channel(self, channel: MessageChannel) -> None:
        """Set the channel for sending responses."""
        self._channel = channel

    async def execute(self) -> bool:
        """
        Find and execute any due schedules.

        Returns:
            True if any schedules were executed, False otherwise
        """
        if not self._channel:
            logger.error("ScheduleExecutor: no channel set")
            return False

        with Session(self.db.engine) as session:
            schedules = session.exec(select(Schedule)).all()

            now_utc = datetime.now(UTC)
            executed_any = False

            for sched in schedules:
                try:
                    # Convert user timezone to ZoneInfo
                    tz = ZoneInfo(sched.user_timezone)
                    now_in_user_tz = now_utc.astimezone(tz)

                    # Parse cron expression
                    cron = croniter(sched.cron_expression, now_in_user_tz)

                    # Get previous scheduled time
                    prev = cron.get_prev(datetime)

                    # Check if prev is within the last 60 seconds (scheduler tick interval)
                    # This ensures we fire schedules that became due since last check
                    time_since_due = (now_in_user_tz - prev).total_seconds()

                    if 0 <= time_since_due <= 60:
                        logger.info(
                            "Executing schedule: user=%s, timing=%s, prompt=%s",
                            sched.user_id,
                            sched.timing_description,
                            sched.prompt_text,
                        )

                        # Execute the scheduled prompt as if the user sent it
                        await self._execute_scheduled_prompt(sched)
                        executed_any = True

                except Exception as e:
                    logger.exception(
                        "Failed to execute schedule id=%s: %s",
                        sched.id,
                        e,
                    )
                    continue

            return executed_any

    async def _execute_scheduled_prompt(self, schedule: Schedule) -> None:
        """Execute a scheduled prompt as if the user sent it."""
        if not self._channel:
            return

        # Get the message agent from the channel
        message_agent = self._channel._message_agent

        # Run the prompt through the message agent
        response = await message_agent.run(prompt=schedule.prompt_text)

        answer = response.answer.strip() if response.answer else None
        if not answer:
            logger.warning("Schedule produced empty response: id=%s", schedule.id)
            return

        # Send the response to the user
        typing_task = asyncio.create_task(self._channel._typing_loop(schedule.user_id))
        try:
            await self._channel.send_response(
                schedule.user_id,
                answer,
                parent_id=None,  # Scheduled prompts are not threaded
                attachments=response.attachments or None,
                quote_message=None,
            )
        finally:
            typing_task.cancel()
            await self._channel.send_typing(schedule.user_id, False)

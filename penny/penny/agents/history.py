"""HistoryAgent — daily conversation topic summarization.

Runs on a schedule. Each cycle:
1. For each user, finds completed days without history entries
2. Gets messages for each day
3. Calls model to summarize topics
4. Stores the summary
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from penny.agents.base import Agent
from penny.constants import PennyConstants
from penny.prompts import Prompt

logger = logging.getLogger(__name__)


class HistoryAgent(Agent):
    """Background worker that compacts daily conversations into topic summaries."""

    @property
    def name(self) -> str:
        return "history"

    async def execute(self) -> bool:
        """Run one history cycle — summarize un-rolled-up days for each user."""
        users = self.db.users.get_all_senders()
        if not users:
            return False

        did_work = False
        for user in users:
            if await self._process_user(user):
                did_work = True
        return did_work

    async def _process_user(self, user: str) -> bool:
        """Find and summarize un-rolled-up days for one user."""
        max_days = int(self.config.runtime.HISTORY_MAX_DAYS_PER_RUN)
        days = self._find_unsummarized_days(user, max_days)
        if not days:
            return False

        for day_start, day_end in days:
            await self._summarize_day(user, day_start, day_end)
        return True

    def _find_unsummarized_days(self, user: str, max_days: int) -> list[tuple[datetime, datetime]]:
        """Find completed calendar days (UTC) without history entries."""
        duration = PennyConstants.HistoryDuration.DAILY
        latest = self.db.history.get_latest(user, duration)
        start = self._resolve_start_date(user, latest)
        if start is None:
            return []

        yesterday_end = self._midnight_today()
        days: list[tuple[datetime, datetime]] = []
        cursor = start
        while cursor < yesterday_end and len(days) < max_days:
            day_end = cursor + timedelta(days=1)
            if not self.db.history.exists(user, cursor, duration):
                days.append((cursor, day_end))
            cursor = day_end

        return days

    def _resolve_start_date(self, user: str, latest: object | None) -> datetime | None:
        """Determine where to start scanning for un-rolled-up days."""
        if latest is not None:
            return getattr(latest, "period_end", None)
        first_msg_time = self.db.messages.get_first_message_time(user)
        if first_msg_time is None:
            return None
        return first_msg_time.replace(hour=0, minute=0, second=0, microsecond=0)

    @staticmethod
    def _midnight_today() -> datetime:
        """Return midnight UTC for today as a naive datetime.

        Naive because SQLite strips timezone info — all stored datetimes are naive UTC.
        """
        return datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)

    async def _summarize_day(self, user: str, day_start: datetime, day_end: datetime) -> None:
        """Get messages for a day and call model to summarize topics."""
        messages = self.db.messages.get_messages_in_range(user, day_start, day_end)
        if not messages:
            logger.debug("No messages for %s on %s, skipping", user, day_start.date())
            return

        message_text = self._format_messages(messages)
        topics = await self._call_model(message_text)
        if not topics:
            logger.debug("Model returned empty topics for %s on %s", user, day_start.date())
            return

        self.db.history.add(
            user=user,
            period_start=day_start,
            period_end=day_end,
            duration=PennyConstants.HistoryDuration.DAILY,
            topics=topics,
        )
        logger.info("History entry created for %s on %s", user, day_start.date())

    @staticmethod
    def _format_messages(messages: list) -> str:
        """Format messages for the summarization prompt."""
        lines: list[str] = []
        for msg in messages:
            ts = msg.timestamp.strftime("%H:%M")
            if msg.direction == PennyConstants.MessageDirection.INCOMING:
                lines.append(f"[{ts}] User: {msg.content}")
            else:
                lines.append(f"[{ts}] Penny: {msg.content}")
        return "\n".join(lines)

    async def _call_model(self, message_text: str) -> str:
        """Call the background model to summarize topics."""
        prompt = Prompt.DAILY_HISTORY_PROMPT.format(messages=message_text)
        messages = [{"role": "user", "content": prompt}]
        try:
            response = await self._background_model_client.chat(messages=messages)
            return response.content.strip()
        except Exception as e:
            logger.error("History summarization failed: %s", e)
            return ""

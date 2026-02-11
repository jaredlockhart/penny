"""SummarizeAgent for summarizing conversation threads."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from penny.agents.base import Agent
from penny.agents.models import MessageRole
from penny.constants import MessageDirection

if TYPE_CHECKING:
    from penny.database.models import MessageLog

logger = logging.getLogger(__name__)


class SummarizeAgent(Agent):
    """Agent for summarizing conversation threads."""

    @property
    def name(self) -> str:
        """Task name for logging."""
        return "summarize"

    async def execute(self) -> bool:
        """
        Find and summarize one unsummarized thread.

        Returns:
            True if a thread was summarized, False if nothing to do
        """
        unsummarized = self.db.get_unsummarized_messages()
        if not unsummarized:
            return False

        msg = unsummarized[0]
        assert msg.id is not None

        thread = self.db._walk_thread(msg.id)
        if len(thread) < 2:
            # Mark as processed but no real summary needed
            self.db.set_parent_summary(msg.id, "")
            return True

        thread_text = self._format_thread(thread)
        response = await self.run(prompt=thread_text)
        summary = response.answer.strip() if response.answer else None

        if summary is not None:
            self.db.set_parent_summary(msg.id, summary)
            logger.info("Summarized thread for message %d (length: %d)", msg.id, len(summary))
            return True

        return False

    def _format_thread(self, thread: list[MessageLog]) -> str:
        """Format thread as 'role: content' lines for summarization."""
        return "\n".join(
            "{}: {}".format(
                MessageRole.USER.value
                if m.direction == MessageDirection.INCOMING
                else MessageRole.ASSISTANT.value,
                m.content,
            )
            for m in thread
        )

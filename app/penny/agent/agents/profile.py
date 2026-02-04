"""ProfileAgent for building user profiles from message history."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from penny.agent.base import Agent
from penny.constants import PROFILE_PROMPT

if TYPE_CHECKING:
    from penny.database.models import MessageLog

logger = logging.getLogger(__name__)


class ProfileAgent(Agent):
    """Agent for generating user profiles from message history."""

    @property
    def name(self) -> str:
        """Task name for logging."""
        return "profile"

    async def execute(self) -> bool:
        """
        Find one user needing a profile update and generate their profile.

        Returns:
            True if a profile was generated/updated, False if nothing to do
        """
        users = self.db.get_users_needing_profile_update()
        if not users:
            return False

        # Process one user per execution (like SummarizeAgent)
        sender = users[0]
        messages = self.db.get_user_messages(sender)

        if not messages:
            return False

        logger.info("Generating profile for user %s (%d messages)", sender, len(messages))

        # Format messages for the prompt
        messages_text = self._format_messages(messages)
        prompt = f"{PROFILE_PROMPT}{messages_text}"

        response = await self.run(prompt=prompt)
        profile_text = response.answer.strip() if response.answer else None

        if profile_text:
            # Use the timestamp of the newest message
            last_timestamp = messages[-1].timestamp
            self.db.save_user_profile(sender, profile_text, last_timestamp)
            logger.info("Generated profile for %s (length: %d)", sender, len(profile_text))
            return True

        return False

    def _format_messages(self, messages: list[MessageLog]) -> str:
        """Format user messages for profile generation."""
        return "\n".join(f"[{m.timestamp.strftime('%Y-%m-%d')}] {m.content}" for m in messages)

"""DiscoveryAgent for finding new things users would enjoy."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import TYPE_CHECKING

from penny.agent.base import Agent
from penny.agent.models import MessageRole
from penny.constants import DISCOVERY_PROMPT

if TYPE_CHECKING:
    from penny.channels import MessageChannel

logger = logging.getLogger(__name__)


class DiscoveryAgent(Agent):
    """Agent for discovering and sharing new things based on user interests."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._channel: MessageChannel | None = None

    @property
    def name(self) -> str:
        """Task name for logging."""
        return "discovery"

    def set_channel(self, channel: MessageChannel) -> None:
        """Set the channel for sending responses."""
        self._channel = channel

    async def execute(self) -> bool:
        """
        Find something new for a random user based on their profile.

        Returns:
            True if a discovery was sent, False if nothing to do
        """
        if not self._channel:
            logger.error("DiscoveryAgent: no channel set")
            return False

        # Get users who have topics
        users = self.db.get_users_with_topics()
        if not users:
            logger.debug("DiscoveryAgent: no users with topics")
            return False

        # Pick a random user
        recipient = random.choice(users)
        topics = self.db.get_user_topics(recipient)
        if not topics:
            return False

        logger.info("Discovering something new for user %s", recipient)

        # Use topics as context for the discovery
        history = [
            (
                MessageRole.SYSTEM.value,
                f"User topics:\n{topics.profile_text}",
            )
        ]
        response = await self.run(prompt=DISCOVERY_PROMPT, history=history)

        answer = response.answer.strip() if response.answer else None
        if not answer:
            return False

        typing_task = asyncio.create_task(self._channel._typing_loop(recipient))
        try:
            await self._channel.send_response(
                recipient,
                answer,
                parent_id=None,  # Not continuing a thread
                attachments=response.attachments or None,
            )
            return True
        finally:
            typing_task.cancel()
            await self._channel.send_typing(recipient, False)

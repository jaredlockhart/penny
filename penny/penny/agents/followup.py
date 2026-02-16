"""FollowupAgent for spontaneously following up on conversations."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import TYPE_CHECKING

from penny.agents.base import Agent
from penny.agents.models import MessageRole
from penny.constants import MessageDirection, PreferenceType
from penny.prompts import FOLLOWUP_PROMPT

if TYPE_CHECKING:
    from penny.channels import MessageChannel
    from penny.database.models import MessageLog

logger = logging.getLogger(__name__)


class FollowupAgent(Agent):
    """Agent for spontaneously following up on conversations."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._channel: MessageChannel | None = None

    @property
    def name(self) -> str:
        """Task name for logging."""
        return "followup"

    def set_channel(self, channel: MessageChannel) -> None:
        """Set the channel for sending responses."""
        self._channel = channel

    async def execute(self) -> bool:
        """
        Pick a random dangling conversation, follow up on it, and send the response.

        Returns:
            True if a conversation was followed up, False if nothing to do
        """
        if not self._channel:
            logger.error("FollowupAgent: no channel set")
            return False

        leaves = self.db.get_conversation_leaves()
        if not leaves:
            return False

        leaf = random.choice(leaves)
        assert leaf.id is not None

        thread = self.db._walk_thread(leaf.id)
        recipient = self._find_recipient(thread)
        if not recipient:
            return False

        logger.info(
            "Following up on conversation (leaf=%d, recipient=%s)",
            leaf.id,
            recipient,
        )

        history = self._format_history(thread)

        # Add dislike exclusions if the user has any
        dislikes = self.db.get_preferences(recipient, PreferenceType.DISLIKE)
        if dislikes:
            dislike_topics = [d.topic for d in dislikes]
            dislike_list = ", ".join(dislike_topics)
            history.insert(
                0,
                (
                    MessageRole.SYSTEM.value,
                    f"Don't include any of the following in your search: {dislike_list}",
                ),
            )

        response = await self.run(prompt=FOLLOWUP_PROMPT, history=history, sender=recipient)

        answer = response.answer.strip() if response.answer else None
        if not answer:
            return False

        typing_task = asyncio.create_task(self._channel._typing_loop(recipient))
        try:
            await self._channel.send_response(
                recipient,
                answer,
                parent_id=leaf.id,
                attachments=response.attachments or None,
                quote_message=leaf,
            )
            return True
        finally:
            typing_task.cancel()
            await self._channel.send_typing(recipient, False)

    def _find_recipient(self, thread: list[MessageLog]) -> str | None:
        """Find the user's sender ID from the thread."""
        for msg in thread:
            if msg.direction == MessageDirection.INCOMING:
                return msg.sender
        return None

    def _format_history(self, thread: list[MessageLog]) -> list[tuple[str, str]]:
        """Format thread as (role, content) tuples for history."""
        return [
            (
                MessageRole.USER.value
                if m.direction == MessageDirection.INCOMING
                else MessageRole.ASSISTANT.value,
                m.content,
            )
            for m in thread
        ]

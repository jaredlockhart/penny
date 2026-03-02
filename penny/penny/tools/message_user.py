"""Message user tool — send a message to the user."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from penny.tools.base import Tool

if TYPE_CHECKING:
    from penny.channels.base import MessageChannel

logger = logging.getLogger(__name__)


class MessageUserTool(Tool):
    """Send a message to the user with an accompanying image."""

    name = "message_user"
    description = (
        "Send a message to the user. Only do this when you have something "
        "genuinely interesting or useful to share. "
        "Include an image_prompt to attach a relevant image."
    )
    parameters = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The message to send",
            },
            "image_prompt": {
                "type": "string",
                "description": "Search query to find a relevant image to attach",
            },
        },
        "required": ["message", "image_prompt"],
    }

    def __init__(self, channel: MessageChannel, user: str):
        self._channel = channel
        self._user = user

    async def execute(self, **kwargs: Any) -> str:
        """Send a message to the user via the channel with an image."""
        message: str = kwargs["message"]
        image_prompt: str = kwargs["image_prompt"]
        logger.info("[inner_monologue] message_user: %s", message[:200])
        typing_task = asyncio.create_task(self._channel._typing_loop(self._user))
        try:
            await self._channel.send_response(
                self._user,
                message,
                parent_id=None,
                image_prompt=image_prompt,
            )
        finally:
            typing_task.cancel()
            await self._channel.send_typing(self._user, False)
        logger.info("Inner monologue sent message to %s: %s", self._user, message[:80])
        return "Message sent."

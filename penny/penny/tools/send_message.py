"""SendMessageTool — model-driven outbound message delivery.

Bound at construction to a specific (channel, recipient, agent_name)
triple.  The model calls this tool once with the message body when
it has decided what to say; the tool dispatches through the channel
and stamps the agent's name onto the side-effect write to
``penny-messages``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from penny.tools.base import Tool
from penny.tools.models import SendMessageArgs

if TYPE_CHECKING:
    from penny.channels.base import MessageChannel

logger = logging.getLogger(__name__)


class SendMessageTool(Tool):
    """Send a message to the user through the bound channel."""

    name = "send_message"
    description = (
        "Send a message to the user.  Use this once you have decided "
        "what to say — the ``content`` is the exact text the user will "
        "see.  Call ``done`` after to exit."
    )
    parameters = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The message text to send to the user.",
            }
        },
        "required": ["content"],
    }

    def __init__(self, channel: MessageChannel, recipient: str, agent_name: str) -> None:
        self._channel = channel
        self._recipient = recipient
        self._agent_name = agent_name

    async def execute(self, **kwargs: Any) -> str:
        args = SendMessageArgs(**kwargs)
        await self._channel.send_response(
            recipient=self._recipient,
            content=args.content,
            parent_id=None,
            author=self._agent_name,
            quote_message=None,
        )
        logger.info("send_message: %s → %s", self._agent_name, self._recipient)
        return "Message sent."

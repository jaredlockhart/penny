"""Discord implementation of MessageChannel (template for contribution)."""

import logging

from penny.channels.base import IncomingMessage, MessageChannel

logger = logging.getLogger(__name__)


class DiscordChannel(MessageChannel):
    """Discord channel implementation."""

    def __init__(self, token: str, **kwargs):
        """
        Initialize Discord channel.

        Args:
            token: Discord bot token
            **kwargs: Additional Discord-specific configuration
        """
        self.token = token
        # TODO: Initialize Discord client
        logger.info("Initialized Discord channel")

    async def send_message(
        self, recipient: str, message: str, attachments: list[str] | None = None
    ) -> bool:
        """Send a message via Discord."""
        # TODO: Implement Discord message sending
        # recipient could be a channel ID or user ID
        logger.info("Sending message to %s: %s", recipient, message)
        return True

    async def send_typing(self, recipient: str, typing: bool) -> bool:
        """Send a typing indicator via Discord."""
        # TODO: Implement Discord typing indicator
        # Discord has typing.start() API
        logger.debug("Typing indicator for %s: %s", recipient, typing)
        return True

    def get_connection_url(self) -> str:
        """Get the connection identifier for Discord."""
        # Discord uses WebSocket gateway, but connection is handled by the library
        # This might just return a descriptive string
        return "discord-gateway"

    def extract_message(self, raw_data: dict) -> IncomingMessage | None:
        """Extract a message from Discord event data."""
        # TODO: Parse Discord message event
        # Discord events have author, content, channel_id, etc.
        # Example structure:
        # {
        #     "author": {"id": "123", "username": "user"},
        #     "content": "Hello!",
        #     "channel_id": "456"
        # }

        try:
            # Placeholder implementation
            sender = raw_data.get("author", {}).get("id", "unknown")
            content = raw_data.get("content", "").strip()

            if not content:
                return None

            return IncomingMessage(sender=sender, content=content)
        except Exception as e:
            logger.error("Failed to extract Discord message: %s", e)
            return None

    async def close(self) -> None:
        """Close Discord client."""
        # TODO: Cleanup Discord client connection
        logger.info("Discord channel closed")

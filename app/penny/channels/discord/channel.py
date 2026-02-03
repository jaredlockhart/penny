"""Discord implementation of MessageChannel using discord.py."""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

import discord

from penny.channels.base import IncomingMessage, MessageChannel
from penny.channels.discord.models import DiscordUser

logger = logging.getLogger(__name__)


class DiscordChannel(MessageChannel):
    """
    Discord channel implementation using discord.py.

    Unlike Signal which uses a simple WebSocket, Discord.py manages its own
    connection internally. This channel provides a message queue that the
    agent can consume, bridging the event-driven discord.py model with
    the pull-based MessageChannel interface.
    """

    def __init__(
        self,
        token: str,
        channel_id: str,
        on_message_callback: Callable[[dict], Coroutine[Any, Any, None]] | None = None,
    ):
        """
        Initialize Discord channel.

        Args:
            token: Discord bot token
            channel_id: The channel ID to listen to and send messages in
            on_message_callback: Optional callback for incoming messages
        """
        self.token = token
        self.channel_id = channel_id
        self.on_message_callback = on_message_callback

        # Set up Discord intents - need guilds to see channels
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        intents.guilds = True

        # Create Discord client
        self.client = discord.Client(intents=intents)
        self._channel: discord.TextChannel | None = None
        self._ready = asyncio.Event()
        self._bot_user_id: str | None = None

        # Register event handlers
        self._setup_events()

        logger.info("Initialized Discord channel for channel_id=%s", channel_id)

    def _setup_events(self) -> None:
        """Set up Discord event handlers."""

        @self.client.event
        async def on_ready() -> None:
            """Called when the bot is ready."""
            logger.info("Discord bot logged in as %s", self.client.user)
            self._bot_user_id = str(self.client.user.id) if self.client.user else None

            # Log available guilds and channels for debugging
            logger.info("Bot is in %d guild(s)", len(self.client.guilds))
            for guild in self.client.guilds:
                logger.info("  Guild: %s (ID: %s)", guild.name, guild.id)
                for ch in guild.text_channels[:5]:  # Log first 5 channels
                    logger.info("    Channel: %s (ID: %s)", ch.name, ch.id)

            # Get the target channel
            channel = self.client.get_channel(int(self.channel_id))
            if channel and isinstance(channel, discord.TextChannel):
                self._channel = channel
                logger.info("Connected to channel: %s", channel.name)
            else:
                logger.error(
                    "Could not find channel with ID: %s. "
                    "Make sure the bot is invited to the server and has access to this channel.",
                    self.channel_id,
                )

            self._ready.set()

        @self.client.event
        async def on_message(message: discord.Message) -> None:
            """Called when a message is received."""
            # Ignore messages from the bot itself
            if message.author == self.client.user:
                return

            # Only process messages from the configured channel
            if str(message.channel.id) != self.channel_id:
                return

            logger.debug(
                "Received Discord message from %s: %s",
                message.author.name,
                message.content[:100],
            )

            # Convert to raw dict format for extract_message
            raw_data = {
                "id": str(message.id),
                "channel_id": str(message.channel.id),
                "author": {
                    "id": str(message.author.id),
                    "username": message.author.name,
                    "discriminator": getattr(message.author, "discriminator", ""),
                    "bot": message.author.bot,
                    "global_name": getattr(message.author, "global_name", None),
                },
                "content": message.content,
                "timestamp": message.created_at.isoformat(),
                "guild_id": str(message.guild.id) if message.guild else None,
            }

            # Call the message callback if set
            if self.on_message_callback:
                await self.on_message_callback(raw_data)

    async def start(self) -> None:
        """
        Start the Discord client.

        This should be called to begin listening for messages.
        The client runs in the background.
        """
        logger.info("Starting Discord client...")
        asyncio.create_task(self.client.start(self.token))
        # Wait for the client to be ready
        await self._ready.wait()
        logger.info("Discord client is ready")

    async def send_message(
        self, recipient: str, message: str, attachments: list[str] | None = None
    ) -> bool:
        """
        Send a message via Discord.

        Args:
            recipient: Channel ID (for Discord, we send to the configured channel)
            message: Message content
            attachments: Optional list of base64-encoded attachments (not yet implemented)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Wait for client to be ready
            await self._ready.wait()

            if not self._channel:
                logger.error("Discord channel not available")
                return False

            # Discord has a 2000 character limit per message
            if len(message) > 2000:
                # Split into chunks
                chunks = [message[i : i + 2000] for i in range(0, len(message), 2000)]
                for chunk in chunks:
                    await self._channel.send(chunk)
            else:
                await self._channel.send(message)

            logger.info("Sent message to Discord channel (length: %d)", len(message))
            return True

        except discord.HTTPException as e:
            logger.error("Failed to send Discord message: %s", e)
            return False
        except Exception as e:
            logger.error("Unexpected error sending Discord message: %s", e)
            return False

    async def send_typing(self, recipient: str, typing: bool) -> bool:
        """
        Send a typing indicator via Discord.

        Args:
            recipient: Channel ID (unused, we use configured channel)
            typing: True to start typing (Discord typing lasts ~10 seconds)

        Returns:
            True if successful, False otherwise
        """
        try:
            if not typing:
                # Discord doesn't have a "stop typing" API, it auto-expires
                return True

            await self._ready.wait()

            if not self._channel:
                logger.warning("Discord channel not available for typing indicator")
                return False

            await self._channel.typing()
            logger.debug("Sent typing indicator to Discord channel")
            return True

        except discord.HTTPException as e:
            logger.warning("Failed to send typing indicator: %s", e)
            return False
        except Exception as e:
            logger.warning("Unexpected error sending typing indicator: %s", e)
            return False

    def get_connection_url(self) -> str:
        """
        Get the connection identifier for Discord.

        Returns:
            A descriptive string (Discord manages its own gateway connection)
        """
        return f"discord-gateway:channel={self.channel_id}"

    def extract_message(self, raw_data: dict) -> IncomingMessage | None:
        """
        Extract a message from Discord event data.

        Args:
            raw_data: Raw message data from Discord event

        Returns:
            IncomingMessage if valid, None if should be ignored
        """
        try:
            # Parse using Pydantic model
            author_data = raw_data.get("author", {})
            author = DiscordUser(
                id=author_data.get("id", "unknown"),
                username=author_data.get("username", "unknown"),
                discriminator=author_data.get("discriminator", ""),
                bot=author_data.get("bot", False),
                global_name=author_data.get("global_name"),
            )

            # Ignore bot messages
            if author.bot:
                logger.debug("Ignoring bot message from %s", author.username)
                return None

            # Ignore messages from ourselves
            if self._bot_user_id and author.id == self._bot_user_id:
                logger.debug("Ignoring own message")
                return None

            content = raw_data.get("content", "").strip()

            if not content:
                logger.debug("Ignoring empty message from %s", author.username)
                return None

            # Use username as sender for readability in logs/db
            sender = f"{author.username}#{author.id}"

            logger.info("Extracted Discord message - sender: %s, content: '%s'", sender, content)

            return IncomingMessage(sender=sender, content=content)

        except Exception as e:
            logger.error("Failed to extract Discord message: %s", e)
            logger.debug("Raw data: %s", raw_data)
            return None

    async def close(self) -> None:
        """Close Discord client."""
        logger.info("Closing Discord client...")
        await self.client.close()
        logger.info("Discord channel closed")

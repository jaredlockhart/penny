"""Base abstractions for communication channels."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from penny.config import Config
from penny.constants import (
    TEST_MODE_PREFIX,
    VISION_NOT_CONFIGURED_MESSAGE,
    MessageDirection,
)
from penny.database.models import MessageLog

if TYPE_CHECKING:
    from penny.agents import MessageAgent
    from penny.commands import CommandRegistry
    from penny.database import Database
    from penny.scheduler import BackgroundScheduler

logger = logging.getLogger(__name__)


class IncomingMessage(BaseModel):
    """A message received from any channel."""

    sender: str
    content: str
    quoted_text: str | None = None
    signal_timestamp: int | None = None  # Original Signal timestamp (ms since epoch)
    is_reaction: bool = False  # True if this is a reaction message
    reacted_to_external_id: str | None = None  # External ID of message being reacted to
    images: list[str] = Field(default_factory=list)  # Base64-encoded image data


class MessageChannel(ABC):
    """Abstract base class for communication channels."""

    def __init__(
        self,
        message_agent: MessageAgent,
        db: Database,
        command_registry: CommandRegistry | None = None,
    ):
        """
        Initialize channel with dependencies.

        Args:
            message_agent: Agent for processing incoming messages
            db: Database for logging messages
            command_registry: Optional command registry for handling commands
        """
        self._message_agent = message_agent
        self._db = db
        self._command_registry = command_registry
        self._scheduler: BackgroundScheduler | None = None
        self._config: Config | None = None

    def set_scheduler(self, scheduler: BackgroundScheduler) -> None:
        """Set the scheduler for message notifications."""
        self._scheduler = scheduler

    def set_command_context(self, config: Config, channel_type: str, start_time: datetime) -> None:
        """
        Set command context for command execution.

        Args:
            config: Penny config
            channel_type: Channel type ("signal" or "discord")
            start_time: Penny startup time
        """
        self._config = config

        from penny.commands import CommandContext
        from penny.ollama import OllamaClient

        # Create an Ollama client for command execution
        ollama_client = OllamaClient(
            api_url=config.ollama_api_url,
            model=config.ollama_foreground_model,
            db=self._db,
            max_retries=config.ollama_max_retries,
            retry_delay=config.ollama_retry_delay,
        )

        self._command_context = CommandContext(
            db=self._db,
            config=config,
            ollama_client=ollama_client,
            user="",  # Will be set per-command
            channel_type=channel_type,
            start_time=start_time,
            scheduler=self._scheduler,
        )

    @property
    @abstractmethod
    def sender_id(self) -> str:
        """Get the identifier for this channel's outgoing messages."""
        pass

    @abstractmethod
    async def listen(self) -> None:
        """
        Start listening for messages and dispatch to handle_message.

        This method blocks until the channel is closed.
        """
        pass

    @abstractmethod
    async def send_message(
        self,
        recipient: str,
        message: str,
        attachments: list[str] | None = None,
        quote_message: MessageLog | None = None,
    ) -> int | None:
        """
        Send a message to a recipient.

        Args:
            recipient: Identifier for the recipient (platform-specific)
            message: Message content (already prepared via prepare_outgoing)
            attachments: Optional list of base64-encoded attachments
            quote_message: Optional message to quote-reply to

        Returns:
            Signal timestamp (ms since epoch) on success, None on failure
        """
        pass

    def prepare_outgoing(self, text: str) -> str:
        """
        Prepare text for sending via this channel.

        Override in subclasses to apply channel-specific formatting.
        The result is both logged to the database and sent to the recipient,
        so quote matching works correctly.

        Args:
            text: Raw text from the agent

        Returns:
            Text formatted for this channel
        """
        return text

    async def prepare_outgoing_with_personality(self, text: str, user: str) -> str:
        """
        Apply personality transformation to outgoing text.

        If the user has a custom personality prompt set, this method sends the text
        through an LLM to transform it according to that personality.

        Args:
            text: The original message text
            user: User identifier (phone number or Discord user ID)

        Returns:
            Transformed text if custom personality exists, original text otherwise
        """
        # Check if user has a custom personality
        personality = self._db.get_personality_prompt(user)
        if not personality:
            return text

        # Import here to avoid circular dependency
        from penny.ollama import OllamaClient

        # Create a lightweight Ollama client for the transformation
        if not self._config:
            logger.warning("Config not set, skipping personality transform")
            return text

        ollama_client = OllamaClient(
            api_url=self._config.ollama_api_url,
            model=self._config.ollama_foreground_model,
            db=self._db,
            max_retries=self._config.ollama_max_retries,
            retry_delay=self._config.ollama_retry_delay,
        )

        # Build personality transform prompt
        system_prompt = (
            f"You are applying a personality filter. "
            f"Transform the following message to match this personality: "
            f"{personality.prompt_text}\n\n"
            f"IMPORTANT: Preserve the core meaning and information. "
            f"Only adjust tone, style, and phrasing."
        )
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": text,
            },
        ]

        try:
            response = await ollama_client.chat(messages=messages)
            return response.content.strip()
        except Exception as e:
            logger.error("Failed to apply personality transform: %s", e)
            # Fall back to original text if transformation fails
            return text

    @abstractmethod
    async def send_typing(self, recipient: str, typing: bool) -> bool:
        """
        Send a typing indicator to a recipient.

        Args:
            recipient: Identifier for the recipient
            typing: True to start typing, False to stop

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def extract_message(self, raw_data: dict) -> IncomingMessage | None:
        """
        Extract a message from raw channel data.

        Args:
            raw_data: Raw data from the channel (WebSocket message, API event, etc.)

        Returns:
            IncomingMessage if valid message, None if should be ignored
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the channel and cleanup resources."""
        pass

    async def _fetch_attachments(self, message: IncomingMessage, raw_data: dict) -> IncomingMessage:
        """
        Fetch attachment data for the message. Override in subclasses.

        Default implementation returns the message unchanged.
        """
        return message

    async def _typing_loop(self, recipient: str, interval: float = 4.0) -> None:
        """Send typing indicators on a loop until cancelled."""
        try:
            while True:
                await self.send_typing(recipient, True)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    async def send_status_message(self, recipient: str, content: str) -> bool:
        """
        Send a status message without logging to database.

        Used for ephemeral status indicators like startup announcements
        that shouldn't be part of conversation history.

        Args:
            recipient: Identifier for the recipient
            content: Message content

        Returns:
            True if send was successful, False otherwise
        """
        # Apply personality transformation
        transformed = await self.prepare_outgoing_with_personality(content, recipient)
        # Apply channel-specific formatting
        prepared = self.prepare_outgoing(transformed)
        external_id = await self.send_message(
            recipient, prepared, attachments=None, quote_message=None
        )
        return external_id is not None

    async def send_response(
        self,
        recipient: str,
        content: str,
        parent_id: int | None,
        attachments: list[str] | None = None,
        quote_message: MessageLog | None = None,
    ) -> int | None:
        """
        Log and send an outgoing message.

        Args:
            recipient: Identifier for the recipient
            content: Message content
            parent_id: Parent message ID for thread linking
            attachments: Optional list of base64-encoded attachments
            quote_message: Optional message to quote-reply to

        Returns:
            Database message ID if send was successful, None otherwise
        """
        # Apply personality transformation
        transformed = await self.prepare_outgoing_with_personality(content, recipient)
        # Apply channel-specific formatting
        # We log the prepared content so quote matching works correctly
        prepared = self.prepare_outgoing(transformed)
        message_id = self._db.log_message(
            MessageDirection.OUTGOING,
            self.sender_id,
            prepared,
            parent_id=parent_id,
        )
        external_id = await self.send_message(recipient, prepared, attachments, quote_message)
        # Store the external ID for future reactions and quote replies
        if external_id and message_id:
            self._db.set_external_id(message_id, str(external_id))
        return message_id if external_id is not None else None

    async def handle_message(self, envelope_data: dict) -> None:
        """
        Process an incoming message through the agent.

        This is the main message handling logic, shared by all channel implementations.
        """
        try:
            message = self.extract_message(envelope_data)
            if message is None:
                return

            # Handle reactions specially - log as message but don't respond
            if message.is_reaction:
                await self._handle_reaction(message)
                return

            # Fetch image attachments if the channel supports them
            message = await self._fetch_attachments(message, envelope_data)

            # Handle vision: check if images are present
            if message.images:
                vision_model = self._config.ollama_vision_model if self._config else None
                if not vision_model:
                    await self.send_status_message(message.sender, VISION_NOT_CONFIGURED_MESSAGE)
                    return

            # Only reset idle timers for real messages, not receipts/sync messages
            if self._scheduler:
                self._scheduler.notify_message()

            logger.info("Received message from %s: %s", message.sender, message.content)

            # Check if message is a command
            if message.content.strip().startswith("/"):
                # Most commands don't support quote-replies, but some do (e.g., /bug)
                # Extract command name to check if it supports quote-replies
                command_name = message.content.strip()[1:].split(maxsplit=1)[0].lower()
                commands_supporting_quotes = {"bug"}  # Commands that can use quote-reply metadata

                if message.quoted_text and command_name not in commands_supporting_quotes:
                    await self.send_status_message(
                        message.sender, "Threading is not supported for commands."
                    )
                    return
                await self._handle_command(message)
                return

            # Check if thread-replying to a command (quoted text is a command)
            if message.quoted_text and message.quoted_text.strip().startswith("/"):
                await self.send_status_message(
                    message.sender, "Threading is not supported for commands."
                )
                return

            # Check if thread-replying to a test mode response
            if message.quoted_text and message.quoted_text.strip().startswith(TEST_MODE_PREFIX):
                await self.send_status_message(
                    message.sender, "Threading is not supported for test mode responses."
                )
                return

            typing_task = asyncio.create_task(self._typing_loop(message.sender))
            try:
                # Notify scheduler that foreground work is starting
                if self._scheduler:
                    self._scheduler.notify_foreground_start()

                # Agent handles context preparation internally
                parent_id, response = await self._message_agent.handle(
                    content=message.content,
                    sender=message.sender,
                    quoted_text=message.quoted_text,
                    images=message.images or None,
                )

                # Log incoming message linked to parent
                incoming_id = self._db.log_message(
                    MessageDirection.INCOMING,
                    message.sender,
                    message.content,
                    parent_id=parent_id,
                    signal_timestamp=message.signal_timestamp,
                )

                answer = (
                    response.answer.strip()
                    if response.answer
                    else "Sorry, I couldn't generate a response."
                )
                # Quote-reply to the user's incoming message
                incoming_log = MessageLog(
                    id=incoming_id,
                    direction=MessageDirection.INCOMING,
                    sender=message.sender,
                    content=message.content,
                    signal_timestamp=message.signal_timestamp,
                )
                await self.send_response(
                    message.sender,
                    answer,
                    parent_id=incoming_id,
                    attachments=response.attachments or None,
                    quote_message=incoming_log,
                )
            finally:
                typing_task.cancel()
                await self.send_typing(message.sender, False)
                # Notify scheduler that foreground work is complete
                if self._scheduler:
                    self._scheduler.notify_foreground_end()

        except Exception as e:
            logger.exception("Error handling message: %s", e)

    async def _handle_reaction(self, message: IncomingMessage) -> None:
        """
        Handle a reaction message by logging it as a message in the thread.

        Reactions keep threads alive for followup without triggering an immediate response.
        """
        if not message.reacted_to_external_id:
            logger.warning("Reaction message missing reacted_to_external_id")
            return

        # Look up the message that was reacted to
        reacted_msg = self._db.find_message_by_external_id(message.reacted_to_external_id)
        if not reacted_msg or not reacted_msg.id:
            logger.warning(
                "Could not find message with external_id=%s for reaction",
                message.reacted_to_external_id,
            )
            return

        # Log the reaction as an incoming message with is_reaction=True
        self._db.log_message(
            MessageDirection.INCOMING,
            message.sender,
            message.content,  # The emoji
            parent_id=reacted_msg.id,
            is_reaction=True,
        )

        logger.info(
            "Logged reaction from %s: %s (parent_id=%d)",
            message.sender,
            message.content,
            reacted_msg.id,
        )

    async def _handle_command(self, message: IncomingMessage) -> None:
        """
        Handle a command message.

        Args:
            message: The incoming command message
        """
        if not self._command_registry:
            logger.warning("Command received but no registry configured")
            return

        # Parse command name and args
        text = message.content.strip()
        parts = text[1:].split(maxsplit=1)  # Skip leading /
        command_name = parts[0].lower()
        command_args = parts[1] if len(parts) > 1 else ""

        # Look up command
        command = self._command_registry.get(command_name)
        if not command:
            response = f"Unknown command: /{command_name}. Use /commands to see available commands."
            await self.send_status_message(message.sender, response)
            self._db.log_command(
                user=message.sender,
                channel_type=self._command_context.channel_type,
                command_name=command_name,
                command_args=command_args,
                response=response,
                error="unknown command",
            )
            return

        # Execute command with typing indicator
        typing_task = asyncio.create_task(self._typing_loop(message.sender))
        try:
            # Notify scheduler that foreground work is starting
            if self._scheduler:
                self._scheduler.notify_foreground_start()

            # Update context with current user and message
            context = self._command_context
            context.user = message.sender
            context.message = message

            result = await command.execute(command_args, context)
            response = result.text

            # Send response (with attachments if present)
            if result.attachments:
                prepared = self.prepare_outgoing(response) if response else ""
                await self.send_message(
                    message.sender, prepared, attachments=result.attachments, quote_message=None
                )
            else:
                await self.send_status_message(message.sender, response)

            # Log command execution
            self._db.log_command(
                user=message.sender,
                channel_type=context.channel_type,
                command_name=command_name,
                command_args=command_args,
                response=response,
            )

            logger.info("Executed command /%s for %s", command_name, message.sender)

        except Exception as e:
            logger.exception("Error executing command /%s: %s", command_name, e)
            error_response = f"Error executing command: {e!s}"
            await self.send_status_message(message.sender, error_response)
            self._db.log_command(
                user=message.sender,
                channel_type=self._command_context.channel_type,
                command_name=command_name,
                command_args=command_args,
                response=error_response,
                error=str(e),
            )
        finally:
            typing_task.cancel()
            await self.send_typing(message.sender, False)
            # Notify scheduler that foreground work is complete
            if self._scheduler:
                self._scheduler.notify_foreground_end()

"""Base abstractions for communication channels."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from penny.config import Config
from penny.constants import PennyConstants
from penny.database.models import MessageLog
from penny.ollama import OllamaClient
from penny.responses import PennyResponse

if TYPE_CHECKING:
    from penny.agents import ChatAgent
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
        message_agent: ChatAgent,
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

    def set_command_context(
        self,
        config: Config,
        channel_type: str,
        start_time: datetime,
        model_client: OllamaClient,
        embedding_model_client: OllamaClient | None = None,
        image_model_client: OllamaClient | None = None,
    ) -> None:
        """
        Set command context for command execution.

        Args:
            config: Penny config
            channel_type: Channel type ("signal" or "discord")
            start_time: Penny startup time
            model_client: Shared OllamaClient for commands
            embedding_model_client: Shared embedding OllamaClient for similarity
            image_model_client: Shared image generation OllamaClient for /draw
        """
        self._config = config
        self._model_client = model_client
        self._embedding_model_client = embedding_model_client

        from penny.commands import CommandContext

        self._command_context = CommandContext(
            db=self._db,
            config=config,
            model_client=model_client,
            user="",  # Will be set per-command
            channel_type=channel_type,
            start_time=start_time,
            embedding_model_client=embedding_model_client,
            image_model_client=image_model_client,
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
        prepared = self.prepare_outgoing(content)
        external_id = await self.send_message(
            recipient, prepared, attachments=None, quote_message=None
        )
        return external_id is not None

    MAX_IMAGE_PROMPT_LENGTH = 100

    async def send_response(
        self,
        recipient: str,
        content: str,
        parent_id: int | None,
        image_prompt: str,
        attachments: list[str] | None = None,
        quote_message: MessageLog | None = None,
        thought_id: int | None = None,
    ) -> int | None:
        """
        Log and send an outgoing message with an image attachment.

        Args:
            recipient: Identifier for the recipient
            content: Message content
            parent_id: Parent message ID for thread linking
            image_prompt: Short search query for image attachment (max 100 chars)
            attachments: Optional list of base64-encoded attachments
            quote_message: Optional message to quote-reply to
            thought_id: Optional FK to the thought that triggered this message

        Returns:
            Database message ID if send was successful, None otherwise
        """
        if len(image_prompt) > self.MAX_IMAGE_PROMPT_LENGTH:
            logger.warning(
                "image_prompt too long (%d chars, max %d): %s",
                len(image_prompt),
                self.MAX_IMAGE_PROMPT_LENGTH,
                image_prompt[:100],
            )
            image_prompt = image_prompt[: self.MAX_IMAGE_PROMPT_LENGTH]

        if not attachments:
            attachments = await self._resolve_image(image_prompt, attachments)

        # Apply channel-specific formatting
        # We log the prepared content so quote matching works correctly
        prepared = self.prepare_outgoing(content)
        message_id = self._db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            self.sender_id,
            prepared,
            parent_id=parent_id,
            recipient=recipient,
            thought_id=thought_id,
        )
        external_id = await self.send_message(recipient, prepared, attachments, quote_message)
        # Store the external ID for future reactions and quote replies
        if external_id and message_id:
            self._db.messages.set_external_id(message_id, str(external_id))
        logger.info("Sent response to %s (%d chars)", recipient, len(content))
        return message_id if external_id is not None else None

    @staticmethod
    def _extract_image_prompt(response) -> str | None:
        """Extract a short image search query from the agent's tool calls."""
        for tc in response.tool_calls or []:
            if tc.tool != "search":
                continue
            # Prefer single query, fall back to first of queries list
            query = tc.arguments.get("query")
            if query:
                return query
            queries = tc.arguments.get("queries")
            if queries:
                return queries[0]
        return None

    async def _resolve_image(
        self, image_prompt: str, attachments: list[str] | None
    ) -> list[str] | None:
        """Search for an image and merge it into the attachments list."""
        from penny.serper.client import search_image

        serper_key = self._config.serper_api_key if self._config else None
        image = await search_image(
            image_prompt,
            api_key=serper_key,
            max_results=int(self._config.runtime.IMAGE_MAX_RESULTS) if self._config else 5,
            timeout=self._config.runtime.IMAGE_DOWNLOAD_TIMEOUT if self._config else 10.0,
        )
        if image:
            return (attachments or []) + [image]
        return attachments

    async def handle_message(self, envelope_data: dict) -> None:
        """
        Process an incoming message through the agent.

        This is the main message handling logic, shared by all channel implementations.
        """
        try:
            message = self.extract_message(envelope_data)
            if message is None:
                return

            if message.is_reaction:
                await self._handle_reaction(message)
                return

            message = await self._fetch_attachments(message, envelope_data)

            if not await self._validate_message(message):
                return

            if await self._dispatch_command(message):
                return

            if await self._reject_unsupported_thread(message):
                return

            await self._dispatch_to_agent(message)

        except Exception as e:
            logger.exception("Error handling message: %s", e)

    async def _validate_message(self, message: IncomingMessage) -> bool:
        """Check vision config and notify scheduler. Returns False if message should be dropped."""
        if message.images:
            vision_model = self._config.ollama_vision_model if self._config else None
            if not vision_model:
                await self.send_status_message(
                    message.sender, PennyResponse.VISION_NOT_CONFIGURED_MESSAGE
                )
                return False

        if self._scheduler:
            self._scheduler.notify_message()

        logger.info("Received message from %s: %s", message.sender, message.content)
        return True

    async def _dispatch_command(self, message: IncomingMessage) -> bool:
        """Detect and route slash commands. Returns True if message was a command."""
        if not message.content.strip().startswith("/"):
            return False

        command_name = message.content.strip()[1:].split(maxsplit=1)[0].lower()
        logger.info("Command detected: /%s from %s", command_name, message.sender)
        commands_supporting_quotes = {"bug"}

        if message.quoted_text and command_name not in commands_supporting_quotes:
            prepared = self.prepare_outgoing(PennyResponse.THREADING_NOT_SUPPORTED_COMMANDS)
            await self.send_message(message.sender, prepared, attachments=None, quote_message=None)
            return True

        await self._handle_command(message)
        return True

    def _is_thread_reply_to_command(self, message: IncomingMessage) -> bool:
        """Check if the message is a thread reply to a slash command."""
        return bool(message.quoted_text and message.quoted_text.strip().startswith("/"))

    def _is_thread_reply_to_test(self, message: IncomingMessage) -> bool:
        """Check if the message is a thread reply to a test mode response."""
        return bool(
            message.quoted_text
            and message.quoted_text.strip().startswith(PennyResponse.TEST_MODE_PREFIX)
        )

    async def _reject_unsupported_thread(self, message: IncomingMessage) -> bool:
        """Reject thread replies to commands or test mode. Returns True if rejected."""
        if self._is_thread_reply_to_command(message):
            prepared = self.prepare_outgoing(PennyResponse.THREADING_NOT_SUPPORTED_COMMANDS)
            await self.send_message(message.sender, prepared, attachments=None, quote_message=None)
            return True

        if self._is_thread_reply_to_test(message):
            await self.send_status_message(
                message.sender, PennyResponse.THREADING_NOT_SUPPORTED_TEST
            )
            return True

        return False

    def _needs_profile(self, sender: str) -> bool:
        """Check if the sender has no profile set up."""
        try:
            return self._db.users.get_info(sender) is None
        except Exception:
            return False

    async def _dispatch_to_agent(self, message: IncomingMessage) -> None:
        """Run the message through the agent loop with typing indicators."""
        if self._needs_profile(message.sender):
            self._db.messages.log_message(
                PennyConstants.MessageDirection.INCOMING,
                message.sender,
                message.content,
                signal_timestamp=message.signal_timestamp,
            )
            await self.send_status_message(message.sender, PennyResponse.PROFILE_REQUIRED)
            return

        typing_task = asyncio.create_task(self._typing_loop(message.sender))
        try:
            if self._scheduler:
                self._scheduler.notify_foreground_start()

            logger.info("Dispatching to message agent for %s", message.sender)
            response = await self._message_agent.handle(
                content=message.content,
                sender=message.sender,
                images=message.images or None,
            )

            incoming_id = self._db.messages.log_message(
                PennyConstants.MessageDirection.INCOMING,
                message.sender,
                message.content,
                signal_timestamp=message.signal_timestamp,
            )

            answer = response.answer.strip() if response.answer else PennyResponse.FALLBACK_RESPONSE
            image_prompt = self._extract_image_prompt(response) or message.content[:100]
            incoming_log = MessageLog(
                id=incoming_id,
                direction=PennyConstants.MessageDirection.INCOMING,
                sender=message.sender,
                content=message.content,
                signal_timestamp=message.signal_timestamp,
            )
            sent = await self.send_response(
                message.sender,
                answer,
                parent_id=incoming_id,
                image_prompt=image_prompt,
                attachments=response.attachments or None,
                quote_message=incoming_log,
            )
            if sent is None:
                logger.error("Failed to deliver response to %s — notifying user", message.sender)
                await self.send_status_message(message.sender, PennyResponse.DELIVERY_FAILURE)
        finally:
            typing_task.cancel()
            await self.send_typing(message.sender, False)
            if self._scheduler:
                self._scheduler.notify_foreground_end()

    async def _handle_reaction(self, message: IncomingMessage) -> None:
        """Log a reaction as a regular incoming message in the thread."""
        if not message.reacted_to_external_id:
            logger.warning("Reaction message missing reacted_to_external_id")
            return

        reacted_msg = self._db.messages.find_by_external_id(message.reacted_to_external_id)
        if not reacted_msg or not reacted_msg.id:
            logger.warning(
                "Could not find message with external_id=%s for reaction",
                message.reacted_to_external_id,
            )
            return

        self._db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            message.sender,
            message.content,
            parent_id=reacted_msg.id,
        )

        logger.info(
            "Logged reaction from %s: %s (parent_id=%d)",
            message.sender,
            message.content,
            reacted_msg.id,
        )

    def _parse_command(self, text: str) -> tuple[str, str]:
        """Parse command name and arguments from a slash command string."""
        parts = text.strip()[1:].split(maxsplit=1)  # Skip leading /
        command_name = parts[0].lower()
        command_args = parts[1] if len(parts) > 1 else ""
        return command_name, command_args

    async def _execute_command(
        self, message: IncomingMessage, command_name: str, command_args: str
    ) -> None:
        """Execute a known command with typing indicator and send the result."""
        command = self._command_registry.get(command_name)  # type: ignore[union-attr]
        typing_task = asyncio.create_task(self._typing_loop(message.sender))
        try:
            context = self._command_context
            context.user = message.sender
            context.message = message

            result = await command.execute(command_args, context)  # type: ignore[union-attr]
            response = result.text

            prepared = self.prepare_outgoing(response) if response else ""
            await self.send_message(
                message.sender, prepared, attachments=result.attachments, quote_message=None
            )
            self._log_command_result(message.sender, command_name, command_args, response)
            logger.info("Executed command /%s for %s", command_name, message.sender)

        except Exception as e:
            logger.exception("Error executing command /%s: %s", command_name, e)
            error_response = PennyResponse.COMMAND_ERROR.format(error=e)
            prepared = self.prepare_outgoing(error_response)
            await self.send_message(message.sender, prepared, attachments=None, quote_message=None)
            self._log_command_result(
                message.sender, command_name, command_args, error_response, error=str(e)
            )
        finally:
            typing_task.cancel()
            await self.send_typing(message.sender, False)

    def _log_command_result(
        self,
        sender: str,
        command_name: str,
        command_args: str,
        response: str,
        error: str | None = None,
    ) -> None:
        """Log a command execution to the database."""
        self._db.messages.log_command(
            user=sender,
            channel_type=self._command_context.channel_type,
            command_name=command_name,
            command_args=command_args,
            response=response,
            error=error,
        )

    async def _handle_command(self, message: IncomingMessage) -> None:
        """Handle a command message: parse, look up, and execute."""
        if not self._command_registry:
            logger.warning("Command received but no registry configured")
            return

        command_name, command_args = self._parse_command(message.content)
        command = self._command_registry.get(command_name)

        if not command:
            response = PennyResponse.UNKNOWN_COMMAND.format(command_name=command_name)
            prepared = self.prepare_outgoing(response)
            await self.send_message(message.sender, prepared, attachments=None, quote_message=None)
            self._log_command_result(
                message.sender, command_name, command_args, response, error="unknown command"
            )
            return

        await self._execute_command(message, command_name, command_args)

"""Base abstractions for communication channels."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel

from penny.constants import MessageDirection

if TYPE_CHECKING:
    from penny.agent import MessageAgent
    from penny.database import Database
    from penny.scheduler import BackgroundScheduler

logger = logging.getLogger(__name__)


class IncomingMessage(BaseModel):
    """A message received from any channel."""

    sender: str
    content: str
    quoted_text: str | None = None


class MessageChannel(ABC):
    """Abstract base class for communication channels."""

    def __init__(
        self,
        message_agent: MessageAgent,
        db: Database,
    ):
        """
        Initialize channel with dependencies.

        Args:
            message_agent: Agent for processing incoming messages
            db: Database for logging messages
        """
        self._message_agent = message_agent
        self._db = db
        self._scheduler: BackgroundScheduler | None = None

    def set_scheduler(self, scheduler: BackgroundScheduler) -> None:
        """Set the scheduler for message notifications."""
        self._scheduler = scheduler

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
        self, recipient: str, message: str, attachments: list[str] | None = None
    ) -> bool:
        """
        Send a message to a recipient.

        Args:
            recipient: Identifier for the recipient (platform-specific)
            message: Message content (already prepared via prepare_outgoing)
            attachments: Optional list of base64-encoded attachments

        Returns:
            True if successful, False otherwise
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

    async def _typing_loop(self, recipient: str, interval: float = 4.0) -> None:
        """Send typing indicators on a loop until cancelled."""
        try:
            while True:
                await self.send_typing(recipient, True)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    async def send_response(
        self,
        recipient: str,
        content: str,
        parent_id: int | None,
        attachments: list[str] | None = None,
    ) -> bool:
        """
        Log and send an outgoing message.

        Args:
            recipient: Identifier for the recipient
            content: Message content
            parent_id: Parent message ID for thread linking
            attachments: Optional list of base64-encoded attachments

        Returns:
            True if send was successful, False otherwise
        """
        # Prepare content for this channel (formatting, escaping, etc.)
        # We log the prepared content so quote matching works correctly
        prepared = self.prepare_outgoing(content)
        self._db.log_message(
            MessageDirection.OUTGOING,
            self.sender_id,
            prepared,
            parent_id=parent_id,
        )
        return await self.send_message(recipient, prepared, attachments)

    async def handle_message(self, envelope_data: dict) -> None:
        """
        Process an incoming message through the agent.

        This is the main message handling logic, shared by all channel implementations.
        """
        try:
            if self._scheduler:
                self._scheduler.notify_message()

            message = self.extract_message(envelope_data)
            if message is None:
                return

            logger.info("Received message from %s: %s", message.sender, message.content)

            typing_task = asyncio.create_task(self._typing_loop(message.sender))
            try:
                # Agent handles context preparation internally
                parent_id, response = await self._message_agent.handle(
                    content=message.content,
                    sender=message.sender,
                    quoted_text=message.quoted_text,
                )

                # Log incoming message linked to parent
                incoming_id = self._db.log_message(
                    MessageDirection.INCOMING, message.sender, message.content, parent_id=parent_id
                )

                answer = (
                    response.answer.strip()
                    if response.answer
                    else "Sorry, I couldn't generate a response."
                )
                await self.send_response(
                    message.sender,
                    answer,
                    parent_id=incoming_id,
                    attachments=response.attachments or None,
                )
            finally:
                typing_task.cancel()
                await self.send_typing(message.sender, False)

        except Exception as e:
            logger.exception("Error handling message: %s", e)

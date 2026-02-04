"""Base abstractions for communication channels."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from typing import Any

from pydantic import BaseModel

# Type alias for message callback
MessageCallback = Callable[[dict], Coroutine[Any, Any, None]]


class IncomingMessage(BaseModel):
    """A message received from any channel."""

    sender: str
    content: str
    quoted_text: str | None = None


class MessageChannel(ABC):
    """Abstract base class for communication channels."""

    @property
    @abstractmethod
    def sender_id(self) -> str:
        """Get the identifier for this channel's outgoing messages."""
        pass

    @abstractmethod
    async def listen(self) -> None:
        """
        Start listening for messages and dispatch to callback.

        This method blocks until the channel is closed. Messages are
        dispatched to the callback provided at construction time.
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
            message: Message content
            attachments: Optional list of base64-encoded attachments

        Returns:
            True if successful, False otherwise
        """
        pass

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
    def get_connection_url(self) -> str:
        """
        Get the connection URL/endpoint for receiving messages.

        Returns:
            URL or connection string for the channel
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

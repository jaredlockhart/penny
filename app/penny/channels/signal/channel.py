"""Signal implementation of MessageChannel."""

import logging

import httpx
from pydantic import ValidationError

from penny.channels.base import IncomingMessage, MessageChannel
from penny.channels.signal.models import (
    HttpMethod,
    SendMessageRequest,
    SignalEnvelope,
    TypingIndicatorRequest,
)

logger = logging.getLogger(__name__)


class SignalChannel(MessageChannel):
    """Signal messenger channel implementation."""

    def __init__(self, api_url: str, phone_number: str):
        """
        Initialize Signal channel.

        Args:
            api_url: Base URL for signal-cli-rest-api (e.g., http://localhost:8080)
            phone_number: Registered Signal phone number
        """
        self.api_url = api_url.rstrip("/")
        self.phone_number = phone_number
        self.http_client = httpx.AsyncClient(timeout=30.0)
        logger.info("Initialized Signal channel: url=%s, number=%s", api_url, phone_number)

    async def send_message(self, recipient: str, message: str) -> bool:
        """Send a message via Signal."""
        try:
            url = f"{self.api_url}/v2/send"
            request = SendMessageRequest(
                message=message,
                number=self.phone_number,
                recipients=[recipient],
            )

            logger.debug("Sending to %s: %s", url, request.model_dump())

            response = await self.http_client.post(
                url,
                json=request.model_dump(),
            )
            response.raise_for_status()

            logger.info(
                "Sent message to %s (length: %d), status: %d",
                recipient,
                len(message),
                response.status_code,
            )
            logger.debug("Response: %s", response.text)
            return True

        except httpx.HTTPError as e:
            logger.error("Failed to send Signal message: %s", e)
            if hasattr(e, "response") and e.response is not None:
                logger.error(
                    "Response status: %d, body: %s",
                    e.response.status_code,
                    e.response.text,
                )
            return False

    async def send_typing(self, recipient: str, typing: bool) -> bool:
        """Send a typing indicator via Signal."""
        try:
            url = f"{self.api_url}/v1/typing-indicator/{self.phone_number}"
            request = TypingIndicatorRequest(recipient=recipient)

            logger.debug("Sending typing indicator to %s: %s", recipient, "started" if typing else "stopped")

            method = HttpMethod.PUT if typing else HttpMethod.DELETE
            response = await self.http_client.request(method.value, url, json=request.model_dump())

            response.raise_for_status()
            return True

        except httpx.HTTPError as e:
            logger.warning("Failed to send typing indicator: %s", e)
            return False

    def get_connection_url(self) -> str:
        """Get the WebSocket URL for receiving Signal messages."""
        ws_url = self.api_url.replace("http://", "ws://").replace("https://", "wss://")
        return f"{ws_url}/v1/receive/{self.phone_number}"

    def extract_message(self, raw_data: dict) -> IncomingMessage | None:
        """Extract a message from a Signal WebSocket envelope."""
        # Parse envelope
        envelope = self._parse_envelope(raw_data)
        if envelope is None:
            return None

        logger.debug("Processing envelope from: %s", envelope.envelope.source)

        # Check if this is a data message
        if envelope.envelope.dataMessage is None:
            logger.debug("Ignoring non-data message")
            return None

        sender = envelope.envelope.source
        content = envelope.envelope.dataMessage.message.strip()

        logger.info("Extracted - sender: %s, content: '%s'", sender, content)

        if not content:
            logger.debug("Ignoring empty message from %s", sender)
            return None

        return IncomingMessage(sender=sender, content=content)

    def _parse_envelope(self, envelope_data: dict) -> SignalEnvelope | None:
        """Parse a Signal WebSocket envelope."""
        try:
            return SignalEnvelope.model_validate(envelope_data)
        except ValidationError as e:
            logger.error("Failed to parse envelope: %s", e)
            logger.debug("Envelope data: %s", envelope_data)
            return None

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.http_client.aclose()
        logger.info("Signal channel closed")

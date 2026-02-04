"""Signal implementation of MessageChannel."""

import asyncio
import json
import logging
import re

import httpx
import websockets
from pydantic import ValidationError

from penny.channels.base import IncomingMessage, MessageCallback, MessageChannel
from penny.channels.signal.models import (
    HttpMethod,
    SendMessageRequest,
    SignalEnvelope,
    TypingIndicatorRequest,
)

logger = logging.getLogger(__name__)


class SignalChannel(MessageChannel):
    """Signal messenger channel implementation."""

    def __init__(self, api_url: str, phone_number: str, on_message: MessageCallback):
        """
        Initialize Signal channel.

        Args:
            api_url: Base URL for signal-cli-rest-api (e.g., http://localhost:8080)
            phone_number: Registered Signal phone number
            on_message: Callback for incoming messages
        """
        self.api_url = api_url.rstrip("/")
        self.phone_number = phone_number
        self._on_message = on_message
        self._running = True
        self.http_client = httpx.AsyncClient(timeout=30.0)
        logger.info("Initialized Signal channel: url=%s, number=%s", api_url, phone_number)

    @property
    def sender_id(self) -> str:
        """Get the identifier for outgoing messages (the Signal phone number)."""
        return self.phone_number

    async def listen(self) -> None:
        """Listen for incoming messages via WebSocket."""
        connection_url = self.get_connection_url()

        while self._running:
            try:
                logger.info("Connecting to channel: %s", connection_url)

                async with websockets.connect(connection_url) as websocket:
                    logger.info("Connected to Signal WebSocket")

                    while self._running:
                        try:
                            message = await asyncio.wait_for(
                                websocket.recv(),
                                timeout=30.0,
                            )

                            logger.debug("Received raw WebSocket message: %s", message[:200])

                            envelope = json.loads(message)
                            logger.info("Parsed envelope with keys: %s", envelope.keys())

                            asyncio.create_task(self._on_message(envelope))

                        except TimeoutError:
                            logger.debug("WebSocket receive timeout, continuing...")
                            continue

                        except json.JSONDecodeError as e:
                            logger.warning("Failed to parse message JSON: %s", e)
                            continue

            except websockets.exceptions.WebSocketException as e:
                logger.error("WebSocket error: %s", e)
                if self._running:
                    logger.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)

            except Exception as e:
                logger.exception("Unexpected error in message listener: %s", e)
                if self._running:
                    logger.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)

        logger.info("Message listener stopped")

    @staticmethod
    def _format_for_signal(text: str) -> str:
        """Format text for signal-cli-rest-api.

        signal-cli-rest-api supports markdown-style formatting:
        - **bold** for bold
        - *italic* for italic
        - ~strikethrough~ for strikethrough (single tilde, not double)
        - `monospace` for monospace
        """
        # Convert ~~strikethrough~~ to ~strikethrough~ (markdown uses double, signal uses single)
        text = re.sub(r"~~(.+?)~~", r"~\1~", text)
        # Remove markdown headings (keep the text)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Convert markdown links [text](url) to just text (url)
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    async def send_message(
        self, recipient: str, message: str, attachments: list[str] | None = None
    ) -> bool:
        """Send a message via Signal."""
        # Validate message is not empty
        if not message or not message.strip():
            logger.error("Attempted to send empty message to %s", recipient)
            raise ValueError("Cannot send empty or whitespace-only message")

        # Format for signal-cli-rest-api (supports markdown-style formatting)
        message = self._format_for_signal(message)

        try:
            url = f"{self.api_url}/v2/send"
            request = SendMessageRequest(
                message=message,
                number=self.phone_number,
                recipients=[recipient],
                base64_attachments=attachments if attachments else None,
            )

            logger.debug("Sending to %s: %s", url, request)

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
            resp = getattr(e, "response", None)
            if resp is not None:
                logger.error(
                    "Response status: %d, body: %s",
                    resp.status_code,
                    resp.text,
                )
            return False

    async def send_typing(self, recipient: str, typing: bool) -> bool:
        """Send a typing indicator via Signal."""
        try:
            url = f"{self.api_url}/v1/typing-indicator/{self.phone_number}"
            request = TypingIndicatorRequest(recipient=recipient)

            logger.debug(
                "Sending typing indicator to %s: %s", recipient, "started" if typing else "stopped"
            )

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

        # Extract quoted text if this is a reply
        quoted_text = None
        if envelope.envelope.dataMessage.quote and envelope.envelope.dataMessage.quote.text:
            quoted_text = envelope.envelope.dataMessage.quote.text
            logger.info("Message includes quote: '%s'", quoted_text[:100])

        return IncomingMessage(sender=sender, content=content, quoted_text=quoted_text)

    def _parse_envelope(self, envelope_data: dict) -> SignalEnvelope | None:
        """Parse a Signal WebSocket envelope."""
        try:
            return SignalEnvelope.model_validate(envelope_data)
        except ValidationError as e:
            logger.error("Failed to parse envelope: %s", e)
            logger.debug("Envelope data: %s", envelope_data)
            return None

    async def close(self) -> None:
        """Stop listening and close the HTTP client."""
        self._running = False
        await self.http_client.aclose()
        logger.info("Signal channel closed")

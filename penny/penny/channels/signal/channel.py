"""Signal implementation of MessageChannel."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import socket
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx
import websockets
from pydantic import ValidationError

from penny.channels.base import IncomingMessage, MessageChannel
from penny.channels.signal.models import (
    HttpMethod,
    SendMessageRequest,
    SendMessageResponse,
    SignalEnvelope,
    TypingIndicatorRequest,
)
from penny.constants import VISION_SUPPORTED_CONTENT_TYPES

if TYPE_CHECKING:
    from penny.agents import MessageAgent
    from penny.commands import CommandRegistry
    from penny.database import Database
    from penny.database.models import MessageLog

logger = logging.getLogger(__name__)


class SignalChannel(MessageChannel):
    """Signal messenger channel implementation."""

    def __init__(
        self,
        api_url: str,
        phone_number: str,
        message_agent: MessageAgent,
        db: Database,
        command_registry: CommandRegistry | None = None,
    ):
        """
        Initialize Signal channel.

        Args:
            api_url: Base URL for signal-cli-rest-api (e.g., http://localhost:8080)
            phone_number: Registered Signal phone number
            message_agent: Agent for processing incoming messages
            db: Database for logging messages
            command_registry: Optional command registry for handling commands
        """
        super().__init__(message_agent=message_agent, db=db, command_registry=command_registry)
        self.api_url = api_url.rstrip("/")
        self.phone_number = phone_number
        self._running = True
        self.http_client = httpx.AsyncClient(timeout=30.0)
        logger.info("Initialized Signal channel: url=%s, number=%s", api_url, phone_number)

    @property
    def sender_id(self) -> str:
        """Get the identifier for outgoing messages (the Signal phone number)."""
        return self.phone_number

    async def validate_connectivity(self) -> None:
        """
        Validate that the Signal API is reachable.

        Raises:
            ConnectionError: If the Signal API hostname cannot be resolved or is unreachable
        """
        try:
            parsed = urlparse(self.api_url)
            hostname = parsed.hostname or parsed.netloc
            port = parsed.port or (443 if parsed.scheme == "https" else 80)

            if not hostname:
                raise ValueError(f"Invalid Signal API URL: {self.api_url}")

            # Test DNS resolution
            logger.info("Validating Signal API connectivity: %s", self.api_url)
            try:
                loop = asyncio.get_running_loop()
                await loop.getaddrinfo(
                    hostname, port, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
                )
            except socket.gaierror as e:
                raise ConnectionError(
                    f"Cannot resolve Signal API hostname '{hostname}'. "
                    f"Please check SIGNAL_API_URL in your .env file. "
                    f"In Docker Compose, use 'http://signal-api:8080' not 'http://localhost:8080'. "
                    f"Original error: {e}"
                ) from e

            # Test HTTP connectivity
            try:
                response = await self.http_client.get(f"{self.api_url}/v1/about", timeout=5.0)
                response.raise_for_status()
                logger.info("Signal API connectivity validated successfully")
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                raise ConnectionError(
                    f"Cannot connect to Signal API at {self.api_url}. "
                    f"Please ensure signal-cli-rest-api is running and accessible. "
                    f"Original error: {e}"
                ) from e
            except httpx.HTTPStatusError as e:
                # 404 is expected if the /v1/about endpoint doesn't exist - that's fine
                if e.response.status_code == 404:
                    logger.info("Signal API is reachable (HTTP %d)", e.response.status_code)
                else:
                    raise ConnectionError(
                        f"Signal API returned error status {e.response.status_code}: {e}"
                    ) from e

        except (ValueError, ConnectionError):
            # Re-raise these specific errors
            raise
        except Exception as e:
            raise ConnectionError(f"Failed to validate Signal API connectivity: {e}") from e

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

                            asyncio.create_task(self.handle_message(envelope))

                        except TimeoutError:
                            continue

                        except json.JSONDecodeError as e:
                            logger.warning("Failed to parse message JSON: %s", e)
                            continue

            except (
                websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK,
            ) as e:
                logger.info("WebSocket connection closed: %s - reconnecting in 5 seconds...", e)
                if self._running:
                    await asyncio.sleep(5)

            except (socket.gaierror, OSError, ConnectionError) as e:
                logger.info(
                    "Network/DNS error connecting to Signal API: %s - reconnecting in 5s...",
                    e,
                )
                if self._running:
                    await asyncio.sleep(5)

            except websockets.exceptions.WebSocketException as e:
                logger.error("Unexpected WebSocket error: %s", e)
                if self._running:
                    logger.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)

            except Exception as e:
                logger.exception("Unexpected error in message listener: %s", e)
                if self._running:
                    logger.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)

        logger.info("Message listener stopped")

    # Regex pattern for markdown tables: header | separator | data rows
    _TABLE_PATTERN = re.compile(
        r"^(\|[^\n]+\|)\n"  # Header row
        r"(\|[-:\s|]+\|)\n"  # Separator row
        r"((?:\|[^\n]+\|\n?)+)",  # Data rows (one or more)
        re.MULTILINE,
    )

    @classmethod
    def _table_to_bullets(cls, text: str) -> str:
        """Convert markdown tables to bullet points.

        Transforms:
            | Model | Price | Type   |
            |-------|-------|--------|
            | Foo   | $100  | Basic  |
            | Bar   | $200  | Pro    |

        Into:
            **Foo**
              • Price: $100
              • Type: Basic

            **Bar**
              • Price: $200
              • Type: Pro
        """

        def convert_table(match: re.Match[str]) -> str:
            header_line, _, data_block = match.groups()
            headers = [c.strip() for c in header_line.strip("|").split("|")]

            result = []
            for line in data_block.strip().split("\n"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                if cells and cells[0]:
                    # Strip existing bold markers to avoid malformed **text**
                    title = cells[0].strip("*").strip()
                    result.append(f"**{title}**")
                    result.extend(
                        f"  • **{h}**: {c}"
                        for h, c in zip(headers[1:], cells[1:], strict=False)
                        if c
                    )
                    result.append("")  # Blank line between entries

            logger.info(
                "Converted markdown table to bullets: %d columns, %d rows",
                len(headers),
                len(data_block.strip().split("\n")),
            )
            return "\n".join(result)

        return cls._TABLE_PATTERN.sub(convert_table, text)

    def prepare_outgoing(self, text: str) -> str:
        """Format text for signal-cli-rest-api.

        signal-cli-rest-api supports markdown-style formatting:
        - **bold** for bold
        - *italic* for italic
        - ~strikethrough~ for strikethrough (single tilde, not double)
        - `monospace` for monospace
        """
        # Convert markdown tables to bullet points
        text = self._table_to_bullets(text)
        # Use placeholder for intentional strikethrough to protect during escaping
        placeholder = "\x00STRIKE\x00"
        # Convert ~~strikethrough~~ to placeholder (markdown uses double tilde)
        text = re.sub(r"~~(.+?)~~", rf"{placeholder}\1{placeholder}", text)
        # Replace remaining tildes with tilde operator (U+223C) to prevent accidental strikethrough
        # (e.g., "~50" meaning "approximately 50" shouldn't become strikethrough)
        # Zero-width space doesn't work - Signal ignores invisible characters
        text = text.replace("~", "\u223c")
        # Restore intentional strikethrough as single tilde (Signal format)
        text = text.replace(placeholder, "~")
        # Remove markdown headings (keep the text)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Convert markdown links [text](url) to just text (url)
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    async def send_message(
        self,
        recipient: str,
        message: str,
        attachments: list[str] | None = None,
        quote_message: MessageLog | None = None,
    ) -> int | None:
        """Send a message via Signal.

        Returns:
            Signal timestamp (ms since epoch) on success, None on failure
        """
        # Validate message is not empty (unless attachments are provided)
        if (not message or not message.strip()) and not attachments:
            logger.error("Attempted to send empty message to %s", recipient)
            raise ValueError("Cannot send empty or whitespace-only message")

        try:
            url = f"{self.api_url}/v2/send"

            # Build quote fields if quote_message provided
            quote_timestamp = None
            quote_author = None
            quote_text = None
            if quote_message:
                # Use the original Signal timestamp if available, otherwise fall back to datetime
                if quote_message.signal_timestamp:
                    quote_timestamp = quote_message.signal_timestamp
                else:
                    quote_timestamp = int(quote_message.timestamp.timestamp() * 1000)
                quote_author = quote_message.sender
                quote_text = quote_message.content

            request = SendMessageRequest(
                message=message,
                number=self.phone_number,
                recipients=[recipient],
                base64_attachments=attachments if attachments else None,
                quote_timestamp=quote_timestamp,
                quote_author=quote_author,
                quote_message=quote_text,
            )

            logger.debug("Sending to %s: %s", url, request)

            response = await self.http_client.post(
                url,
                json=request.model_dump(exclude_none=True),
            )
            response.raise_for_status()

            # Parse response to get the timestamp
            send_response = SendMessageResponse.model_validate(response.json())
            timestamp = send_response.timestamp

            logger.info(
                "Sent message to %s (length: %d, timestamp: %s), status: %d",
                recipient,
                len(message),
                timestamp,
                response.status_code,
            )
            logger.debug("Response: %s", response.text)
            return timestamp

        except (httpx.ConnectError, httpx.NetworkError) as e:
            logger.info("Network or DNS error sending Signal message: %s", e)
            return None

        except httpx.HTTPError as e:
            logger.error("Failed to send Signal message: %s", e)
            resp = getattr(e, "response", None)
            if resp is not None:
                logger.error(
                    "Response status: %d, body: %s",
                    resp.status_code,
                    resp.text,
                )
            return None

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

        # Check if this is a reaction
        if envelope.envelope.dataMessage.reaction:
            reaction = envelope.envelope.dataMessage.reaction
            if reaction.isRemove:
                logger.debug("Ignoring reaction removal from %s", sender)
                return None

            # Handle both string and ReactionEmoji object formats
            emoji = reaction.emoji if isinstance(reaction.emoji, str) else reaction.emoji.value

            logger.info(
                "Extracted reaction - sender: %s, emoji: %s, target: %s",
                sender,
                emoji,
                reaction.targetSentTimestamp,
            )
            return IncomingMessage(
                sender=sender,
                content=emoji,
                is_reaction=True,
                reacted_to_external_id=str(reaction.targetSentTimestamp),
            )

        # Check for text and/or attachments
        has_text = envelope.envelope.dataMessage.message is not None
        has_attachments = bool(envelope.envelope.dataMessage.attachments)

        if not has_text and not has_attachments:
            logger.debug("Ignoring message with no text and no attachments from %s", sender)
            return None

        content = (envelope.envelope.dataMessage.message or "").strip()

        if not content and not has_attachments:
            logger.debug("Ignoring empty message from %s", sender)
            return None

        logger.info("Extracted - sender: %s, content: '%s'", sender, content)

        # Extract quoted text if this is a reply
        quoted_text = None
        if envelope.envelope.dataMessage.quote and envelope.envelope.dataMessage.quote.text:
            quoted_text = envelope.envelope.dataMessage.quote.text
            logger.info("Message includes quote: '%s'", quoted_text[:100])

        # Extract the Signal timestamp for quote reply support
        signal_timestamp = envelope.envelope.dataMessage.timestamp

        return IncomingMessage(
            sender=sender,
            content=content,
            quoted_text=quoted_text,
            signal_timestamp=signal_timestamp,
        )

    async def _fetch_attachments(self, message: IncomingMessage, raw_data: dict) -> IncomingMessage:
        """Download image attachments from Signal API and add to message."""
        envelope = self._parse_envelope(raw_data)
        if not envelope or not envelope.envelope.dataMessage:
            return message

        attachments = envelope.envelope.dataMessage.attachments
        if not attachments:
            return message

        images: list[str] = []
        for attachment in attachments:
            if attachment.contentType not in VISION_SUPPORTED_CONTENT_TYPES:
                logger.debug("Skipping non-image attachment: %s", attachment.contentType)
                continue

            image_data = await self._download_attachment(attachment.id)
            if image_data:
                images.append(image_data)

        if images:
            logger.info("Downloaded %d image attachment(s)", len(images))
            return message.model_copy(update={"images": images})
        return message

    async def _download_attachment(self, attachment_id: str) -> str | None:
        """Download an attachment from Signal API and return as base64 string."""
        try:
            url = f"{self.api_url}/v1/attachments/{attachment_id}"
            response = await self.http_client.get(url)
            response.raise_for_status()

            image_b64 = base64.b64encode(response.content).decode()

            logger.info(
                "Downloaded attachment %s (%d bytes)",
                attachment_id,
                len(response.content),
            )
            return image_b64

        except httpx.HTTPError as e:
            logger.error("Failed to download attachment %s: %s", attachment_id, e)
            return None

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

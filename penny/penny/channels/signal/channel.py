"""Signal implementation of MessageChannel."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import socket
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx
import websockets
from pydantic import ValidationError

from penny.channels.base import IncomingMessage, MessageChannel
from penny.channels.signal.models import (
    DataMessage,
    HttpMethod,
    Reaction,
    SendMessageRequest,
    SendMessageResponse,
    SignalEnvelope,
    TypingIndicatorRequest,
)
from penny.constants import PennyConstants

# Error substrings that indicate a transient signal-cli transport failure.
# These appear in the 400 response body when signal-cli's socket to Signal's
# servers was broken — the request itself was valid and should be retried.
_TRANSIENT_ERROR_INDICATORS = ("SocketException", "UnexpectedErrorException")

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
        max_retries: int = 3,
        retry_delay: float = 0.5,
    ):
        """
        Initialize Signal channel.

        Args:
            api_url: Base URL for signal-cli-rest-api (e.g., http://localhost:8080)
            phone_number: Registered Signal phone number
            message_agent: Agent for processing incoming messages
            db: Database for logging messages
            command_registry: Optional command registry for handling commands
            max_retries: Number of retry attempts for transient send failures (default: 3)
            retry_delay: Base delay in seconds between retries, doubled each attempt (default: 0.5)
        """
        super().__init__(message_agent=message_agent, db=db, command_registry=command_registry)
        self.api_url = api_url.rstrip("/")
        self.phone_number = phone_number
        self._running = True
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
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

        except ValueError, ConnectionError:
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
                    await self._receive_websocket_messages(websocket)

            except (
                websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK,
            ) as e:
                await self._handle_reconnect("WebSocket connection closed", e)

            except (socket.gaierror, OSError, ConnectionError) as e:
                await self._handle_reconnect("Network/DNS error connecting to Signal API", e)

            except websockets.exceptions.WebSocketException as e:
                logger.error("Unexpected WebSocket error: %s", e)
                await self._handle_reconnect("Unexpected WebSocket error", e)

            except Exception as e:
                logger.exception("Unexpected error in message listener: %s", e)
                await self._handle_reconnect("Unexpected error in message listener", e)

        logger.info("Message listener stopped")

    async def _receive_websocket_messages(self, websocket: Any) -> None:
        """Receive and dispatch messages from an open WebSocket connection."""
        while self._running:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                logger.debug("Received raw WebSocket message: %s", message[:200])

                envelope = json.loads(message)
                logger.info("Parsed envelope with keys: %s", envelope.keys())

                asyncio.create_task(self.handle_message(envelope))

            except TimeoutError:
                continue

            except json.JSONDecodeError as e:
                logger.warning("Failed to parse message JSON: %s", e)
                continue

    async def _handle_reconnect(self, context: str, error: Exception) -> None:
        """Log a reconnection message and sleep before retrying."""
        logger.info("%s: %s - reconnecting in 5 seconds...", context, error)
        if self._running:
            await asyncio.sleep(5)

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
        text = self._table_to_bullets(text)
        text, code_blocks, urls = self._protect_blocks(text)
        text = self._sanitize_tildes(text)
        text = self._sanitize_asterisks(text)
        text = self._strip_markdown_elements(text)
        text = self._restore_blocks(text, code_blocks, urls)
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _protect_blocks(text: str) -> tuple[str, list[str], list[str]]:
        """Protect fenced code blocks and URLs from formatting changes."""
        code_blocks: list[str] = []
        urls: list[str] = []

        def _protect_code(m: re.Match[str]) -> str:
            code_blocks.append(m.group(0))
            return f"\x00CODE{len(code_blocks) - 1}\x00"

        def _protect_url(m: re.Match[str]) -> str:
            urls.append(m.group(0))
            return f"\x00URL{len(urls) - 1}\x00"

        text = re.sub(r"```[\s\S]*?```", _protect_code, text)
        text = re.sub(r"https?://[^\s<>)]+", _protect_url, text)
        return text, code_blocks, urls

    @staticmethod
    def _sanitize_tildes(text: str) -> str:
        """Convert ~~strikethrough~~ to Signal format, escape stray tildes."""
        placeholder = "\x00STRIKE\x00"
        # Convert ~~strikethrough~~ to placeholder (markdown uses double tilde)
        text = re.sub(r"~~(.+?)~~", rf"{placeholder}\1{placeholder}", text)
        # Replace remaining tildes with tilde operator (U+223C) to prevent accidental
        # strikethrough (e.g., "~50" meaning "approximately 50")
        text = text.replace("~", "\u223c")
        # Restore intentional strikethrough as single tilde (Signal format)
        text = text.replace(placeholder, "~")
        return text

    @staticmethod
    def _sanitize_asterisks(text: str) -> str:
        """Protect bold/italic pairs and remove stray asterisks.

        A lone * (e.g., footnote "$950*") cascades through Signal's parser,
        breaking **bold** markers and creating random italic spans.
        """
        bold_ph = "\x00BOLD\x00"
        italic_ph = "\x00ITALIC\x00"
        text = re.sub(r"\*\*(.+?)\*\*", rf"{bold_ph}\1{bold_ph}", text)
        text = re.sub(r"\*(.+?)\*", rf"{italic_ph}\1{italic_ph}", text)
        text = text.replace("*", "")
        text = text.replace(bold_ph, "**")
        text = text.replace(italic_ph, "*")
        return text

    @staticmethod
    def _strip_markdown_elements(text: str) -> str:
        """Remove markdown headings, horizontal rules, blockquotes, BR tags, and links."""
        # Remove markdown headings (keep the text)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Remove horizontal rules (--- or more dashes on a line by themselves)
        text = re.sub(r"^-{3,}\s*$", "", text, flags=re.MULTILINE)
        # Strip blockquote markers (Signal doesn't render > as blockquotes)
        text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
        # Convert HTML <br> tags to newlines
        text = re.sub(r"<br\s*/?>", "\n", text)
        # Convert markdown links [text](url) to just text (url)
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
        return text

    @staticmethod
    def _restore_blocks(text: str, code_blocks: list[str], urls: list[str]) -> str:
        """Restore previously protected code blocks and URLs."""
        for i, url in enumerate(urls):
            text = text.replace(f"\x00URL{i}\x00", url)
        for i, block in enumerate(code_blocks):
            text = text.replace(f"\x00CODE{i}\x00", block)
        return text

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
        if (not message or not message.strip()) and not attachments:
            logger.error("Attempted to send empty message to %s", recipient)
            raise ValueError("Cannot send empty or whitespace-only message")

        request = self._build_send_request(recipient, message, attachments, quote_message)
        return await self._post_message(request, recipient, message)

    def _build_send_request(
        self,
        recipient: str,
        message: str,
        attachments: list[str] | None,
        quote_message: MessageLog | None,
    ) -> SendMessageRequest:
        """Build a SendMessageRequest with optional quote fields."""
        quote_timestamp = None
        quote_author = None
        quote_text = None

        if quote_message:
            # Signal timestamp priority: signal_timestamp > external_id > db timestamp
            if quote_message.signal_timestamp:
                quote_timestamp = quote_message.signal_timestamp
            elif quote_message.external_id:
                quote_timestamp = int(quote_message.external_id)
            else:
                quote_timestamp = int(quote_message.timestamp.timestamp() * 1000)
            quote_author = quote_message.sender
            quote_text = quote_message.content

        return SendMessageRequest(
            message=message,
            number=self.phone_number,
            recipients=[recipient],
            base64_attachments=attachments if attachments else None,
            quote_timestamp=quote_timestamp,
            quote_author=quote_author,
            quote_message=quote_text,
        )

    async def _post_message(
        self, request: SendMessageRequest, recipient: str, message: str
    ) -> int | None:
        """Post a message to the Signal API with retry on transient errors."""
        url = f"{self.api_url}/v2/send"
        logger.debug("Sending to %s: %s", url, request)

        delay = self.retry_delay
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.http_client.post(
                    url, json=request.model_dump(exclude_none=True)
                )
                response.raise_for_status()
                return self._handle_send_response(response, recipient, message)

            except (httpx.ConnectError, httpx.NetworkError) as e:
                logger.info("Network or DNS error sending Signal message: %s", e)
                return None

            except httpx.HTTPStatusError as e:
                self._log_send_error(e)
                if self._is_transient_error(e) and attempt < self.max_retries:
                    logger.info(
                        "Transient signal-cli error — retrying in %.2fs (attempt %d/%d)",
                        delay,
                        attempt + 1,
                        self.max_retries,
                    )
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                return None

            except httpx.HTTPError as e:
                logger.error("Failed to send Signal message: %s", e)
                return None

        return None

    def _handle_send_response(
        self, response: httpx.Response, recipient: str, message: str
    ) -> int | None:
        """Parse a successful send response and return the timestamp."""
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

    @staticmethod
    def _log_send_error(error: httpx.HTTPStatusError) -> None:
        """Log details of an HTTP send error."""
        logger.error(
            "Failed to send Signal message: %s — status: %d, body: %s",
            error,
            error.response.status_code,
            error.response.text,
        )

    @staticmethod
    def _is_transient_error(error: httpx.HTTPStatusError) -> bool:
        """Check if an HTTP error is a transient signal-cli transport failure."""
        return error.response.status_code == 400 and any(
            indicator in error.response.text for indicator in _TRANSIENT_ERROR_INDICATORS
        )

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
        envelope = self._parse_envelope(raw_data)
        if envelope is None:
            return None

        logger.debug("Processing envelope from: %s", envelope.envelope.source)

        if envelope.envelope.dataMessage is None:
            logger.debug("Ignoring non-data message")
            return None

        sender = envelope.envelope.source

        # Reactions take priority — check before text messages
        if envelope.envelope.dataMessage.reaction:
            return self._extract_reaction(sender, envelope.envelope.dataMessage.reaction)

        return self._extract_data_message(sender, envelope.envelope.dataMessage)

    def _extract_reaction(self, sender: str, reaction: Reaction) -> IncomingMessage | None:
        """Extract a reaction message, or None if it's a removal."""
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

    def _extract_data_message(
        self, sender: str, data_message: DataMessage
    ) -> IncomingMessage | None:
        """Extract a text/attachment message from a Signal data message."""
        has_text = data_message.message is not None
        has_attachments = bool(data_message.attachments)

        if not has_text and not has_attachments:
            logger.debug("Ignoring message with no text and no attachments from %s", sender)
            return None

        content = (data_message.message or "").strip()

        if not content and not has_attachments:
            logger.debug("Ignoring empty message from %s", sender)
            return None

        logger.info("Extracted - sender: %s, content: '%s'", sender, content)

        quoted_text = self._extract_quoted_text(data_message)
        signal_timestamp = data_message.timestamp

        return IncomingMessage(
            sender=sender,
            content=content,
            quoted_text=quoted_text,
            signal_timestamp=signal_timestamp,
        )

    @staticmethod
    def _extract_quoted_text(data_message: DataMessage) -> str | None:
        """Extract quoted text from a reply message, if present."""
        if data_message.quote and data_message.quote.text:
            quoted_text = data_message.quote.text
            logger.info("Message includes quote: '%s'", quoted_text[:100])
            return quoted_text
        return None

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
            if attachment.contentType not in PennyConstants.VISION_SUPPORTED_CONTENT_TYPES:
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

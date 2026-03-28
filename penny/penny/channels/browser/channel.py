"""Browser extension channel — WebSocket server implementing MessageChannel."""

from __future__ import annotations

import asyncio
import contextlib
import html
import json
import logging
import re
import uuid
from typing import TYPE_CHECKING

import websockets
from pydantic import BaseModel
from websockets.asyncio.server import Server, ServerConnection

from penny.channels.base import IncomingMessage, MessageChannel
from penny.channels.browser.models import (
    BROWSER_MSG_TYPE_MESSAGE,
    BROWSER_MSG_TYPE_THOUGHT_REACTION,
    BROWSER_MSG_TYPE_THOUGHTS_REQUEST,
    BROWSER_MSG_TYPE_TOOL_RESPONSE,
    BROWSER_RESP_TYPE_MESSAGE,
    BROWSER_RESP_TYPE_STATUS,
    BROWSER_RESP_TYPE_THOUGHTS,
    BROWSER_RESP_TYPE_TYPING,
    BrowserIncoming,
    BrowserOutgoing,
    BrowserToolRequest,
    BrowserToolResponse,
)
from penny.constants import ChannelType, PennyConstants

if TYPE_CHECKING:
    from penny.agents import ChatAgent
    from penny.commands import CommandRegistry
    from penny.database import Database
    from penny.database.models import MessageLog

logger = logging.getLogger(__name__)

TOOL_REQUEST_TIMEOUT = 30.0


def _attachment_to_src(attachment: str) -> str | None:
    """Convert an attachment string to an <img> src value."""
    if attachment.startswith("http"):
        return attachment
    if attachment.startswith("data:"):
        return attachment
    # Raw base64 — assume PNG (Ollama image generation output)
    if len(attachment) > 100:
        return f"data:image/png;base64,{attachment}"
    return None


class BrowserChannel(MessageChannel):
    """WebSocket server channel for the browser extension sidebar."""

    def __init__(
        self,
        host: str,
        port: int,
        message_agent: ChatAgent,
        db: Database,
        command_registry: CommandRegistry | None = None,
    ):
        super().__init__(message_agent=message_agent, db=db, command_registry=command_registry)
        self._host = host
        self._port = port
        self._server: Server | None = None
        self._connections: dict[str, ServerConnection] = {}
        self._pending_requests: dict[str, asyncio.Future[str]] = {}

    @property
    def sender_id(self) -> str:
        """Identifier for outgoing browser messages."""
        return "penny"

    @property
    def has_tool_connection(self) -> bool:
        """Whether any browser is connected for tool execution."""
        return len(self._connections) > 0

    # --- WebSocket server ---

    async def listen(self) -> None:
        """Start the WebSocket server and block forever."""
        self._server = await websockets.serve(
            self._handle_connection,
            self._host,
            self._port,
        )
        logger.info("Browser channel listening on ws://%s:%d", self._host, self._port)
        await asyncio.Future()

    async def _handle_connection(self, ws: ServerConnection) -> None:
        """Handle a single browser extension connection."""
        logger.info("Browser connected")
        await self._send_ws(ws, BrowserOutgoing(type=BROWSER_RESP_TYPE_STATUS, connected=True))

        device_label: str | None = None
        try:
            async for raw in ws:
                device_label = await self._process_raw_message(ws, raw, device_label)
        except websockets.ConnectionClosed:
            pass
        finally:
            self._cleanup_connection(device_label)

    def _cleanup_connection(self, device_label: str | None) -> None:
        """Remove connection and reject pending requests on disconnect."""
        if device_label:
            self._connections.pop(device_label, None)
        # Reject any pending tool requests from this connection
        for _request_id, future in list(self._pending_requests.items()):
            if not future.done():
                future.set_exception(ConnectionError("Browser disconnected"))
        logger.info("Browser disconnected: %s", device_label or "unregistered")

    # --- Message dispatch ---

    async def _process_raw_message(
        self, ws: ServerConnection, raw: str | bytes, device_label: str | None
    ) -> str | None:
        """Parse and dispatch a single WebSocket message. Returns updated device_label."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON from browser: %s", str(raw)[:200])
            return device_label

        msg_type = data.get("type", "")

        if msg_type == BROWSER_MSG_TYPE_TOOL_RESPONSE:
            self._handle_tool_response(data)
            return device_label

        if msg_type == BROWSER_MSG_TYPE_THOUGHTS_REQUEST:
            await self._handle_thoughts_request(ws)
            return device_label

        if msg_type == BROWSER_MSG_TYPE_THOUGHT_REACTION:
            self._handle_thought_reaction(data)
            return device_label

        if msg_type == BROWSER_MSG_TYPE_MESSAGE:
            return await self._handle_chat_message(ws, data, device_label)

        return device_label

    def _handle_tool_response(self, data: dict) -> None:
        """Resolve a pending tool request future."""
        try:
            response = BrowserToolResponse(**data)
        except Exception:
            logger.warning("Invalid tool response: %s", str(data)[:200])
            return

        future = self._pending_requests.pop(response.request_id, None)
        if not future or future.done():
            logger.warning("No pending request for id: %s", response.request_id)
            return

        if response.error:
            future.set_exception(RuntimeError(response.error))
        else:
            future.set_result(response.result or "")

    async def _handle_thoughts_request(self, ws: ServerConnection) -> None:
        """Query recent thoughts and send them to the browser."""
        primary = self._db.users.get_primary_sender()
        if not primary:
            await self._send_ws(ws, BrowserOutgoing(type=BROWSER_RESP_TYPE_THOUGHTS))
            return

        thoughts = self._db.thoughts.get_recent(primary, limit=50)
        seed_topics = self._resolve_seed_topics(thoughts)
        cards = [
            {
                "id": t.id,
                "title": t.title or "",
                "content": self.prepare_outgoing(t.content),
                "image_url": t.image_url or "",
                "created_at": t.created_at.isoformat() if t.created_at else "",
                "notified": t.notified_at is not None,
                "seed_topic": seed_topics.get(t.preference_id, ""),
            }
            for t in thoughts
        ]

        response = {"type": BROWSER_RESP_TYPE_THOUGHTS, "thoughts": cards}
        with contextlib.suppress(websockets.ConnectionClosed):
            await ws.send(json.dumps(response))

    def _handle_thought_reaction(self, data: dict) -> None:
        """Handle a thumbs up/down reaction to a thought from the feed page."""
        thought_id = data.get("thought_id")
        emoji = data.get("emoji", "")
        if not thought_id or not emoji:
            return

        primary = self._db.users.get_primary_sender()
        if not primary:
            return

        # Mark the thought as notified
        self._db.thoughts.mark_notified(thought_id)

        # Log a synthetic outgoing message for the thought (so reaction has a parent)
        thought = self._db.thoughts.get_by_id(thought_id)
        if not thought:
            return
        outgoing_id = self._db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            self.sender_id,
            thought.content[:500],
            recipient=primary,
            thought_id=thought_id,
        )

        # Log the reaction as an incoming message (same as Signal reactions)
        if outgoing_id:
            self._db.messages.log_message(
                PennyConstants.MessageDirection.INCOMING,
                primary,
                emoji,
                parent_id=outgoing_id,
                is_reaction=True,
            )

        logger.info("Thought %d reacted with %s from feed", thought_id, emoji)

    def _resolve_seed_topics(self, thoughts: list) -> dict[int | None, str]:
        """Build a map of preference_id → preference content for seed topics."""
        pref_ids = {t.preference_id for t in thoughts if t.preference_id is not None}
        if not pref_ids:
            return {}
        result: dict[int | None, str] = {}
        for pref_id in pref_ids:
            pref = self._db.preferences.get_by_id(pref_id)
            if pref:
                result[pref_id] = pref.content
        return result

    async def _handle_chat_message(
        self, ws: ServerConnection, data: dict, device_label: str | None
    ) -> str | None:
        """Process a chat message from the browser."""
        try:
            msg = BrowserIncoming(**data)
        except Exception:
            logger.warning("Invalid chat message: %s", str(data)[:200])
            return device_label

        if not msg.content.strip():
            return device_label

        device_label = msg.sender or "browser-user"
        self._connections[device_label] = ws
        self._auto_register_device(device_label)

        envelope: dict = {"browser_sender": device_label, "content": msg.content}
        if msg.page_context:
            envelope["page_context"] = msg.page_context.model_dump()
        asyncio.create_task(self.handle_message(envelope))
        return device_label

    # --- Tool requests ---

    async def send_tool_request(self, tool: str, arguments: dict) -> str:
        """Send a tool request to a connected browser and await the response."""
        ws = self._get_tool_connection()
        if ws is None:
            raise RuntimeError("No browser connected for tool execution")

        request_id = str(uuid.uuid4())
        future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        request = BrowserToolRequest(
            request_id=request_id,
            tool=tool,
            arguments=arguments,
        )
        await self._send_ws(ws, request)

        try:
            return await asyncio.wait_for(future, timeout=TOOL_REQUEST_TIMEOUT)
        except TimeoutError as e:
            raise TimeoutError(
                f"Browser tool '{tool}' timed out after {TOOL_REQUEST_TIMEOUT}s"
            ) from e
        finally:
            self._pending_requests.pop(request_id, None)

    def _get_tool_connection(self) -> ServerConnection | None:
        """Get the first available browser connection for tool execution."""
        if self._connections:
            return next(iter(self._connections.values()))
        return None

    # --- Device registration ---

    def _auto_register_device(self, device_label: str) -> None:
        """Register the browser device if not already known."""
        self._db.devices.register(
            channel_type=ChannelType.BROWSER,
            identifier=device_label,
            label=device_label,
        )

    # --- MessageChannel interface ---

    def extract_message(self, raw_data: dict) -> IncomingMessage | None:
        """Extract a message from browser WebSocket data."""
        sender = raw_data.get("browser_sender", "browser-user")
        content = raw_data.get("content", "").strip()
        if not content:
            return None
        return IncomingMessage(
            sender=sender,
            content=content,
            channel_type=ChannelType.BROWSER,
            device_identifier=sender,
            page_context=raw_data.get("page_context"),
        )

    async def send_message(
        self,
        recipient: str,
        message: str,
        attachments: list[str] | None = None,
        quote_message: MessageLog | None = None,
    ) -> int | None:
        """Send a message to a browser client by device label."""
        ws = self._connections.get(recipient)
        if not ws:
            logger.warning("No browser connection for device: %s", recipient)
            return None
        content = self._prepend_images(message, attachments)
        await self._send_ws(ws, BrowserOutgoing(type=BROWSER_RESP_TYPE_MESSAGE, content=content))
        return 1

    @staticmethod
    def _prepend_images(message: str, attachments: list[str] | None) -> str:
        """Prepend image attachments as <img> tags before the message HTML."""
        if not attachments:
            return message
        tags: list[str] = []
        for att in attachments:
            src = _attachment_to_src(att)
            if src:
                tags.append(f'<img src="{src}" alt="image"><br>')
        return f"{''.join(tags)}{message}" if tags else message

    async def send_typing(self, recipient: str, typing: bool) -> bool:
        """Send a typing indicator to a browser client."""
        ws = self._connections.get(recipient)
        if not ws:
            return False
        await self._send_ws(ws, BrowserOutgoing(type=BROWSER_RESP_TYPE_TYPING, active=typing))
        return True

    # --- Image handling ---

    async def _resolve_image(
        self, image_prompt: str, attachments: list[str] | None
    ) -> list[str] | None:
        """Search for an image URL and inline it as an <img> tag (no base64 download)."""
        from penny.serper.client import search_image_url

        serper_key = self._config.serper_api_key if self._config else None
        url = await search_image_url(
            image_prompt,
            api_key=serper_key,
            max_results=int(self._config.runtime.IMAGE_MAX_RESULTS) if self._config else 5,
            timeout=self._config.runtime.IMAGE_DOWNLOAD_TIMEOUT if self._config else 10.0,
        )
        if url:
            return (attachments or []) + [url]
        return attachments

    # --- Markdown to HTML formatting ---

    _TABLE_PATTERN = re.compile(
        r"^(\|[^\n]+\|)\n"
        r"(\|[-:\s|]+\|)\n"
        r"((?:\|[^\n]+\|\n?)+)",
        re.MULTILINE,
    )

    def prepare_outgoing(self, text: str) -> str:
        """Convert markdown to HTML for the browser sidebar."""
        text = self._table_to_bullets(text)
        text = html.escape(text)
        text = self._convert_markdown_to_html(text)
        text = self._collapse_blank_lines(text)
        return text.strip()

    @classmethod
    def _table_to_bullets(cls, text: str) -> str:
        """Convert markdown tables to bullet points (same as Signal)."""

        def convert_table(match: re.Match[str]) -> str:
            header_line, _, data_block = match.groups()
            headers = [c.strip() for c in header_line.strip("|").split("|")]
            result = []
            for line in data_block.strip().split("\n"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                if cells and cells[0]:
                    title = cells[0].strip("*").strip()
                    result.append(f"**{title}**")
                    result.extend(
                        f"  \u2022 **{h}**: {c}"
                        for h, c in zip(headers[1:], cells[1:], strict=False)
                        if c
                    )
                    result.append("")
            return "\n".join(result)

        return cls._TABLE_PATTERN.sub(convert_table, text)

    @staticmethod
    def _convert_markdown_to_html(text: str) -> str:
        """Convert markdown formatting to HTML tags (text is already escaped)."""
        text = re.sub(r"```([\s\S]*?)```", r"<pre><code>\1</code></pre>", text)
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
        text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)
        text = re.sub(r"^#{1,6}\s+(.+)$", r"<strong>\1</strong>", text, flags=re.MULTILINE)
        text = re.sub(r"^-{3,}\s*$", "<hr>", text, flags=re.MULTILINE)
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" target="_blank">\1</a>', text)
        text = re.sub(r"(https?://[^\s<>&]+)", r'<a href="\1" target="_blank">\1</a>', text)
        text = text.replace("\n", "<br>")
        return text

    @staticmethod
    def _collapse_blank_lines(text: str) -> str:
        """Collapse multiple consecutive <br> tags."""
        return re.sub(r"(<br>){3,}", "<br><br>", text)

    # --- Connection management ---

    async def close(self) -> None:
        """Shut down the WebSocket server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("Browser channel closed")

    @staticmethod
    async def _send_ws(ws: ServerConnection, msg: BaseModel) -> None:
        """Send a message to a WebSocket connection, suppressing closed errors."""
        with contextlib.suppress(websockets.ConnectionClosed):
            await ws.send(msg.model_dump_json(exclude_none=True))


# Backward compat alias
BrowserServer = BrowserChannel

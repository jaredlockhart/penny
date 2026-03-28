"""Browser extension channel — WebSocket server implementing MessageChannel."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import TYPE_CHECKING

import websockets
from websockets.asyncio.server import Server, ServerConnection

from penny.channels.base import IncomingMessage, MessageChannel
from penny.channels.browser.models import (
    BROWSER_MSG_TYPE_MESSAGE,
    BROWSER_RESP_TYPE_MESSAGE,
    BROWSER_RESP_TYPE_STATUS,
    BROWSER_RESP_TYPE_TYPING,
    BrowserIncoming,
    BrowserOutgoing,
)
from penny.constants import ChannelType

if TYPE_CHECKING:
    from penny.agents import ChatAgent
    from penny.commands import CommandRegistry
    from penny.database import Database
    from penny.database.models import MessageLog

logger = logging.getLogger(__name__)


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

    @property
    def sender_id(self) -> str:
        """Identifier for outgoing browser messages."""
        return "penny"

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
            if device_label:
                self._connections.pop(device_label, None)
            logger.info("Browser disconnected: %s", device_label or "unregistered")

    async def _process_raw_message(
        self, ws: ServerConnection, raw: str | bytes, device_label: str | None
    ) -> str | None:
        """Parse and dispatch a single WebSocket message. Returns updated device_label."""
        try:
            data = json.loads(raw)
            msg = BrowserIncoming(**data)
        except json.JSONDecodeError, ValueError:
            logger.warning("Invalid message from browser: %s", str(raw)[:200])
            return device_label

        if msg.type != BROWSER_MSG_TYPE_MESSAGE or not msg.content.strip():
            return device_label

        device_label = msg.sender or "browser-user"
        self._connections[device_label] = ws
        self._auto_register_device(device_label)

        envelope = {"browser_sender": device_label, "content": msg.content}
        asyncio.create_task(self.handle_message(envelope))
        return device_label

    def _auto_register_device(self, device_label: str) -> None:
        """Register the browser device if not already known."""
        self._db.devices.register(
            channel_type=ChannelType.BROWSER,
            identifier=device_label,
            label=device_label,
        )

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
        await self._send_ws(ws, BrowserOutgoing(type=BROWSER_RESP_TYPE_MESSAGE, content=message))
        return 1

    async def send_typing(self, recipient: str, typing: bool) -> bool:
        """Send a typing indicator to a browser client."""
        ws = self._connections.get(recipient)
        if not ws:
            return False
        await self._send_ws(ws, BrowserOutgoing(type=BROWSER_RESP_TYPE_TYPING, active=typing))
        return True

    async def close(self) -> None:
        """Shut down the WebSocket server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("Browser channel closed")

    @staticmethod
    async def _send_ws(ws: ServerConnection, msg: BrowserOutgoing) -> None:
        """Send a message to a WebSocket connection, suppressing closed errors."""
        with contextlib.suppress(websockets.ConnectionClosed):
            await ws.send(msg.model_dump_json(exclude_none=True))


# Backward compat alias
BrowserServer = BrowserChannel

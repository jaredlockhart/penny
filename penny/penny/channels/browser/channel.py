"""Browser extension WebSocket server.

Echo-only for now — proves the connection works end-to-end.
Will be evolved into a proper MessageChannel later.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging

import websockets
from websockets.asyncio.server import Server, ServerConnection

from penny.channels.browser.models import (
    BROWSER_MSG_TYPE_MESSAGE,
    BROWSER_RESP_TYPE_MESSAGE,
    BROWSER_RESP_TYPE_STATUS,
    BrowserIncoming,
    BrowserOutgoing,
)

logger = logging.getLogger(__name__)


class BrowserServer:
    """Standalone WebSocket server for the browser extension sidebar."""

    def __init__(self, host: str = "localhost", port: int = 9090):
        self._host = host
        self._port = port
        self._server: Server | None = None

    async def start(self) -> None:
        """Start the WebSocket server and block forever."""
        self._server = await websockets.serve(
            self._handle_connection,
            self._host,
            self._port,
        )
        logger.info("Browser server listening on ws://%s:%d", self._host, self._port)
        await asyncio.Future()

    async def _handle_connection(self, ws: ServerConnection) -> None:
        """Handle a single browser extension connection."""
        logger.info("Browser connected")
        await self._send(ws, BrowserOutgoing(type=BROWSER_RESP_TYPE_STATUS, connected=True))

        try:
            async for raw in ws:
                try:
                    data = json.loads(raw)
                    msg = BrowserIncoming(**data)
                    if msg.type == BROWSER_MSG_TYPE_MESSAGE and msg.content.strip():
                        await self._send(
                            ws,
                            BrowserOutgoing(
                                type=BROWSER_RESP_TYPE_MESSAGE,
                                content=msg.content,
                            ),
                        )
                except json.JSONDecodeError, ValueError:
                    logger.warning("Invalid message from browser: %s", raw[:200])
        except websockets.ConnectionClosed:
            logger.info("Browser disconnected")

    async def close(self) -> None:
        """Shut down the server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    @staticmethod
    async def _send(ws: ServerConnection, msg: BrowserOutgoing) -> None:
        with contextlib.suppress(websockets.ConnectionClosed):
            await ws.send(msg.model_dump_json(exclude_none=True))

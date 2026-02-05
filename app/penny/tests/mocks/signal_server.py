"""Mock Signal server for integration testing."""

import asyncio
import json
import time
import uuid

from aiohttp import WSMsgType, web


class MockSignalServer:
    """Mock Signal API server with WebSocket and REST endpoints."""

    def __init__(self) -> None:
        self.outgoing_messages: list[dict] = []
        self.typing_events: list[tuple[str, str]] = []  # (action, recipient)
        self._websockets: list[web.WebSocketResponse] = []
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self.port: int | None = None

    async def start(self, port: int = 0) -> None:
        """Start the mock server on specified port (0 = random available port)."""
        self._app = web.Application()
        self._app.router.add_post("/v2/send", self._handle_send)
        self._app.router.add_put("/v1/typing-indicator/{phone_number}", self._handle_typing_start)
        self._app.router.add_delete("/v1/typing-indicator/{phone_number}", self._handle_typing_stop)
        self._app.router.add_get("/v1/receive/{phone_number}", self._handle_websocket)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "localhost", port)
        await self._site.start()

        # Get the actual port (important when port=0)
        sock = self._site._server.sockets[0]  # type: ignore[union-attr]
        self.port = sock.getsockname()[1]

    async def stop(self) -> None:
        """Stop the mock server and close all connections."""
        for ws in self._websockets:
            await ws.close()
        self._websockets.clear()

        if self._runner:
            await self._runner.cleanup()

    async def push_message(
        self,
        sender: str,
        content: str,
        quote: dict | None = None,
    ) -> None:
        """Push an incoming message to all connected WebSocket clients."""
        ts = int(time.time() * 1000)
        envelope = {
            "envelope": {
                "source": sender,
                "sourceNumber": sender,
                "sourceUuid": str(uuid.uuid4()),
                "sourceName": "Test User",
                "sourceDevice": 1,
                "timestamp": ts,
                "serverReceivedTimestamp": ts,
                "serverDeliveredTimestamp": ts,
                "dataMessage": {
                    "timestamp": ts,
                    "message": content,
                    "expiresInSeconds": 0,
                    "isExpirationUpdate": False,
                    "viewOnce": False,
                    "quote": quote,
                },
            },
            "account": "+15551234567",
        }

        for ws in self._websockets:
            if not ws.closed:
                await ws.send_json(envelope)

    async def wait_for_message(self, timeout: float = 10.0) -> dict:
        """Wait for an outgoing message to be captured."""
        start = time.time()
        initial_count = len(self.outgoing_messages)
        while time.time() - start < timeout:
            if len(self.outgoing_messages) > initial_count:
                return self.outgoing_messages[-1]
            await asyncio.sleep(0.05)
        msg = f"Timeout waiting for outgoing message after {timeout}s"
        raise TimeoutError(msg)

    async def _handle_send(self, request: web.Request) -> web.Response:
        """Handle POST /v2/send - capture outgoing messages."""
        data = await request.json()
        self.outgoing_messages.append(data)
        return web.json_response({"timestamp": int(time.time() * 1000)})

    async def _handle_typing_start(self, request: web.Request) -> web.Response:
        """Handle PUT /v1/typing-indicator/{phone_number}."""
        try:
            data = await request.json()
            recipient = data.get("recipient", "")
        except json.JSONDecodeError:
            recipient = ""
        self.typing_events.append(("start", recipient))
        return web.Response()

    async def _handle_typing_stop(self, request: web.Request) -> web.Response:
        """Handle DELETE /v1/typing-indicator/{phone_number}."""
        try:
            data = await request.json()
            recipient = data.get("recipient", "")
        except json.JSONDecodeError:
            recipient = ""
        self.typing_events.append(("stop", recipient))
        return web.Response()

    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle GET /v1/receive/{phone_number} WebSocket connection."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._websockets.append(ws)

        try:
            async for msg in ws:
                if msg.type == WSMsgType.ERROR:
                    break
        finally:
            self._websockets.remove(ws)

        return ws

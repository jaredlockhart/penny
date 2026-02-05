"""Mock Ollama server for integration testing."""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from aiohttp import web


class MockOllamaServer:
    """Mock Ollama API server for testing."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self.port: int | None = None
        self._response_handler: Callable[[dict, int], dict] | None = None
        self._request_count = 0

    async def start(self, port: int = 0) -> None:
        """Start the mock server on specified port (0 = random available port)."""
        self._app = web.Application()
        self._app.router.add_post("/api/chat", self._handle_chat)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "localhost", port)
        await self._site.start()

        sock = self._site._server.sockets[0]  # type: ignore[union-attr]
        self.port = sock.getsockname()[1]

    async def stop(self) -> None:
        """Stop the mock server."""
        if self._runner:
            await self._runner.cleanup()

    def set_response_handler(self, handler: Callable[[dict, int], dict]) -> None:
        """
        Set a custom response handler.

        Args:
            handler: Function that takes (request_data, request_count) and returns response dict
        """
        self._response_handler = handler

    def set_default_flow(
        self, search_query: str = "test query", final_response: str = "test response"
    ) -> None:
        """
        Set up default two-step flow: tool call then final response.

        Args:
            search_query: Query for the search tool call
            final_response: Final text response
        """

        def handler(request: dict, count: int) -> dict:
            if count == 1:
                # First request: return search tool call
                return self._make_tool_call_response(request, "search", {"query": search_query})
            # Subsequent requests: return final response
            return self._make_text_response(request, final_response)

        self._response_handler = handler

    async def _handle_chat(self, request: web.Request) -> web.Response:
        """Handle POST /api/chat requests."""
        data = await request.json()
        self.requests.append(data)
        self._request_count += 1

        if self._response_handler:
            response = self._response_handler(data, self._request_count)
        else:
            # Default: simple text response
            response = self._make_text_response(data, "default mock response")

        return web.json_response(response)

    def _make_text_response(self, request: dict, content: str) -> dict:
        """Create a text-only response."""
        return {
            "model": request.get("model", "test-model"),
            "created_at": datetime.now(UTC).isoformat(),
            "message": {
                "role": "assistant",
                "content": content,
            },
            "done": True,
            "total_duration": 1000000000,
            "load_duration": 100000000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 100000000,
            "eval_count": 20,
            "eval_duration": 800000000,
        }

    def _make_tool_call_response(
        self, request: dict, tool_name: str, arguments: dict[str, Any]
    ) -> dict:
        """Create a response with a tool call."""
        return {
            "model": request.get("model", "test-model"),
            "created_at": datetime.now(UTC).isoformat(),
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": tool_name,
                            "arguments": arguments,
                        }
                    }
                ],
            },
            "done": True,
            "total_duration": 1000000000,
            "load_duration": 100000000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 100000000,
            "eval_count": 20,
            "eval_duration": 800000000,
        }

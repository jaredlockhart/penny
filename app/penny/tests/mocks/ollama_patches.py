"""Patches for Ollama SDK."""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest


class MockOllamaResponse:
    """Mock response from Ollama chat API."""

    def __init__(self, response_dict: dict):
        self._data = response_dict

    def model_dump(self) -> dict:
        """Return response as dict (matches ollama SDK interface)."""
        return self._data


class MockOllamaAsyncClient:
    """Mock for ollama.AsyncClient."""

    def __init__(self, host: str | None = None):
        self.host = host
        self.requests: list[dict] = []
        self._response_handler: Callable[[dict, int], dict] | None = None
        self._request_count = 0

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
                return self._make_tool_call_response(request, "search", {"query": search_query})
            return self._make_text_response(request, final_response)

        self._response_handler = handler

    async def chat(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> MockOllamaResponse:
        """Mock chat() call."""
        request_data = {"model": model, "messages": messages, "tools": tools}
        self.requests.append(request_data)
        self._request_count += 1

        if self._response_handler:
            response_dict = self._response_handler(request_data, self._request_count)
        else:
            response_dict = self._make_text_response(request_data, "default mock response")

        return MockOllamaResponse(response_dict)

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


# Shared instance for tests to configure and inspect
_mock_client: MockOllamaAsyncClient | None = None


def _create_mock_client(host: str | None = None) -> MockOllamaAsyncClient:
    """Factory that returns the shared mock client instance."""
    global _mock_client
    if _mock_client is None:
        _mock_client = MockOllamaAsyncClient(host)
    return _mock_client


@pytest.fixture
def mock_ollama(monkeypatch: pytest.MonkeyPatch) -> MockOllamaAsyncClient:
    """Fixture to patch ollama.AsyncClient with a mock."""
    global _mock_client
    _mock_client = MockOllamaAsyncClient()
    monkeypatch.setattr("penny.ollama.client.ollama.AsyncClient", _create_mock_client)
    return _mock_client

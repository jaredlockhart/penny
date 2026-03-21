"""Patches for search-related SDKs (Perplexity, Serper)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


class MockPerplexityResponse:
    """Mock response from Perplexity API."""

    def __init__(
        self,
        text: str = "Mock search results for your query.",
        urls: list[str] | None = None,
    ):
        self.output_text = text
        self.output: list[Any] = []

        # Add URL annotations if provided
        if urls:
            mock_content = MagicMock()
            mock_content.annotations = [MagicMock(url=url) for url in urls]
            mock_message = MagicMock()
            mock_message.content = [mock_content]
            # Mark as MessageOutputItem type for URL extraction
            self.output.append(mock_message)


_captured_perplexity_queries: list[str] = []


class MockPerplexityResponses:
    """Mock for Perplexity responses API."""

    def __init__(self, response: MockPerplexityResponse | None = None):
        self._response = response or MockPerplexityResponse()

    def create(self, preset: str, input: str) -> MockPerplexityResponse:
        """Mock responses.create() call."""
        _captured_perplexity_queries.append(input)
        return self._response


class MockPerplexity:
    """Mock Perplexity client."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.responses = MockPerplexityResponses()


@pytest.fixture
def mock_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixture to patch Perplexity and image search with default mocks."""
    monkeypatch.setattr("penny.tools.search.Perplexity", MockPerplexity)
    monkeypatch.setattr("penny.tools.search.search_image", AsyncMock(return_value=None))


@pytest.fixture
def mock_search_with_results(monkeypatch: pytest.MonkeyPatch):
    """
    Fixture factory to patch Perplexity and image search with custom results.

    Usage:
        def test_something(mock_search_with_results):
            mock_search_with_results(
                text="Custom search results",
                urls=["https://example.com"],
            )
    """

    def _configure(
        text: str = "Mock search results.",
        urls: list[str] | None = None,
    ) -> None:
        response = MockPerplexityResponse(text=text, urls=urls)

        class ConfiguredPerplexity:
            def __init__(self, api_key: str):
                self.api_key = api_key
                self.responses = MockPerplexityResponses(response)

        monkeypatch.setattr("penny.tools.search.Perplexity", ConfiguredPerplexity)
        monkeypatch.setattr("penny.tools.search.search_image", AsyncMock(return_value=None))

    return _configure

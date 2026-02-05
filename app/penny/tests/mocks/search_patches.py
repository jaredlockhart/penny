"""Patches for search-related SDKs (Perplexity, DuckDuckGo)."""

from typing import Any
from unittest.mock import MagicMock

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


class MockPerplexityResponses:
    """Mock for Perplexity responses API."""

    def __init__(self, response: MockPerplexityResponse | None = None):
        self._response = response or MockPerplexityResponse()

    def create(self, preset: str, input: str) -> MockPerplexityResponse:
        """Mock responses.create() call."""
        return self._response


class MockPerplexity:
    """Mock Perplexity client."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.responses = MockPerplexityResponses()


class MockDDGS:
    """Mock DuckDuckGo Search client."""

    def __init__(self, results: list[dict] | None = None):
        self._results = results or []

    def images(self, query: str, max_results: int = 3) -> list[dict]:
        """Mock images() call - returns empty list by default."""
        return self._results


@pytest.fixture
def mock_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixture to patch Perplexity and DuckDuckGo SDKs with default mocks."""
    monkeypatch.setattr("penny.tools.builtin.Perplexity", MockPerplexity)
    monkeypatch.setattr("penny.tools.builtin.DDGS", MockDDGS)


@pytest.fixture
def mock_search_with_results(monkeypatch: pytest.MonkeyPatch):
    """
    Fixture factory to patch search SDKs with custom results.

    Usage:
        def test_something(mock_search_with_results):
            mock_search_with_results(
                text="Custom search results",
                urls=["https://example.com"],
                images=[{"image": "https://example.com/image.jpg"}]
            )
    """

    def _configure(
        text: str = "Mock search results.",
        urls: list[str] | None = None,
        images: list[dict] | None = None,
    ) -> None:
        response = MockPerplexityResponse(text=text, urls=urls)

        class ConfiguredPerplexity:
            def __init__(self, api_key: str):
                self.api_key = api_key
                self.responses = MockPerplexityResponses(response)

        class ConfiguredDDGS:
            def __init__(self) -> None:
                pass

            def images(self, query: str, max_results: int = 3) -> list[dict]:
                return images or []

        monkeypatch.setattr("penny.tools.builtin.Perplexity", ConfiguredPerplexity)
        monkeypatch.setattr("penny.tools.builtin.DDGS", ConfiguredDDGS)

    return _configure

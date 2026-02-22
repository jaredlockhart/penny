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


def _make_image_mock(images: list[dict] | None = None) -> AsyncMock:
    """Create an AsyncMock for search_image that returns None (default)."""
    mock = AsyncMock(return_value=None)
    # Store images config for tests that check it, but default returns None
    mock._images = images or []
    return mock


@pytest.fixture
def mock_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixture to patch Perplexity and image search with default mocks."""
    monkeypatch.setattr("penny.tools.builtin.Perplexity", MockPerplexity)
    mock_image = _make_image_mock()
    monkeypatch.setattr("penny.tools.builtin.search_image", mock_image)
    monkeypatch.setattr("penny.agents.base.search_image", mock_image)


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

        monkeypatch.setattr("penny.tools.builtin.Perplexity", ConfiguredPerplexity)
        mock_image = _make_image_mock(images)
        monkeypatch.setattr("penny.tools.builtin.search_image", mock_image)
        monkeypatch.setattr("penny.agents.base.search_image", mock_image)

    return _configure

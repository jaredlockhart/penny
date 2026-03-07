"""Tests for search query redaction of personal information."""

from unittest.mock import MagicMock

import perplexity as perplexity_sdk
import pytest

from penny.constants import PennyConstants
from penny.responses import PennyResponse
from penny.tools.search import SearchTool


class MockResponseNullOutput:
    """Mock Perplexity response where output is None (nullable in practice)."""

    def __init__(self, text: str = "Some results"):
        self.output_text = text
        self.output = None


class MockResponseNullResults:
    """Mock Perplexity response where a SearchResultsOutputItem has results=None."""

    def __init__(self):
        from unittest.mock import MagicMock

        from perplexity.types.output_item import SearchResultsOutputItem

        self.output_text = "Some results"
        item = MagicMock(spec=SearchResultsOutputItem)
        item.results = None
        self.output = [item]


class MockResponseNullContent:
    """Mock Perplexity response where a MessageOutputItem has content=None."""

    def __init__(self):
        from unittest.mock import MagicMock

        from perplexity.types.output_item import MessageOutputItem

        self.output_text = "Some results"
        item = MagicMock(spec=MessageOutputItem)
        item.content = None
        self.output = [item]


class MockPerplexityForNullTests:
    """Minimal Perplexity mock that returns a given response."""

    def __init__(self, response):
        self._response = response

        class _Responses:
            def __init__(self, resp):
                self._resp = resp

            def create(self, preset, input):
                return self._resp

        self.responses = _Responses(response)


def _make_search_tool(response) -> SearchTool:
    """Create a SearchTool wired to a given mock Perplexity response."""
    tool = object.__new__(SearchTool)
    tool.perplexity = MockPerplexityForNullTests(response)
    tool.db = None
    tool.redact_terms = []
    tool.skip_images = True
    tool.serper_api_key = None
    tool.image_max_results = 3
    tool.image_download_timeout = 5.0
    tool._quota_exceeded = False
    return tool


class MockRaisingPerplexity:
    """Minimal Perplexity mock that raises a given exception on create()."""

    def __init__(self, exc: Exception):
        self._exc = exc
        self.create_call_count = 0

        class _Responses:
            def __init__(self, exc):
                self._exc = exc
                self.call_count = 0

            def create(self, preset, input):
                self.call_count += 1
                raise self._exc

        self.responses = _Responses(exc)


class TestSearchTextQuotaError:
    """Tests for graceful degradation and circuit breaker on Perplexity quota errors."""

    @staticmethod
    def _make_auth_error() -> perplexity_sdk.AuthenticationError:
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}
        return perplexity_sdk.AuthenticationError(
            response=mock_response,
            body={
                "error": {
                    "message": "insufficient_quota",
                    "type": "insufficient_quota",
                    "code": 401,
                }
            },
            message="You exceeded your current quota",
        )

    @staticmethod
    def _make_tool_with_raising_perplexity(exc: Exception) -> SearchTool:
        tool = object.__new__(SearchTool)
        tool.perplexity = MockRaisingPerplexity(exc)
        tool.db = None
        tool.redact_terms = []
        tool.skip_images = True
        tool.serper_api_key = None
        tool.image_max_results = 3
        tool.image_download_timeout = 5.0
        tool.default_trigger = PennyConstants.SearchTrigger.USER_MESSAGE
        tool._quota_exceeded = False
        return tool

    @pytest.mark.asyncio
    async def test_authentication_error_returns_quota_message(self):
        """AuthenticationError (quota exceeded) returns graceful message, no raise."""
        tool = self._make_tool_with_raising_perplexity(self._make_auth_error())
        text, urls = await tool._search_text("test query")
        assert text == PennyResponse.SEARCH_QUOTA_EXCEEDED
        assert urls == []

    @pytest.mark.asyncio
    async def test_authentication_error_sets_circuit_breaker(self):
        """After AuthenticationError, _quota_exceeded flag is set to True."""
        tool = self._make_tool_with_raising_perplexity(self._make_auth_error())
        assert tool._quota_exceeded is False
        await tool._search_text("test query")
        assert tool._quota_exceeded is True

    @pytest.mark.asyncio
    async def test_circuit_breaker_skips_api_on_subsequent_calls(self):
        """After first quota error, subsequent calls skip the API entirely."""
        tool = self._make_tool_with_raising_perplexity(self._make_auth_error())
        await tool._search_text("first query")
        # Second call — circuit is tripped; Perplexity should NOT be called again
        text, urls = await tool._search_text("second query")
        assert text == PennyResponse.SEARCH_QUOTA_EXCEEDED
        assert urls == []
        assert tool.perplexity.responses.call_count == 1  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_execute_quota_error_returns_search_result(self):
        """execute() with skip_images=True returns SearchResult with quota message."""
        tool = self._make_tool_with_raising_perplexity(self._make_auth_error())
        result = await tool.execute(query="weather today", skip_images=True)
        assert result.text == PennyResponse.SEARCH_QUOTA_EXCEEDED
        assert result.urls == []


class TestSearchTextNullOutput:
    """Tests that _search_text handles None output fields without raising TypeError."""

    @pytest.mark.asyncio
    async def test_null_output_returns_text(self):
        """response.output is None — should not raise, should return output_text."""
        tool = _make_search_tool(MockResponseNullOutput("Results text"))
        text, urls = await tool._search_text("test query")
        assert text == "Results text"
        assert urls == []

    @pytest.mark.asyncio
    async def test_null_results_in_search_output_item(self):
        """SearchResultsOutputItem.results is None — should not raise."""
        tool = _make_search_tool(MockResponseNullResults())
        text, urls = await tool._search_text("test query")
        assert text == "Some results"
        assert urls == []

    @pytest.mark.asyncio
    async def test_null_content_in_message_output_item(self):
        """MessageOutputItem.content is None — should not raise."""
        tool = _make_search_tool(MockResponseNullContent())
        text, urls = await tool._search_text("test query")
        assert text == "Some results"
        assert urls == []


class TestRedactQuery:
    """Unit tests for SearchTool._redact_query()."""

    def _make_tool(self, redact_terms: list[str]) -> SearchTool:
        """Create a SearchTool with mock Perplexity client and given redact terms."""
        tool = object.__new__(SearchTool)
        tool.redact_terms = redact_terms
        return tool

    def test_no_redact_terms(self):
        tool = self._make_tool([])
        assert tool._redact_query("weather in Toronto") == "weather in Toronto"

    def test_redacts_name_case_insensitive(self):
        tool = self._make_tool(["Alex"])
        assert tool._redact_query("Alex Toronto weather") == "Toronto weather"
        assert tool._redact_query("alex Toronto weather") == "Toronto weather"
        assert tool._redact_query("ALEX Toronto weather") == "Toronto weather"

    def test_whole_word_only(self):
        tool = self._make_tool(["Ed"])
        assert tool._redact_query("Ed Sheeran music") == "Sheeran music"
        # "ed" inside "education" should not be redacted
        assert tool._redact_query("education news") == "education news"

    def test_collapses_whitespace(self):
        tool = self._make_tool(["Alex"])
        result = tool._redact_query("news for Alex in Toronto")
        assert "  " not in result
        assert result == "news for in Toronto"

    def test_multiple_occurrences(self):
        tool = self._make_tool(["Alex"])
        result = tool._redact_query("Alex likes what Alex likes")
        assert "Alex" not in result
        assert "alex" not in result.lower()

    def test_empty_terms_ignored(self):
        tool = self._make_tool(["", "Alex"])
        assert tool._redact_query("Alex news") == "news"

    def test_preserves_query_when_no_match(self):
        tool = self._make_tool(["Alex"])
        assert tool._redact_query("Toronto weather forecast") == "Toronto weather forecast"

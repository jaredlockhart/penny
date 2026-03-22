"""Tests for search query redaction of personal information."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import perplexity
import pytest

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
    tool._quota_exceeded = False
    return tool


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


def _make_quota_error() -> perplexity.AuthenticationError:
    """Build a perplexity.AuthenticationError with insufficient_quota body."""
    mock_request = MagicMock(spec=httpx.Request)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.request = mock_request
    mock_response.status_code = 401
    return perplexity.AuthenticationError(
        "insufficient_quota",
        response=mock_response,
        body={
            "error": {"type": "insufficient_quota", "message": "You exceeded your current quota"}
        },
    )


class TestQuotaExceededHandling:
    """Tests that quota-exceeded AuthenticationErrors produce a user-friendly message."""

    @pytest.mark.asyncio
    async def test_quota_error_returns_friendly_message(self):
        """_search_text returns SEARCH_QUOTA_EXCEEDED text when quota is exceeded."""
        tool = _make_search_tool(MockResponseNullOutput())
        quota_error = _make_quota_error()
        with patch.object(tool, "_call_perplexity", AsyncMock(side_effect=quota_error)):
            text, urls = await tool._search_text("test query")
        assert text == PennyResponse.SEARCH_QUOTA_EXCEEDED
        assert urls == []

    @pytest.mark.asyncio
    async def test_non_quota_auth_error_propagates(self):
        """AuthenticationError without insufficient_quota body is re-raised."""
        tool = _make_search_tool(MockResponseNullOutput())
        mock_request = MagicMock(spec=httpx.Request)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.request = mock_request
        mock_response.status_code = 401
        auth_error = perplexity.AuthenticationError(
            "invalid_api_key",
            response=mock_response,
            body={"error": {"type": "invalid_api_key", "message": "Invalid API key"}},
        )
        with (
            patch.object(tool, "_call_perplexity", AsyncMock(side_effect=auth_error)),
            pytest.raises(perplexity.AuthenticationError),
        ):
            await tool._search_text("test query")

    def test_is_quota_error_with_quota_body(self):
        """_is_quota_error returns True for insufficient_quota body."""
        assert SearchTool._is_quota_error(_make_quota_error()) is True

    def test_is_quota_error_with_other_body(self):
        """_is_quota_error returns False for non-quota body."""
        mock_request = MagicMock(spec=httpx.Request)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.request = mock_request
        mock_response.status_code = 401
        e = perplexity.AuthenticationError(
            "invalid_api_key",
            response=mock_response,
            body={"error": {"type": "invalid_api_key"}},
        )
        assert SearchTool._is_quota_error(e) is False

    def test_is_quota_error_with_none_body(self):
        """_is_quota_error returns False when body is None."""
        mock_request = MagicMock(spec=httpx.Request)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.request = mock_request
        mock_response.status_code = 401
        e = perplexity.AuthenticationError("error", response=mock_response, body=None)
        assert SearchTool._is_quota_error(e) is False

    @pytest.mark.asyncio
    async def test_circuit_breaker_skips_api_after_quota_error(self):
        """After a quota error, subsequent _search_text calls skip the API entirely."""
        tool = _make_search_tool(MockResponseNullOutput())
        quota_error = _make_quota_error()
        with patch.object(
            tool, "_call_perplexity", AsyncMock(side_effect=quota_error)
        ) as mock_call:
            text1, urls1 = await tool._search_text("first query")
            text2, urls2 = await tool._search_text("second query")
        assert text1 == PennyResponse.SEARCH_QUOTA_EXCEEDED
        assert text2 == PennyResponse.SEARCH_QUOTA_EXCEEDED
        assert urls1 == []
        assert urls2 == []
        # Only the first call hits the API; the second short-circuits via _quota_exceeded flag
        mock_call.assert_called_once()

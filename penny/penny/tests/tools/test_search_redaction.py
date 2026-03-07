"""Tests for search query redaction of personal information."""

from unittest.mock import MagicMock

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
    tool.skip_images = True
    tool.serper_api_key = None
    tool.image_max_results = 3
    tool.image_download_timeout = 5.0
    tool._quota_exceeded = False
    return tool


def _make_auth_error(error_type: str) -> perplexity.AuthenticationError:
    """Build a perplexity.AuthenticationError with the given error type in body."""
    resp = MagicMock()
    resp.status_code = 401
    resp.headers = {}
    resp.text = "Unauthorized"
    body = {"error": {"type": error_type, "message": "test error", "code": 401}}
    return perplexity.AuthenticationError("test", response=resp, body=body)


class MockPerplexityRaisesError:
    """Minimal Perplexity mock whose responses.create() raises a given error."""

    def __init__(self, error: Exception):
        class _Responses:
            def __init__(self, err: Exception):
                self._err = err

            def create(self, preset: str, input: str) -> None:
                raise self._err

        self.responses = _Responses(error)


def _make_search_tool_with_error(error: Exception) -> SearchTool:
    """Create a SearchTool whose Perplexity client raises the given error."""
    from penny.constants import PennyConstants

    tool = object.__new__(SearchTool)
    tool.perplexity = MockPerplexityRaisesError(error)
    tool.db = None
    tool.redact_terms = []
    tool.skip_images = True
    tool.serper_api_key = None
    tool.image_max_results = 3
    tool.image_download_timeout = 5.0
    tool.default_trigger = PennyConstants.SearchTrigger.USER_MESSAGE
    tool._quota_exceeded = False
    return tool


class TestPerplexityAuthError:
    """Tests that AuthenticationError from Perplexity yields user-friendly messages."""

    @pytest.mark.asyncio
    async def test_quota_exceeded_returns_friendly_message(self):
        """insufficient_quota auth error returns the quota-exceeded response constant."""
        tool = _make_search_tool_with_error(_make_auth_error("insufficient_quota"))
        text, urls = await tool._search_text("test query")
        assert text == PennyResponse.SEARCH_QUOTA_EXCEEDED
        assert urls == []

    @pytest.mark.asyncio
    async def test_bad_key_returns_auth_failed_message(self):
        """Generic auth error (bad API key) returns the auth-failed response constant."""
        tool = _make_search_tool_with_error(_make_auth_error("invalid_api_key"))
        text, urls = await tool._search_text("test query")
        assert text == PennyResponse.SEARCH_AUTH_FAILED
        assert urls == []

    @pytest.mark.asyncio
    async def test_execute_returns_search_result_on_quota_error(self):
        """execute() returns SearchResult with friendly text instead of raising on quota error."""
        tool = _make_search_tool_with_error(_make_auth_error("insufficient_quota"))
        result = await tool.execute(query="test query", skip_images=True)
        assert result.text == PennyResponse.SEARCH_QUOTA_EXCEEDED
        assert result.urls == []

    @pytest.mark.asyncio
    async def test_quota_exceeded_short_circuits_subsequent_calls(self):
        """After insufficient_quota error, subsequent calls skip the API without hitting it."""
        tool = _make_search_tool_with_error(_make_auth_error("insufficient_quota"))

        # First call: hits API, gets quota error, sets circuit-breaker flag
        text, urls = await tool._search_text("first query")
        assert text == PennyResponse.SEARCH_QUOTA_EXCEEDED
        assert urls == []
        assert tool._quota_exceeded is True

        # Replace mock with one that would raise if called — proves API is not hit
        api_calls: list[str] = []

        class TrackingResponses:
            def create(self, preset: str, input: str) -> None:
                api_calls.append(input)
                raise RuntimeError("API should not be called after quota exceeded")

        class TrackingPerplexity:
            responses = TrackingResponses()

        tool.perplexity = TrackingPerplexity()

        # Second call: short-circuits, never touches Perplexity
        text2, urls2 = await tool._search_text("second query")
        assert text2 == PennyResponse.SEARCH_QUOTA_EXCEEDED
        assert urls2 == []
        assert api_calls == []

    @pytest.mark.asyncio
    async def test_bad_key_does_not_set_quota_exceeded_flag(self):
        """A non-quota auth error does not activate the circuit-breaker."""
        tool = _make_search_tool_with_error(_make_auth_error("invalid_api_key"))
        await tool._search_text("test query")
        assert tool._quota_exceeded is False


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

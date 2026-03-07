"""Tests for search query redaction of personal information."""

from unittest.mock import MagicMock

import perplexity as perplexity_sdk
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


class MockPerplexityQuotaError:
    """Perplexity mock that raises AuthenticationError (quota exceeded)."""

    class _Responses:
        def create(self, preset, input):
            error_response = MagicMock()
            error_response.status_code = 401
            raise perplexity_sdk.AuthenticationError(
                message="insufficient_quota",
                response=error_response,
                body={"error": {"message": "insufficient_quota", "code": 401}},
            )

    def __init__(self):
        self.responses = self._Responses()


def _make_quota_error_tool() -> SearchTool:
    """Create a SearchTool wired to a Perplexity mock that raises quota AuthenticationError."""
    tool = object.__new__(SearchTool)
    tool.perplexity = MockPerplexityQuotaError()
    tool.db = None
    tool.redact_terms = []
    tool.skip_images = True
    tool.serper_api_key = None
    tool.image_max_results = 3
    tool.image_download_timeout = 5.0
    return tool


class TestSearchTextQuotaError:
    """Tests for the Perplexity quota circuit breaker."""

    @pytest.mark.asyncio
    async def test_quota_error_returns_quota_exceeded_message(self):
        """First call after quota error returns the quota-exceeded message."""
        tool = _make_quota_error_tool()
        text, urls = await tool._search_text("test query")
        assert text == PennyResponse.SEARCH_QUOTA_EXCEEDED
        assert urls == []

    @pytest.mark.asyncio
    async def test_quota_error_trips_circuit_breaker(self):
        """After a quota AuthenticationError, _quota_exceeded is True."""
        tool = _make_quota_error_tool()
        await tool._search_text("test query")
        assert SearchTool._quota_exceeded_flag is True

    @pytest.mark.asyncio
    async def test_circuit_breaker_short_circuits_subsequent_calls(self):
        """Once the breaker is tripped, subsequent calls skip the API entirely."""
        tool = _make_quota_error_tool()
        # Trip the breaker
        await tool._search_text("first query")
        # Replace mock with one that would raise if called
        tool.perplexity = MagicMock()
        tool.perplexity.responses.create.side_effect = AssertionError("API should not be called")
        text, urls = await tool._search_text("second query")
        assert text == PennyResponse.SEARCH_QUOTA_EXCEEDED
        assert urls == []

    @pytest.mark.asyncio
    async def test_circuit_breaker_shared_across_instances(self):
        """A second SearchTool instance sees the breaker tripped by the first."""
        tool_a = _make_quota_error_tool()
        await tool_a._search_text("trip the breaker")

        tool_b = _make_search_tool(MockResponseNullOutput("Would succeed if called"))
        tool_b.perplexity = MagicMock()
        tool_b.perplexity.responses.create.side_effect = AssertionError("API should not be called")
        text, urls = await tool_b._search_text("query from second instance")
        assert text == PennyResponse.SEARCH_QUOTA_EXCEEDED
        assert urls == []

    def test_new_instance_does_not_reset_circuit_breaker(self):
        """Creating a new SearchTool instance must NOT reset the shared breaker."""
        SearchTool._quota_exceeded_flag = True
        _make_quota_error_tool()  # creates a new instance
        assert SearchTool._quota_exceeded_flag is True


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

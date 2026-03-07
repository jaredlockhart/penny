"""Tests for search query redaction and quota circuit breaker."""

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
    tool.quota_state_file = None
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


def _make_quota_error() -> perplexity_sdk.AuthenticationError:
    """Build a perplexity AuthenticationError that mimics a quota-exceeded 401."""
    from unittest.mock import MagicMock

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.request = MagicMock()
    return perplexity_sdk.AuthenticationError(
        message="insufficient_quota",
        response=mock_response,
        body={"error": {"type": "insufficient_quota", "code": 401}},
    )


class MockRaisingPerplexity:
    """Perplexity mock that raises AuthenticationError on create()."""

    def __init__(self, api_key: str):
        self.api_key = api_key

        class _Responses:
            def create(self, preset, input):
                raise _make_quota_error()

        self.responses = _Responses()


def _make_quota_tool(quota_state_file=None) -> SearchTool:
    """Create a SearchTool wired to a mock Perplexity that raises quota errors."""
    tool = object.__new__(SearchTool)
    tool.perplexity = MockRaisingPerplexity("fake-key")
    tool.db = None
    tool.redact_terms = []
    tool.skip_images = True
    tool.serper_api_key = None
    tool.image_max_results = 3
    tool.image_download_timeout = 5.0
    tool.quota_state_file = quota_state_file
    return tool


@pytest.fixture(autouse=True)
def reset_quota_breaker():
    """Reset the shared circuit breaker state before each test."""
    SearchTool._quota_exceeded_flag = False
    SearchTool._quota_exceeded_at = None
    yield
    SearchTool._quota_exceeded_flag = False
    SearchTool._quota_exceeded_at = None


class TestSearchQuotaCircuitBreaker:
    """Tests for the Perplexity quota circuit breaker in SearchTool."""

    @pytest.mark.asyncio
    async def test_quota_error_trips_breaker(self):
        """AuthenticationError on first call should set _quota_exceeded_flag."""
        tool = _make_quota_tool()
        assert not SearchTool._quota_exceeded_flag
        text, urls = await tool._search_text("test query")
        assert SearchTool._quota_exceeded_flag
        assert text == PennyResponse.SEARCH_QUOTA_EXCEEDED
        assert urls == []

    @pytest.mark.asyncio
    async def test_breaker_short_circuits_subsequent_calls(self):
        """Once breaker is tripped, _search_text returns immediately without calling API."""
        tool = _make_quota_tool()
        # Trip the breaker
        await tool._search_text("first query")
        assert SearchTool._quota_exceeded_flag

        # Replace with a mock that would raise if called
        call_count = [0]

        class _CountingResponses:
            def create(self, preset, input):
                call_count[0] += 1
                raise AssertionError("API should not be called after breaker is tripped")

        tool.perplexity.responses = _CountingResponses()  # type: ignore[assignment]
        text, urls = await tool._search_text("second query")
        assert call_count[0] == 0
        assert text == PennyResponse.SEARCH_QUOTA_EXCEEDED
        assert urls == []

    @pytest.mark.asyncio
    async def test_breaker_is_shared_across_instances(self):
        """Circuit breaker flag is class-level — tripping it on one instance affects others."""
        tool_a = _make_quota_tool()
        tool_b = _make_quota_tool()

        # Trip breaker via tool_a
        await tool_a._search_text("query")
        assert SearchTool._quota_exceeded_flag

        # tool_b should see the tripped breaker without calling the API
        call_count = [0]

        class _CountingResponses:
            def create(self, preset, input):
                call_count[0] += 1
                raise AssertionError("should not be called")

        tool_b.perplexity.responses = _CountingResponses()  # type: ignore[assignment]
        text, urls = await tool_b._search_text("another query")
        assert call_count[0] == 0
        assert text == PennyResponse.SEARCH_QUOTA_EXCEEDED

    @pytest.mark.asyncio
    async def test_new_instance_does_not_reset_breaker(self):
        """Creating a new SearchTool instance must not reset the shared circuit breaker."""
        # Trip the breaker
        SearchTool._quota_exceeded_flag = True

        # Simulate creating a new instance (as /test command would do)
        new_tool = _make_quota_tool()
        assert new_tool._quota_exceeded  # breaker still tripped


class TestSearchQuotaPersistence:
    """Tests for file-based quota circuit breaker persistence across restarts."""

    @pytest.mark.asyncio
    async def test_quota_error_writes_state_file(self, tmp_path):
        """Tripping the breaker writes the timestamp to the state file."""
        state_file = tmp_path / "quota_state"
        tool = _make_quota_tool(quota_state_file=state_file)
        await tool._search_text("test query")
        assert state_file.exists()
        # File should contain a valid ISO timestamp
        from datetime import UTC, datetime

        ts = datetime.fromisoformat(state_file.read_text().strip())
        assert (datetime.now(UTC) - ts).total_seconds() < 5

    def test_restore_from_recent_file_trips_breaker(self, tmp_path):
        """On init, a recent state file restores the tripped breaker."""
        from datetime import UTC, datetime, timedelta

        state_file = tmp_path / "quota_state"
        exceeded_at = datetime.now(UTC) - timedelta(hours=2)
        state_file.write_text(exceeded_at.isoformat())

        tool = object.__new__(SearchTool)
        tool.quota_state_file = state_file
        tool._restore_quota_state()

        assert SearchTool._quota_exceeded_flag
        assert SearchTool._quota_exceeded_at == exceeded_at

    def test_restore_from_expired_file_clears_breaker(self, tmp_path):
        """On init, an expired state file deletes itself and does not trip the breaker."""
        from datetime import UTC, datetime, timedelta

        state_file = tmp_path / "quota_state"
        exceeded_at = datetime.now(UTC) - timedelta(hours=25)
        state_file.write_text(exceeded_at.isoformat())

        tool = object.__new__(SearchTool)
        tool.quota_state_file = state_file
        tool._restore_quota_state()

        assert not SearchTool._quota_exceeded_flag
        assert not state_file.exists()

    @pytest.mark.asyncio
    async def test_breaker_resets_after_cooldown(self):
        """_quota_exceeded returns False once QUOTA_COOLDOWN_HOURS has elapsed."""
        from datetime import UTC, datetime, timedelta

        # Manually set the flag as if it was tripped 25h ago
        SearchTool._quota_exceeded_flag = True
        SearchTool._quota_exceeded_at = datetime.now(UTC) - timedelta(hours=25)

        tool = _make_quota_tool()
        # Should auto-reset and return False
        assert not tool._quota_exceeded
        assert not SearchTool._quota_exceeded_flag

    def test_no_state_file_no_error(self):
        """_restore_quota_state with quota_state_file=None is a no-op."""
        tool = object.__new__(SearchTool)
        tool.quota_state_file = None
        # Should not raise
        tool._restore_quota_state()
        assert not SearchTool._quota_exceeded_flag

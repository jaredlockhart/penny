"""Tests for search query redaction of personal information."""

import asyncio

import perplexity as perplexity_sdk
import pytest

from penny.responses import PennyResponse
from penny.tools.models import SearchResult
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
    from penny.constants import PennyConstants

    tool = object.__new__(SearchTool)
    tool.perplexity = MockPerplexityForNullTests(response)
    tool.db = None
    tool.redact_terms = []
    tool.skip_images = True
    tool.serper_api_key = None
    tool.image_max_results = 3
    tool.image_download_timeout = 5.0
    tool.default_trigger = PennyConstants.SearchTrigger.USER_MESSAGE
    tool.quota_state_file = None
    return tool


def _make_quota_error() -> perplexity_sdk.AuthenticationError:
    """Build a Perplexity AuthenticationError that looks like an insufficient_quota 401."""
    from unittest.mock import MagicMock

    response = MagicMock()
    response.status_code = 401
    response.headers = {}
    response.json.return_value = {
        "error": {
            "message": "You exceeded your current quota",
            "type": "insufficient_quota",
            "code": 401,
        }
    }
    err = perplexity_sdk.AuthenticationError.__new__(perplexity_sdk.AuthenticationError)
    err.__init__(
        message="Error code: 401 - {'error': {'type': 'insufficient_quota'}}",
        response=response,
        body={
            "error": {
                "message": "You exceeded your current quota",
                "type": "insufficient_quota",
                "code": 401,
            }
        },
    )
    return err


class MockRaisingPerplexity:
    """Perplexity mock that always raises an insufficient_quota AuthenticationError."""

    def __init__(self):
        def _make_quota_error_inner() -> perplexity_sdk.AuthenticationError:
            return _make_quota_error()

        class _Responses:
            def create(self, preset, input):
                raise _make_quota_error_inner()

        self.responses = _Responses()


def _make_quota_tool(quota_state_file=None) -> SearchTool:
    """Create a SearchTool wired to a mock Perplexity that raises quota errors."""
    from penny.constants import PennyConstants

    tool = object.__new__(SearchTool)
    tool.perplexity = MockRaisingPerplexity()
    tool.db = None
    tool.redact_terms = []
    tool.skip_images = True
    tool.serper_api_key = None
    tool.image_max_results = 3
    tool.image_download_timeout = 5.0
    tool.default_trigger = PennyConstants.SearchTrigger.USER_MESSAGE
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
        await tool._search_text("first query")
        assert SearchTool._quota_exceeded_flag

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

        await tool_a._search_text("query")
        assert SearchTool._quota_exceeded_flag

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
        """Creating a new SearchTool instance via real __init__ must not reset the breaker."""
        from unittest.mock import patch

        SearchTool._quota_exceeded_flag = True
        SearchTool._quota_exceeded_at = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        )

        # Patch Perplexity constructor to avoid needing a real API key
        with patch("penny.tools.search.Perplexity"):
            new_tool = SearchTool(
                perplexity_api_key="fake-key",
                skip_images=True,
                serper_api_key=None,
                image_max_results=3,
                image_download_timeout=5.0,
                quota_state_file=None,
            )
        assert new_tool._quota_exceeded
        assert SearchTool._quota_exceeded_flag

    @pytest.mark.asyncio
    async def test_breaker_resets_after_cooldown(self):
        """_quota_exceeded returns False once QUOTA_COOLDOWN_HOURS has elapsed."""
        from datetime import UTC, datetime, timedelta

        SearchTool._quota_exceeded_flag = True
        SearchTool._quota_exceeded_at = datetime.now(UTC) - timedelta(hours=25)

        tool = _make_quota_tool()
        assert not tool._quota_exceeded
        assert not SearchTool._quota_exceeded_flag


class TestSearchQuotaViaExecute:
    """Tests that execute() handles quota errors gracefully — the exact production traceback path.

    These cover the code path seen in the production traceback:
    _execute_with_timeout -> execute() -> _search_text -> _call_perplexity
    """

    @pytest.mark.asyncio
    async def test_execute_quota_error_returns_graceful_result(self):
        """execute() must NOT raise when quota is exceeded; returns SearchResult with message."""
        tool = _make_quota_tool()
        result = await tool.execute(query="test query")
        assert isinstance(result, SearchResult)
        assert result.text == PennyResponse.SEARCH_QUOTA_EXCEEDED
        assert result.image_base64 is None

    @pytest.mark.asyncio
    async def test_execute_trips_breaker_on_first_quota_error(self):
        """First call to execute() that hits quota should trip the circuit breaker."""
        tool = _make_quota_tool()
        assert not SearchTool._quota_exceeded_flag
        await tool.execute(query="test query")
        assert SearchTool._quota_exceeded_flag

    @pytest.mark.asyncio
    async def test_execute_respects_tripped_breaker(self):
        """With breaker already tripped, execute() returns quota message without hitting API."""
        tool = _make_quota_tool()
        SearchTool._quota_exceeded_flag = True

        call_count = [0]

        class _CountingResponses:
            def create(self, preset, input):
                call_count[0] += 1
                raise AssertionError("API should not be called")

        tool.perplexity.responses = _CountingResponses()  # type: ignore[assignment]
        result = await tool.execute(query="test query")
        assert call_count[0] == 0
        assert isinstance(result, SearchResult)
        assert result.text == PennyResponse.SEARCH_QUOTA_EXCEEDED

    @pytest.mark.asyncio
    async def test_concurrent_quota_errors_both_handled_gracefully(self):
        """Concurrent execute() calls when quota first trips are both handled gracefully.

        asyncio concurrency means multiple requests can pass the _quota_exceeded check
        before any of them trips the breaker. Both must return SearchResult without raising.
        """
        tool = _make_quota_tool()
        tool.skip_images = True

        results = await asyncio.gather(
            tool.execute(query="concurrent query 1"),
            tool.execute(query="concurrent query 2"),
        )

        for result in results:
            assert isinstance(result, SearchResult), f"Expected SearchResult, got {type(result)}"
            assert result.text == PennyResponse.SEARCH_QUOTA_EXCEEDED
            assert result.image_base64 is None

        assert SearchTool._quota_exceeded_flag


class TestSearchQuotaPersistence:
    """Tests for file-based quota circuit breaker persistence across restarts."""

    @pytest.mark.asyncio
    async def test_quota_error_writes_state_file(self, tmp_path):
        """Tripping the breaker writes the timestamp to the state file."""
        from datetime import UTC, datetime

        state_file = tmp_path / "quota_state"
        tool = _make_quota_tool(quota_state_file=state_file)
        await tool._search_text("test query")
        assert state_file.exists()
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

    def test_restore_from_expired_file_clears_file(self, tmp_path):
        """On init, an expired state file is deleted and does not trip the breaker."""
        from datetime import UTC, datetime, timedelta

        state_file = tmp_path / "quota_state"
        exceeded_at = datetime.now(UTC) - timedelta(hours=25)
        state_file.write_text(exceeded_at.isoformat())

        tool = object.__new__(SearchTool)
        tool.quota_state_file = state_file
        tool._restore_quota_state()

        assert not state_file.exists()
        assert not SearchTool._quota_exceeded_flag

"""Tests for search query redaction of personal information."""

import logging
import time
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


class MockPerplexityRaisesError:
    """Minimal Perplexity mock whose responses.create() raises a given error."""

    def __init__(self, error: Exception):
        class _Responses:
            def __init__(self, err: Exception):
                self._err = err

            def create(self, preset: str, input: str) -> None:
                raise self._err

        self.responses = _Responses(error)


def _make_auth_error(error_type: str) -> perplexity.AuthenticationError:
    """Build a perplexity.AuthenticationError with the given error type in body."""
    resp = MagicMock()
    resp.status_code = 401
    resp.headers = {}
    resp.text = "Unauthorized"
    body = {"error": {"type": error_type, "message": "test error", "code": 401}}
    return perplexity.AuthenticationError("test", response=resp, body=body)


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
    tool._quota_exceeded_at = None
    tool._quota_ever_exceeded = False
    return tool


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
    tool._quota_exceeded_at = None
    tool._quota_ever_exceeded = False
    return tool


class TestPerplexityAuthError:
    """Tests for AuthenticationError handling and the time-based circuit-breaker."""

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
    async def test_quota_exceeded_trips_circuit_breaker(self):
        """After insufficient_quota, _quota_exceeded_at is set and subsequent calls skip API."""
        tool = _make_search_tool_with_error(_make_auth_error("insufficient_quota"))

        text, urls = await tool._search_text("first query")
        assert text == PennyResponse.SEARCH_QUOTA_EXCEEDED
        assert tool._quota_exceeded_at is not None

        # Replace mock with one that would raise if called — proves API is not hit
        api_calls: list[str] = []

        class TrackingResponses:
            def create(self, preset: str, input: str) -> None:
                api_calls.append(input)
                raise RuntimeError("API should not be called after quota exceeded")

        class TrackingPerplexity:
            responses = TrackingResponses()

        tool.perplexity = TrackingPerplexity()

        text2, urls2 = await tool._search_text("second query")
        assert text2 == PennyResponse.SEARCH_QUOTA_EXCEEDED
        assert urls2 == []
        assert api_calls == []

    @pytest.mark.asyncio
    async def test_bad_key_does_not_trip_circuit_breaker(self):
        """A non-quota auth error does not activate the circuit-breaker."""
        tool = _make_search_tool_with_error(_make_auth_error("invalid_api_key"))
        await tool._search_text("test query")
        assert tool._quota_exceeded_at is None

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_after_retry_window(self):
        """After _QUOTA_RETRY_SECONDS, the circuit auto-resets and allows retrying Perplexity."""
        tool = _make_search_tool_with_error(_make_auth_error("insufficient_quota"))

        # Trip the circuit-breaker
        await tool._search_text("first query")
        assert tool._quota_exceeded_at is not None

        # Simulate the retry window having elapsed
        tool._quota_exceeded_at = time.time() - SearchTool._QUOTA_RETRY_SECONDS - 1

        # The circuit should reset and try Perplexity — it will fail again, re-tripping
        text, urls = await tool._search_text("retry query")
        assert text == PennyResponse.SEARCH_QUOTA_EXCEEDED
        # Circuit is re-tripped (timestamp updated)
        assert tool._quota_exceeded_at is not None
        assert time.time() - tool._quota_exceeded_at < 5  # recently set

    @pytest.mark.asyncio
    async def test_quota_exceeded_sets_ever_exceeded_flag(self):
        """First quota hit sets _quota_ever_exceeded so retries log at WARNING."""
        tool = _make_search_tool_with_error(_make_auth_error("insufficient_quota"))
        assert tool._quota_ever_exceeded is False
        await tool._search_text("first query")
        assert tool._quota_ever_exceeded is True

    @pytest.mark.asyncio
    async def test_retry_after_reset_logs_warning_not_error(self, caplog):
        """After circuit reset with quota still exceeded, logs WARNING not ERROR."""
        tool = _make_search_tool_with_error(_make_auth_error("insufficient_quota"))

        # Trip the circuit the first time (ERROR logged)
        await tool._search_text("first query")
        assert tool._quota_ever_exceeded is True

        # Simulate retry window elapsed
        tool._quota_exceeded_at = time.time() - SearchTool._QUOTA_RETRY_SECONDS - 1

        with caplog.at_level(logging.WARNING, logger="penny.tools.search"):
            caplog.clear()
            text, urls = await tool._search_text("retry query")

        assert text == PennyResponse.SEARCH_QUOTA_EXCEEDED
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert error_records == []
        assert len(warning_records) == 1
        assert "still exceeded" in warning_records[0].message

    @pytest.mark.asyncio
    async def test_ever_exceeded_flag_not_cleared_after_successful_search(self):
        """_quota_ever_exceeded stays True after a successful call so future quota errors
        log at WARNING (not ERROR), preventing repeated monitor bug reports.

        Regression test for #680.
        """
        error = _make_auth_error("insufficient_quota")
        tool = _make_search_tool_with_error(error)

        # Trip the circuit
        await tool._search_text("first query")
        assert tool._quota_ever_exceeded is True

        # Swap in a successful mock and let the circuit reset
        tool.perplexity = MockPerplexityForNullTests(MockResponseNullOutput("Results"))
        tool._quota_exceeded_at = time.time() - SearchTool._QUOTA_RETRY_SECONDS - 1

        text, urls = await tool._search_text("successful query")
        assert text == "Results"
        # Flag is intentionally NOT cleared — future quota errors must log at WARNING.
        assert tool._quota_ever_exceeded is True

    @pytest.mark.asyncio
    async def test_quota_error_after_success_logs_warning_not_error(self, caplog):
        """A quota error after a previously-successful session logs at WARNING, not ERROR.

        Regression test for #680: quota exhausted, circuit clears after replenishment,
        then quota is exhausted again — should be WARNING (known recurring issue).
        """
        # Step 1: trip circuit, then succeed
        tool = _make_search_tool_with_error(_make_auth_error("insufficient_quota"))
        await tool._search_text("first query")
        tool.perplexity = MockPerplexityForNullTests(MockResponseNullOutput("Results"))
        tool._quota_exceeded_at = time.time() - SearchTool._QUOTA_RETRY_SECONDS - 1
        await tool._search_text("successful query")  # clears circuit, keeps flag

        # Step 2: quota exhausted again — swap back to error mock
        tool.perplexity = MockPerplexityRaisesError(_make_auth_error("insufficient_quota"))

        with caplog.at_level(logging.WARNING, logger="penny.tools.search"):
            caplog.clear()
            text, _ = await tool._search_text("quota exceeded again")

        assert text == PennyResponse.SEARCH_QUOTA_EXCEEDED
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert error_records == [], "Recurring quota error should log at WARNING, not ERROR"
        assert len(warning_records) == 1


class TestQuotaPersistence:
    """Tests for quota-exceeded state persisting to/from the database across restarts."""

    @pytest.fixture
    def db_with_engine(self, tmp_path):
        """Minimal DB-like object with a real SQLite engine for persistence tests."""
        from sqlmodel import SQLModel, create_engine

        from penny.database.models import RuntimeConfig  # noqa: F401 — registers table

        engine = create_engine(f"sqlite:///{tmp_path}/test.db")
        SQLModel.metadata.create_all(engine)

        class _MockSearches:
            def log(self, **kwargs):
                pass

        class _MockDb:
            searches = _MockSearches()

        obj = _MockDb()
        obj.engine = engine  # type: ignore[attr-defined]
        return obj

    def _make_tool_with_db(self, db, error: Exception | None = None) -> SearchTool:
        """Create a SearchTool with a real DB and optional Perplexity error."""
        from penny.constants import PennyConstants

        tool = object.__new__(SearchTool)
        if error:
            tool.perplexity = MockPerplexityRaisesError(error)
        else:
            tool.perplexity = MockPerplexityForNullTests(MockResponseNullOutput("Results"))
        tool.db = db
        tool.redact_terms = []
        tool.skip_images = True
        tool.serper_api_key = None
        tool.image_max_results = 3
        tool.image_download_timeout = 5.0
        tool.default_trigger = PennyConstants.SearchTrigger.USER_MESSAGE
        tool._quota_exceeded_at = None
        tool._quota_ever_exceeded = False
        tool._load_quota_state()
        return tool

    @pytest.mark.asyncio
    async def test_quota_exceeded_saved_to_db(self, db_with_engine):
        """Tripping the circuit saves the timestamp to RuntimeConfig in DB."""
        from sqlmodel import Session, select

        from penny.database.models import RuntimeConfig

        tool = self._make_tool_with_db(db_with_engine, _make_auth_error("insufficient_quota"))
        await tool._search_text("first query")

        assert tool._quota_exceeded_at is not None
        with Session(db_with_engine.engine) as session:
            row = session.exec(
                select(RuntimeConfig).where(RuntimeConfig.key == SearchTool._DB_QUOTA_KEY)
            ).first()
        assert row is not None
        assert float(row.value) == pytest.approx(tool._quota_exceeded_at, abs=1.0)

    @pytest.mark.asyncio
    async def test_ever_exceeded_flag_saved_to_db(self, db_with_engine):
        """Tripping the circuit also saves the ever-exceeded flag to a separate DB key."""
        from sqlmodel import Session, select

        from penny.database.models import RuntimeConfig

        tool = self._make_tool_with_db(db_with_engine, _make_auth_error("insufficient_quota"))
        await tool._search_text("first query")

        with Session(db_with_engine.engine) as session:
            row = session.exec(
                select(RuntimeConfig).where(RuntimeConfig.key == SearchTool._DB_EVER_EXCEEDED_KEY)
            ).first()
        assert row is not None
        assert row.value == "1"

    @pytest.mark.asyncio
    async def test_quota_state_restored_on_new_instance(self, db_with_engine):
        """A new SearchTool using the same DB restores quota state from DB."""
        tool1 = self._make_tool_with_db(db_with_engine, _make_auth_error("insufficient_quota"))
        await tool1._search_text("first query")
        saved_ts = tool1._quota_exceeded_at

        # Simulate restart: new instance, same DB
        tool2 = self._make_tool_with_db(db_with_engine, _make_auth_error("insufficient_quota"))
        assert tool2._quota_exceeded_at == pytest.approx(saved_ts, abs=1.0)
        assert tool2._quota_ever_exceeded is True

        # New instance should skip API — circuit is still active
        text, urls = await tool2._search_text("second query")
        assert text == PennyResponse.SEARCH_QUOTA_EXCEEDED

    @pytest.mark.asyncio
    async def test_db_entry_preserved_when_circuit_resets(self, db_with_engine):
        """When retry window elapses, the DB timestamp row is kept (not cleared) until
        a successful call. This ensures _quota_ever_exceeded is restored on restart.
        """
        from sqlmodel import Session, select

        from penny.database.models import RuntimeConfig

        tool = self._make_tool_with_db(db_with_engine, _make_auth_error("insufficient_quota"))
        await tool._search_text("first query")

        # Simulate window expiry and trigger circuit reset via _is_quota_exceeded
        tool._quota_exceeded_at = time.time() - SearchTool._QUOTA_RETRY_SECONDS - 1
        tool._is_quota_exceeded()

        # In-memory circuit is reset
        assert tool._quota_exceeded_at is None
        # But DB timestamp row is still present — preserved for restart recovery
        with Session(db_with_engine.engine) as session:
            row = session.exec(
                select(RuntimeConfig).where(RuntimeConfig.key == SearchTool._DB_QUOTA_KEY)
            ).first()
        assert row is not None

    @pytest.mark.asyncio
    async def test_restart_after_circuit_reset_logs_warning_not_error(self, db_with_engine, caplog):
        """Restart after circuit reset still sets _quota_ever_exceeded=True from DB."""
        # Step 1: trip circuit
        tool1 = self._make_tool_with_db(db_with_engine, _make_auth_error("insufficient_quota"))
        await tool1._search_text("first query")

        # Step 2: simulate circuit reset (window elapsed)
        tool1._quota_exceeded_at = time.time() - SearchTool._QUOTA_RETRY_SECONDS - 1
        tool1._is_quota_exceeded()  # resets in-memory, keeps DB

        # Step 3: simulate restart — new instance loads from DB
        tool2 = self._make_tool_with_db(db_with_engine, _make_auth_error("insufficient_quota"))
        assert tool2._quota_ever_exceeded is True

        # Step 4: retry attempt after restart — should log WARNING, not ERROR
        tool2._quota_exceeded_at = time.time() - SearchTool._QUOTA_RETRY_SECONDS - 1

        with caplog.at_level(logging.WARNING, logger="penny.tools.search"):
            caplog.clear()
            text, _ = await tool2._search_text("retry after restart")

        assert text == PennyResponse.SEARCH_QUOTA_EXCEEDED
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert error_records == [], "Expected WARNING not ERROR after restart"
        assert len(warning_records) == 1

    @pytest.mark.asyncio
    async def test_ever_exceeded_flag_survives_successful_call_in_db(self, db_with_engine):
        """After a successful call, the ever-exceeded DB key persists so restarted
        processes still restore _quota_ever_exceeded=True. Regression test for #680.
        """
        from sqlmodel import Session, select

        from penny.database.models import RuntimeConfig

        # Trip circuit with error tool
        tool = self._make_tool_with_db(db_with_engine, _make_auth_error("insufficient_quota"))
        await tool._search_text("first query")

        # Let window elapse, swap in a successful mock
        tool._quota_exceeded_at = time.time() - SearchTool._QUOTA_RETRY_SECONDS - 1
        tool.perplexity = MockPerplexityForNullTests(MockResponseNullOutput("Results"))

        text, _ = await tool._search_text("successful query")
        assert text == "Results"

        # Active circuit timestamp should be cleared
        with Session(db_with_engine.engine) as session:
            ts_row = session.exec(
                select(RuntimeConfig).where(RuntimeConfig.key == SearchTool._DB_QUOTA_KEY)
            ).first()
            ever_row = session.exec(
                select(RuntimeConfig).where(RuntimeConfig.key == SearchTool._DB_EVER_EXCEEDED_KEY)
            ).first()
        assert ts_row is None, "Active circuit timestamp should be cleared after success"
        assert ever_row is not None, "Ever-exceeded flag must persist after successful call"

    @pytest.mark.asyncio
    async def test_restart_after_success_restores_ever_exceeded(self, db_with_engine, caplog):
        """After a successful call + restart, _quota_ever_exceeded is still True because
        the ever-exceeded DB key was kept. Next quota error logs at WARNING. Regression
        test for #680.
        """
        # Step 1: trip circuit, then succeed (clears timestamp, keeps ever-exceeded flag)
        tool1 = self._make_tool_with_db(db_with_engine, _make_auth_error("insufficient_quota"))
        await tool1._search_text("first query")
        tool1._quota_exceeded_at = time.time() - SearchTool._QUOTA_RETRY_SECONDS - 1
        tool1.perplexity = MockPerplexityForNullTests(MockResponseNullOutput("Results"))
        await tool1._search_text("successful query")

        # Step 2: simulate restart — new instance loads only ever-exceeded flag
        tool2 = self._make_tool_with_db(db_with_engine, _make_auth_error("insufficient_quota"))
        assert tool2._quota_exceeded_at is None  # active circuit was cleared
        assert tool2._quota_ever_exceeded is True  # flag was preserved

        # Step 3: quota exhausted again — should log WARNING, not ERROR
        with caplog.at_level(logging.WARNING, logger="penny.tools.search"):
            caplog.clear()
            text, _ = await tool2._search_text("quota exceeded again after restart")

        assert text == PennyResponse.SEARCH_QUOTA_EXCEEDED
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert error_records == [], "Recurring quota error after restart should log WARNING"
        assert len(warning_records) == 1


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

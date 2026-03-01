"""Tests for NewsTool rate limit persistence."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from penny.database import Database
from penny.database.migrate import migrate
from penny.tools.news import RATE_LIMIT_HOURS, RATE_LIMIT_KEY, NewsTool


@pytest.fixture
def app_state_store(tmp_path):
    """Create a real AppStateStore backed by a temporary database."""
    db_path = str(tmp_path / "test_news.db")
    db = Database(db_path)
    db.create_tables()
    migrate(db_path)
    return db.app_state


def test_rate_limit_not_set_by_default(app_state_store):
    """NewsTool starts with no rate limit when DB has no persisted state."""
    tool = NewsTool(api_key="fake-key", app_state=app_state_store)
    assert not tool._is_rate_limited()


def test_apply_rate_limit_persists_to_db(app_state_store):
    """Applying a rate limit writes the deadline to the DB."""
    tool = NewsTool(api_key="fake-key", app_state=app_state_store)
    tool._apply_rate_limit()

    assert tool._is_rate_limited()
    persisted = app_state_store.get_datetime(RATE_LIMIT_KEY)
    assert persisted is not None
    expected_min = datetime.now(UTC) + timedelta(hours=RATE_LIMIT_HOURS - 1)
    assert persisted > expected_min


def test_rate_limit_survives_restart(app_state_store):
    """Rate limit set on one NewsTool instance is honoured by a new instance (simulates restart)."""
    tool1 = NewsTool(api_key="fake-key", app_state=app_state_store)
    tool1._apply_rate_limit()

    # Simulate container restart: create a fresh NewsTool with the same store
    tool2 = NewsTool(api_key="fake-key", app_state=app_state_store)
    assert tool2._is_rate_limited()


@pytest.mark.asyncio
async def test_search_skipped_when_rate_limited(app_state_store):
    """search() returns empty list without calling the API when rate limited."""
    tool = NewsTool(api_key="fake-key", app_state=app_state_store)
    tool._apply_rate_limit()

    # No API call should be made — search returns empty immediately
    results = await tool.search(["test query"])
    assert results == []


def test_rate_limit_expires(app_state_store):
    """An expired rate limit deadline is treated as no rate limit."""
    tool = NewsTool(api_key="fake-key", app_state=app_state_store)
    # Manually set an already-expired deadline
    expired = datetime.now(UTC) - timedelta(hours=1)
    tool._rate_limited_until = expired

    assert not tool._is_rate_limited()
    assert tool._rate_limited_until is None  # cleared on expiry check


def test_no_app_state_still_works():
    """NewsTool without an AppStateStore operates normally (in-memory only)."""
    tool = NewsTool(api_key="fake-key", app_state=None)
    assert not tool._is_rate_limited()
    tool._apply_rate_limit()
    assert tool._is_rate_limited()
    # No DB write attempted — no crash

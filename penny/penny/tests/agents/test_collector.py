"""Unit tests for the dispatcher Collector — picks ready collections per cycle.

Construction-level + dispatch-selection tests only.  Full lifecycle
integration (scheduling, log → write → cursor advance) is exercised
through the existing test_chat_agent / test_message integration tests
plus the migrated likes/dislikes/knowledge prompts.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from penny.agents.collector import Collector
from penny.agents.models import ControllerResponse, ToolCallRecord
from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory_store import RecallMode
from penny.database.models import Memory
from penny.llm.client import LlmClient


def _llm_client() -> LlmClient:
    return LlmClient(
        api_url="http://localhost:11434",
        model="test-model",
        max_retries=1,
        retry_delay=0.0,
    )


def _make_collector(test_config, tmp_path) -> tuple[Collector, Database]:
    db = Database(str(tmp_path / "t.db"))
    db.create_tables()
    collector = Collector(
        model_client=_llm_client(),
        db=db,
        config=test_config,
    )
    return collector, db


def test_collector_name_is_singular(test_config, tmp_path):
    """One agent identity for all collections — cursors stay partitioned via
    (agent_name, memory_name) on agent_cursor."""
    collector, _ = _make_collector(test_config, tmp_path)
    assert collector.name == "collector"


def test_dispatcher_returns_none_when_no_collections_have_prompts(test_config, tmp_path):
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection("plain", "no collector wired", RecallMode.OFF)
    assert collector._next_ready_collection() is None


def test_dispatcher_picks_collection_with_extraction_prompt(test_config, tmp_path):
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "wired",
        "has a collector",
        RecallMode.OFF,
        extraction_prompt="extract things",
    )
    target = collector._next_ready_collection()
    assert target is not None
    assert target.name == "wired"


def test_dispatcher_skips_archived(test_config, tmp_path):
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "wired",
        "has a collector",
        RecallMode.OFF,
        extraction_prompt="extract",
    )
    db.memories.archive("wired")
    assert collector._next_ready_collection() is None


def test_dispatcher_skips_collections_within_interval(test_config, tmp_path):
    """A collection just collected stays out of the running until its
    interval has elapsed."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "wired",
        "has a collector",
        RecallMode.OFF,
        extraction_prompt="extract",
        collector_interval_seconds=300,
    )
    db.memories.mark_collected("wired")  # last_collected_at = now
    assert collector._next_ready_collection() is None


def test_dispatcher_picks_most_overdue(test_config, tmp_path):
    """When multiple collections are ready the oldest last_collected_at wins."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "fresh", "x", RecallMode.OFF, extraction_prompt="x", collector_interval_seconds=60
    )
    db.memories.create_collection(
        "stale", "x", RecallMode.OFF, extraction_prompt="x", collector_interval_seconds=60
    )
    # Both collected, but `stale` was much earlier
    db.memories.mark_collected("fresh")
    # Backdate `stale`'s last_collected_at by an hour
    with db.engine.connect() as conn:
        from sqlalchemy import text

        conn.execute(
            text("UPDATE memory SET last_collected_at = :ts WHERE name = 'stale'"),
            {"ts": (datetime.now(UTC) - timedelta(hours=1)).isoformat()},
        )
        conn.commit()

    target = collector._next_ready_collection()
    assert target is not None
    assert target.name == "stale"


def test_dispatcher_uses_default_interval_when_unset(test_config, tmp_path):
    """A collection with NULL collector_interval_seconds falls back to the
    PennyConstants default."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection("wired", "x", RecallMode.OFF, extraction_prompt="x")
    # Just collected → not ready until DEFAULT_INTERVAL elapses
    db.memories.mark_collected("wired")
    assert collector._next_ready_collection() is None

    # Backdate by exactly the default interval
    backdate = datetime.now(UTC) - timedelta(seconds=PennyConstants.COLLECTOR_DEFAULT_INTERVAL + 1)
    with db.engine.connect() as conn:
        from sqlalchemy import text

        conn.execute(
            text("UPDATE memory SET last_collected_at = :ts WHERE name = 'wired'"),
            {"ts": backdate.isoformat()},
        )
        conn.commit()

    assert collector._next_ready_collection() is not None


@pytest.mark.asyncio
async def test_get_tools_raises_outside_cycle(test_config, tmp_path):
    """The tool surface is per-target — accessing it without an active
    cycle is a programmer error, not a silent empty list."""
    collector, _ = _make_collector(test_config, tmp_path)
    with pytest.raises(RuntimeError, match="outside an execute"):
        collector.get_tools()


# ── Collector-runs audit log ─────────────────────────────────────────────


def _seed_collector_runs_log(db: Database) -> None:
    """Migration 0034 creates the log in production; tests using create_tables
    directly need to declare it themselves."""
    db.memories.create_log("collector-runs", "audit log", RecallMode.OFF)


def test_log_run_writes_done_summary_on_success(test_config, tmp_path):
    collector, db = _make_collector(test_config, tmp_path)
    _seed_collector_runs_log(db)
    target = Memory(
        name="prague-trip",
        type="collection",
        description="x",
        recall=RecallMode.OFF.value,
        archived=False,
        extraction_prompt="x",
    )
    response = ControllerResponse(
        answer="",
        tool_calls=[
            ToolCallRecord(tool="collection_write", arguments={}),
            ToolCallRecord(
                tool="done",
                arguments={"success": True, "summary": "wrote 2 new spots"},
            ),
        ],
    )
    collector._log_run(target, response)
    entries = db.memories.read_latest("collector-runs")
    assert len(entries) == 1
    assert "[prague-trip]" in entries[0].content
    assert "✅" in entries[0].content
    assert "wrote 2 new spots" in entries[0].content


def test_log_run_marks_failure_when_done_says_so(test_config, tmp_path):
    collector, db = _make_collector(test_config, tmp_path)
    _seed_collector_runs_log(db)
    target = Memory(
        name="prague-trip",
        type="collection",
        description="x",
        recall=RecallMode.OFF.value,
        archived=False,
        extraction_prompt="x",
    )
    response = ControllerResponse(
        answer="",
        tool_calls=[
            ToolCallRecord(
                tool="done",
                arguments={"success": False, "summary": "no source URL found"},
            ),
        ],
    )
    collector._log_run(target, response)
    content = db.memories.read_latest("collector-runs")[0].content
    assert "❌" in content
    assert "no source URL found" in content


def test_log_run_handles_no_done_call(test_config, tmp_path):
    """If the cycle hits max_steps without ever calling done(), the audit
    log still gets a row — with success=false and a sentinel summary."""
    collector, db = _make_collector(test_config, tmp_path)
    _seed_collector_runs_log(db)
    target = Memory(
        name="prague-trip",
        type="collection",
        description="x",
        recall=RecallMode.OFF.value,
        archived=False,
        extraction_prompt="x",
    )
    response = ControllerResponse(
        answer="",
        tool_calls=[
            ToolCallRecord(tool="browse", arguments={"queries": ["x"]}),
        ],
    )
    collector._log_run(target, response)
    content = db.memories.read_latest("collector-runs")[0].content
    assert "❌" in content
    assert "max steps" in content.lower() or "no done" in content.lower()

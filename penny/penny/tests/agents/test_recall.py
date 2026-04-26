"""Tests for build_recall_block — the ambient recall assembler.

Test organisation:
1. Happy paths — each recall mode renders correctly
2. Conversation pair — pair merge + primary individual pass
3. Edge/skip cases — off mode, archived, empty, no embedding
"""

from __future__ import annotations

import hashlib

import pytest

from penny.agents.recall import build_recall_block
from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory_store import EntryInput, LogEntryInput, RecallMode
from penny.llm.client import LlmClient


def _make_db(tmp_path) -> Database:
    """Empty test DB with schema only — no migrations.

    Migration 0026 seeds three system log memories; these recall tests
    declare exactly the memories they need.
    """
    db_path = str(tmp_path / "test.db")
    db = Database(db_path)
    db.create_tables()
    return db


def _hash_embed(model: str, text: str | list[str]) -> list[list[float]]:
    inputs = text if isinstance(text, list) else [text]
    return [_single_hash_vec(t) for t in inputs]


def _single_hash_vec(text: str, dim: int = 4096) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    axis = int.from_bytes(digest[:8], "big") % dim
    vec = [0.0] * dim
    vec[axis] = 1.0
    return vec


def _make_llm_client(mock_llm) -> LlmClient:
    mock_llm.set_embed_handler(_hash_embed)
    return LlmClient(
        api_url="http://localhost:11434",
        model="test-model",
        max_retries=1,
        retry_delay=0.0,
    )


def _write_entry(
    db: Database, name: str, key: str | None, content: str, author: str = "test"
) -> None:
    if key is None:
        db.memories.append(name, [LogEntryInput(content=content)], author=author)
    else:
        db.memories.write(name, [EntryInput(key=key, content=content)], author=author)


def _write_entry_embedded(
    db: Database, name: str, key: str | None, content: str, author: str = "test"
) -> None:
    """Write an entry with a deterministic content embedding for similarity tests."""
    vec = _single_hash_vec(content)
    if key is None:
        db.memories.append(
            name, [LogEntryInput(content=content, content_embedding=vec)], author=author
        )
    else:
        db.memories.write(
            name, [EntryInput(key=key, content=content, content_embedding=vec)], author=author
        )


# ── 1. Happy paths ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recent_mode_renders_latest_entries(tmp_path):
    db = _make_db(tmp_path)
    db.memories.create_log("events", "test log", RecallMode.RECENT)
    _write_entry(db, "events", None, "first")
    _write_entry(db, "events", None, "second")

    result = await build_recall_block(db, None, "anything")

    assert result is not None
    assert "### events" in result
    assert "second" in result
    assert "first" in result


@pytest.mark.asyncio
async def test_all_mode_renders_all_entries(tmp_path):
    db = _make_db(tmp_path)
    db.memories.create_collection("facts", "test facts", RecallMode.ALL)
    _write_entry(db, "facts", "a", "alpha")
    _write_entry(db, "facts", "b", "beta")

    result = await build_recall_block(db, None, None)

    assert result is not None
    assert "alpha" in result
    assert "beta" in result


@pytest.mark.asyncio
async def test_relevant_mode_uses_embedding(tmp_path, mock_llm):
    db = _make_db(tmp_path)
    db.memories.create_collection("prefs", "user prefs", RecallMode.RELEVANT)
    client = _make_llm_client(mock_llm)
    _write_entry_embedded(db, "prefs", "coffee", "loves dark roast")
    _write_entry_embedded(db, "prefs", "noise", "hates construction sounds")

    result = await build_recall_block(db, client, "dark roast coffee", similarity_floor=0.0)

    assert result is not None
    assert "loves dark roast" in result


@pytest.mark.asyncio
async def test_relevant_mode_without_client_returns_none(tmp_path):
    db = _make_db(tmp_path)
    db.memories.create_collection("prefs", "user prefs", RecallMode.RELEVANT)
    _write_entry(db, "prefs", "coffee", "loves coffee")

    result = await build_recall_block(db, None, "coffee")

    assert result is None


@pytest.mark.asyncio
async def test_relevant_mode_without_message_returns_none(tmp_path, mock_llm):
    db = _make_db(tmp_path)
    db.memories.create_collection("prefs", "user prefs", RecallMode.RELEVANT)
    _write_entry(db, "prefs", "coffee", "loves coffee")

    result = await build_recall_block(db, _make_llm_client(mock_llm), None)

    assert result is None


# ── 2. Conversation pair ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_conversation_pair_merges_chronologically(tmp_path):
    db = _make_db(tmp_path)
    primary, secondary = PennyConstants.MEMORY_CONVERSATION_PAIRS[0]
    db.memories.create_log(primary, "user messages", RecallMode.RECENT)
    db.memories.create_log(secondary, "penny messages", RecallMode.RECENT)
    _write_entry(db, primary, None, "hello", author="user")
    _write_entry(db, secondary, None, "hi there", author="penny")

    result = await build_recall_block(db, None, None)

    assert result is not None
    assert "### Conversation" in result
    assert "[user] hello" in result
    assert "[penny] hi there" in result


@pytest.mark.asyncio
async def test_pair_secondary_not_rendered_individually(tmp_path):
    db = _make_db(tmp_path)
    primary, secondary = PennyConstants.MEMORY_CONVERSATION_PAIRS[0]
    db.memories.create_log(primary, "user messages", RecallMode.RECENT)
    db.memories.create_log(secondary, "penny messages", RecallMode.RECENT)
    _write_entry(db, primary, None, "hello", author="user")
    _write_entry(db, secondary, None, "hi there", author="penny")

    result = await build_recall_block(db, None, None)

    assert result is not None
    sections = result.split("### ")
    secondary_headers = [s for s in sections if s.startswith(secondary)]
    assert not secondary_headers, "secondary log should not have its own section"


@pytest.mark.asyncio
async def test_pair_primary_also_rendered_individually(tmp_path, mock_llm):
    db = _make_db(tmp_path)
    primary, secondary = PennyConstants.MEMORY_CONVERSATION_PAIRS[0]
    db.memories.create_log(primary, "user messages", RecallMode.RELEVANT)
    db.memories.create_log(secondary, "penny messages", RecallMode.RECENT)
    client = _make_llm_client(mock_llm)
    _write_entry_embedded(db, primary, None, "I love dark roast", author="user")
    _write_entry(db, secondary, None, "sounds great!", author="penny")

    result = await build_recall_block(db, client, "dark roast coffee", similarity_floor=0.0)

    assert result is not None
    assert "### Conversation" in result
    assert f"### {primary}" in result


@pytest.mark.asyncio
async def test_pair_missing_secondary_renders_primary_normally(tmp_path, mock_llm):
    db = _make_db(tmp_path)
    primary, _ = PennyConstants.MEMORY_CONVERSATION_PAIRS[0]
    db.memories.create_log(primary, "user messages", RecallMode.RECENT)
    _write_entry(db, primary, None, "hello", author="user")

    result = await build_recall_block(db, None, None)

    assert result is not None
    assert "### Conversation" not in result
    assert f"### {primary}" in result


# ── 3. Edge / skip cases ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_off_mode_skipped(tmp_path):
    db = _make_db(tmp_path)
    db.memories.create_collection("hidden", "not shown", RecallMode.OFF)
    _write_entry(db, "hidden", "k", "content")

    result = await build_recall_block(db, None, None)

    assert result is None


@pytest.mark.asyncio
async def test_archived_memory_skipped(tmp_path):
    db = _make_db(tmp_path)
    db.memories.create_collection("old", "archived", RecallMode.RECENT)
    _write_entry(db, "old", "k", "stale content")
    db.memories.archive("old")

    result = await build_recall_block(db, None, None)

    assert result is None


@pytest.mark.asyncio
async def test_empty_database_returns_none(tmp_path):
    db = _make_db(tmp_path)
    result = await build_recall_block(db, None, None)
    assert result is None


@pytest.mark.asyncio
async def test_memory_with_no_entries_omitted(tmp_path):
    db = _make_db(tmp_path)
    db.memories.create_collection("empty", "no entries yet", RecallMode.ALL)

    result = await build_recall_block(db, None, None)

    assert result is None

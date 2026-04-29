"""Tests for the recall module — memory inventory + ambient recall.

Test organisation:
1. Happy paths — each recall mode renders correctly
2. Edge/skip cases — off mode, archived, empty, no embedding
3. Memory inventory — present for every non-archived memory
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest
from sqlmodel import Session, select

from penny.agents.recall import build_memory_inventory, build_recall_block
from penny.database import Database
from penny.database.memory_store import EntryInput, LogEntryInput, RecallMode
from penny.database.models import MemoryEntry
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
    """Bag-of-words deterministic embedding: each word picks an axis,
    vector is normalized.  Shared words across two strings give meaningful
    cosine > 0; fully-distinct strings give cosine = 0."""
    vec = [0.0] * dim
    words = text.lower().split() or [text]
    for word in words:
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        axis = int.from_bytes(digest[:8], "big") % dim
        vec[axis] += 1.0
    # L2-normalise so cosine is comparable across vectors.
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
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


def _backfill_created_at(db: Database, name: str, content: str, when: datetime) -> None:
    """Override an entry's created_at timestamp for temporal-window tests."""
    with Session(db.engine) as session:
        rows = session.exec(
            select(MemoryEntry).where(
                MemoryEntry.memory_name == name,
                MemoryEntry.content == content,
            )
        ).all()
        for row in rows:
            row.created_at = when
            session.add(row)
        session.commit()


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
async def test_relevant_mode_hybrid_lifts_entry_via_conversation_history(tmp_path, mock_llm):
    """A vague current message ('yeah') alone wouldn't surface the entry —
    the prior turn shares the entry's keywords, so weighted-decay scoring
    pulls the entry above the absolute floor."""
    db = _make_db(tmp_path)
    db.memories.create_collection("prefs", "user prefs", RecallMode.RELEVANT)
    client = _make_llm_client(mock_llm)
    _write_entry_embedded(db, "prefs", "coffee", "loves dark roast coffee")

    result = await build_recall_block(
        db,
        client,
        "yeah",
        conversation_history=["dark roast coffee"],
    )

    assert result is not None
    assert "loves dark roast coffee" in result


@pytest.mark.asyncio
async def test_relevant_mode_log_expands_with_temporal_neighbors(tmp_path, mock_llm):
    """Log-shaped relevant memory expands similarity hits with surrounding
    entries from the same log (within the temporal window), so a single
    keyword match pulls in the rest of its conversation."""
    db = _make_db(tmp_path)
    db.memories.create_log("conversation", "shared chat log", RecallMode.RELEVANT)
    client = _make_llm_client(mock_llm)

    # The topic entry shares words with the anchor; the neighbors share
    # nothing (lexical zero), so they only surface via temporal expansion.
    _write_entry_embedded(db, "conversation", None, "dark roast coffee notes")
    _write_entry_embedded(db, "conversation", None, "follow up question one")
    _write_entry_embedded(db, "conversation", None, "follow up question two")
    _write_entry_embedded(db, "conversation", None, "stale earlier comment")

    base = datetime.now(UTC)
    _backfill_created_at(db, "conversation", "stale earlier comment", base - timedelta(hours=1))
    _backfill_created_at(db, "conversation", "dark roast coffee notes", base - timedelta(minutes=2))
    _backfill_created_at(db, "conversation", "follow up question one", base - timedelta(minutes=1))
    _backfill_created_at(db, "conversation", "follow up question two", base)

    result = await build_recall_block(db, client, "dark roast coffee")

    assert result is not None
    assert "dark roast coffee notes" in result
    assert "follow up question one" in result
    assert "follow up question two" in result
    assert "stale earlier comment" not in result


@pytest.mark.asyncio
async def test_relevant_mode_collection_skips_temporal_expansion(tmp_path, mock_llm):
    """Collections don't have a temporal-stream meaning, so similarity hits
    are returned without neighbor expansion even if entries are nearby in time."""
    db = _make_db(tmp_path)
    db.memories.create_collection("prefs", "user prefs", RecallMode.RELEVANT)
    client = _make_llm_client(mock_llm)
    _write_entry_embedded(db, "prefs", "coffee", "loves dark roast coffee")
    _write_entry_embedded(db, "prefs", "noise", "hates loud construction")

    result = await build_recall_block(db, client, "dark roast coffee")

    assert result is not None
    assert "loves dark roast coffee" in result
    assert "hates loud construction" not in result


@pytest.mark.asyncio
async def test_relevant_mode_without_history_skips_vague_current_message(tmp_path, mock_llm):
    """Vague current message with no history → absolute floor suppresses everything."""
    db = _make_db(tmp_path)
    db.memories.create_collection("prefs", "user prefs", RecallMode.RELEVANT)
    client = _make_llm_client(mock_llm)
    _write_entry_embedded(db, "prefs", "coffee", "loves dark roast coffee")

    result = await build_recall_block(db, client, "yeah")

    assert result is None  # nothing scores above the floor → no recall content


@pytest.mark.asyncio
async def test_relevant_mode_without_client_returns_none(tmp_path):
    db = _make_db(tmp_path)
    db.memories.create_collection("prefs", "user prefs", RecallMode.RELEVANT)
    _write_entry(db, "prefs", "coffee", "loves coffee")

    result = await build_recall_block(db, None, "coffee")

    assert result is None  # similarity unavailable → no recall content


@pytest.mark.asyncio
async def test_relevant_mode_without_message_returns_none(tmp_path, mock_llm):
    db = _make_db(tmp_path)
    db.memories.create_collection("prefs", "user prefs", RecallMode.RELEVANT)
    _write_entry(db, "prefs", "coffee", "loves coffee")

    result = await build_recall_block(db, _make_llm_client(mock_llm), None)

    assert result is None  # no anchor → no recall content


# ── 2. Edge / skip cases ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_off_mode_skipped(tmp_path):
    """off-mode memories never contribute recall content."""
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
    """A memory with no entries contributes nothing to recall content."""
    db = _make_db(tmp_path)
    db.memories.create_collection("empty", "no entries yet", RecallMode.ALL)

    result = await build_recall_block(db, None, None)

    assert result is None


# ── 3. Memory inventory ───────────────────────────────────────────────────


def test_inventory_lists_all_non_archived_memories(tmp_path):
    """Inventory names every non-archived memory regardless of recall mode."""
    db = _make_db(tmp_path)
    db.memories.create_collection("likes", "positive prefs", RecallMode.RELEVANT)
    db.memories.create_collection("dislikes", "negative prefs", RecallMode.OFF)
    db.memories.create_log("messages", "convo log", RecallMode.RECENT)

    result = build_memory_inventory(db)

    assert result is not None
    assert "### Memory Inventory" in result
    assert "likes (collection) — positive prefs" in result
    assert "dislikes (collection) — negative prefs" in result  # off-mode still listed
    assert "messages (log) — convo log" in result


def test_inventory_excludes_archived(tmp_path):
    db = _make_db(tmp_path)
    db.memories.create_collection("active", "live", RecallMode.RELEVANT)
    db.memories.create_collection("retired", "archived", RecallMode.RELEVANT)
    db.memories.archive("retired")

    result = build_memory_inventory(db)

    assert result is not None
    assert "active (collection)" in result
    assert "retired" not in result


def test_inventory_returns_none_when_no_memories(tmp_path):
    db = _make_db(tmp_path)
    assert build_memory_inventory(db) is None

"""Tests for memory tools.

Each tool is exercised through its ``execute`` coroutine end-to-end against a
real Database. The embedding path uses the existing ``mock_llm`` fixture so
similarity reads and dedup have something to work with.
"""

from __future__ import annotations

import hashlib

import pytest

from penny.database import Database
from penny.database.migrate import migrate
from penny.llm.client import LlmClient
from penny.tools.memory_context import current_agent, set_current_agent
from penny.tools.memory_tools import (
    CollectionArchiveTool,
    CollectionCreateTool,
    CollectionGetTool,
    CollectionKeysTool,
    CollectionMoveTool,
    CollectionReadAllTool,
    CollectionReadLatestTool,
    CollectionReadRandomTool,
    CollectionReadSimilarTool,
    CollectionUnarchiveTool,
    CollectionUpdateTool,
    CollectionWriteTool,
    DoneTool,
    ExistsTool,
    ListMemoriesTool,
    LogAppendTool,
    LogCreateTool,
    LogReadAllTool,
    LogReadLatestTool,
    LogReadRecentTool,
    LogReadSimilarTool,
    build_memory_tools,
)


def _make_db(tmp_path) -> Database:
    db_path = str(tmp_path / "test.db")
    db = Database(db_path)
    db.create_tables()
    migrate(db_path)
    return db


def _make_llm_client(mock_llm) -> LlmClient:
    """Build an LlmClient whose default embed handler returns distinct vectors
    per input text, so identical inputs collide and distinct inputs don't."""
    mock_llm.set_embed_handler(_hash_embed)
    return LlmClient(
        api_url="http://localhost:11434",
        model="test-model",
        max_retries=1,
        retry_delay=0.0,
    )


def _hash_embed(model: str, text: str | list[str]) -> list[list[float]]:
    """Deterministic embedding: text → unit vector where one axis is 1.0.

    Identical strings map to identical vectors; distinct strings map to
    different axes (cosine = 0), so dedup and similarity behave sensibly in
    tests without depending on a real embedding model.
    """
    inputs = text if isinstance(text, list) else [text]
    return [_single_hash_vec(t) for t in inputs]


def _single_hash_vec(text: str, dim: int = 4096) -> list[float]:
    """Deterministic one-hot vector. SHA-256 (process-stable, not salted like
    Python's built-in hash) → modulo dim picks an axis. Dim 4096 keeps accidental
    collisions between distinct short strings vanishingly rare in tests."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    axis = int.from_bytes(digest[:8], "big") % dim
    vec = [0.0] * dim
    vec[axis] = 1.0
    return vec


class TestCreateAndList:
    @pytest.mark.asyncio
    async def test_create_collection_then_list(self, tmp_path):
        db = _make_db(tmp_path)
        result = await CollectionCreateTool(db).execute(
            name="likes", description="positive prefs", recall="relevant"
        )
        assert "Created collection 'likes'" in result
        listed = await ListMemoriesTool(db).execute()
        assert "likes (collection, recall=relevant)" in listed
        assert "positive prefs" in listed

    @pytest.mark.asyncio
    async def test_create_log_then_list(self, tmp_path):
        db = _make_db(tmp_path)
        await LogCreateTool(db).execute(
            name="user-messages", description="inbound", recall="recent"
        )
        listed = await ListMemoriesTool(db).execute()
        assert "user-messages (log, recall=recent)" in listed

    @pytest.mark.asyncio
    async def test_list_empty_returns_sentinel(self, tmp_path):
        db = _make_db(tmp_path)
        assert await ListMemoriesTool(db).execute() == "(no memories)"


class TestCollectionWritesAndReads:
    @pytest.mark.asyncio
    async def test_write_read_roundtrip(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="likes", description="x", recall="relevant")
        write = CollectionWriteTool(db, _make_llm_client(mock_llm))
        result = await write.execute(
            memory="likes",
            entries=[
                {"key": "dark roast", "content": "loves dark roast"},
                {"key": "cold brew", "content": "enjoys cold brew"},
            ],
        )
        assert "Wrote 2 entries to 'likes'" in result
        latest = await CollectionReadLatestTool(db).execute(memory="likes")
        assert "dark roast" in latest
        assert "cold brew" in latest

    @pytest.mark.asyncio
    async def test_write_reports_duplicate_via_tcr(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="likes", description="x", recall="off")
        write = CollectionWriteTool(db, _make_llm_client(mock_llm))
        await write.execute(
            memory="likes", entries=[{"key": "dark roast", "content": "first body"}]
        )
        result = await write.execute(
            memory="likes",
            entries=[{"key": "dark roast coffee", "content": "different body entirely"}],
        )
        assert "Rejected as duplicates" in result
        assert "dark roast coffee" in result

    @pytest.mark.asyncio
    async def test_get_returns_entry_or_not_found(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="likes", description="x", recall="off")
        await CollectionWriteTool(db, _make_llm_client(mock_llm)).execute(
            memory="likes", entries=[{"key": "k", "content": "hello"}]
        )
        assert "hello" in await CollectionGetTool(db).execute(memory="likes", key="k")
        missing = await CollectionGetTool(db).execute(memory="likes", key="absent")
        assert "not found" in missing

    @pytest.mark.asyncio
    async def test_keys_lists_unique_keys_in_order(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="likes", description="x", recall="off")
        write = CollectionWriteTool(db, _make_llm_client(mock_llm))
        await write.execute(memory="likes", entries=[{"key": "first", "content": "1"}])
        await write.execute(memory="likes", entries=[{"key": "second", "content": "2"}])
        listing = await CollectionKeysTool(db).execute(memory="likes")
        assert listing == "- first\n- second"

    @pytest.mark.asyncio
    async def test_read_random_returns_all_when_few(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="likes", description="x", recall="off")
        write = CollectionWriteTool(db, _make_llm_client(mock_llm))
        await write.execute(memory="likes", entries=[{"key": "a", "content": "1"}])
        rendered = await CollectionReadRandomTool(db).execute(memory="likes", k=5)
        assert "[a] 1" in rendered

    @pytest.mark.asyncio
    async def test_read_similar_uses_embedding(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="likes", description="x", recall="off")
        client = _make_llm_client(mock_llm)
        await CollectionWriteTool(db, client).execute(
            memory="likes", entries=[{"key": "coffee", "content": "loves coffee"}]
        )
        rendered = await CollectionReadSimilarTool(db, client).execute(
            memory="likes", anchor="caffeine"
        )
        assert "coffee" in rendered

    @pytest.mark.asyncio
    async def test_read_similar_without_llm_client_returns_sentinel(self, tmp_path):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="likes", description="x", recall="off")
        result = await CollectionReadSimilarTool(db, None).execute(
            memory="likes", anchor="whatever"
        )
        assert "similarity search unavailable" in result

    @pytest.mark.asyncio
    async def test_read_all_empty_sentinel(self, tmp_path):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="likes", description="x", recall="off")
        assert await CollectionReadAllTool(db).execute(memory="likes") == "(no entries)"


class TestCollectionMutations:
    @pytest.mark.asyncio
    async def test_update_replaces_content(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="likes", description="x", recall="off")
        await CollectionWriteTool(db, _make_llm_client(mock_llm)).execute(
            memory="likes", entries=[{"key": "k", "content": "old"}]
        )
        result = await CollectionUpdateTool(db).execute(memory="likes", key="k", content="new")
        assert "Updated 'k' in 'likes'" in result
        fetched = await CollectionGetTool(db).execute(memory="likes", key="k")
        assert "new" in fetched

    @pytest.mark.asyncio
    async def test_update_missing_reports_not_found(self, tmp_path):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="likes", description="x", recall="off")
        result = await CollectionUpdateTool(db).execute(memory="likes", key="k", content="new")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_move_between_collections(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="unnotified", description="x", recall="off")
        await CollectionCreateTool(db).execute(name="notified", description="x", recall="off")
        await CollectionWriteTool(db, _make_llm_client(mock_llm)).execute(
            memory="unnotified", entries=[{"key": "t1", "content": "x"}]
        )
        result = await CollectionMoveTool(db).execute(
            key="t1", from_memory="unnotified", to_memory="notified"
        )
        assert "Moved 't1'" in result

    @pytest.mark.asyncio
    async def test_move_collision(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="a", description="x", recall="off")
        await CollectionCreateTool(db).execute(name="b", description="x", recall="off")
        write = CollectionWriteTool(db, _make_llm_client(mock_llm))
        await write.execute(memory="a", entries=[{"key": "k", "content": "src"}])
        await write.execute(memory="b", entries=[{"key": "k", "content": "dst"}])
        result = await CollectionMoveTool(db).execute(key="k", from_memory="a", to_memory="b")
        assert "already has a 'k' entry" in result

    @pytest.mark.asyncio
    async def test_archive_and_unarchive(self, tmp_path):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="likes", description="x", recall="off")
        assert "Archived 'likes'" in await CollectionArchiveTool(db).execute(memory="likes")
        assert "Unarchived 'likes'" in await CollectionUnarchiveTool(db).execute(memory="likes")


class TestLogTools:
    @pytest.mark.asyncio
    async def test_append_and_read_latest(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await LogCreateTool(db).execute(name="events", description="x", recall="recent")
        append = LogAppendTool(db, _make_llm_client(mock_llm))
        await append.execute(memory="events", content="first")
        await append.execute(memory="events", content="second")
        rendered = await LogReadLatestTool(db).execute(memory="events")
        assert rendered.splitlines() == ["- second", "- first"]

    @pytest.mark.asyncio
    async def test_read_all_returns_oldest_first(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await LogCreateTool(db).execute(name="events", description="x", recall="recent")
        append = LogAppendTool(db, _make_llm_client(mock_llm))
        await append.execute(memory="events", content="first")
        await append.execute(memory="events", content="second")
        rendered = await LogReadAllTool(db).execute(memory="events")
        assert rendered.splitlines() == ["- first", "- second"]

    @pytest.mark.asyncio
    async def test_read_recent_window(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await LogCreateTool(db).execute(name="events", description="x", recall="recent")
        await LogAppendTool(db, _make_llm_client(mock_llm)).execute(
            memory="events", content="hello"
        )
        rendered = await LogReadRecentTool(db).execute(memory="events", window_seconds=3600)
        assert "hello" in rendered

    @pytest.mark.asyncio
    async def test_log_similar_with_client(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await LogCreateTool(db).execute(name="events", description="x", recall="relevant")
        client = _make_llm_client(mock_llm)
        await LogAppendTool(db, client).execute(memory="events", content="coffee is great")
        rendered = await LogReadSimilarTool(db, client).execute(memory="events", anchor="beverage")
        assert "coffee is great" in rendered


class TestExistsAndDone:
    @pytest.mark.asyncio
    async def test_exists_yes_via_exact_key(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="likes", description="x", recall="off")
        client = _make_llm_client(mock_llm)
        await CollectionWriteTool(db, client).execute(
            memory="likes", entries=[{"key": "dark roast", "content": "body"}]
        )
        result = await ExistsTool(db, client).execute(
            memories=["likes"], key="dark roast", content="body"
        )
        assert result == "yes"

    @pytest.mark.asyncio
    async def test_exists_no(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="likes", description="x", recall="off")
        result = await ExistsTool(db, _make_llm_client(mock_llm)).execute(
            memories=["likes"], key="not there", content="nothing"
        )
        assert result == "no"

    @pytest.mark.asyncio
    async def test_done_returns_done(self):
        assert await DoneTool().execute() == "done"


class TestAuthorAttribution:
    @pytest.mark.asyncio
    async def test_writes_use_current_agent(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="likes", description="x", recall="off")
        set_current_agent("preference-extractor")
        try:
            await CollectionWriteTool(db, _make_llm_client(mock_llm)).execute(
                memory="likes", entries=[{"key": "k", "content": "v"}]
            )
        finally:
            set_current_agent("unknown")

        rows = db.memories.get_entry("likes", "k")
        assert rows[0].author == "preference-extractor"

    def test_default_agent_is_unknown(self):
        # The preceding test restores the default in its finally block, so
        # whether this runs first or after, current_agent() must equal the
        # module-level default.
        assert current_agent() == "unknown"


class TestFactory:
    def test_build_memory_tools_registers_every_tool(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        tools = build_memory_tools(db, _make_llm_client(mock_llm))
        names = {tool.name for tool in tools}
        expected = {
            "collection_create",
            "collection_get",
            "collection_read_latest",
            "collection_read_random",
            "collection_read_similar",
            "collection_read_all",
            "collection_keys",
            "collection_write",
            "collection_update",
            "collection_move",
            "collection_archive",
            "collection_unarchive",
            "log_create",
            "log_read_latest",
            "log_read_recent",
            "log_read_similar",
            "log_read_all",
            "log_append",
            "list_memories",
            "exists",
            "done",
        }
        assert names == expected

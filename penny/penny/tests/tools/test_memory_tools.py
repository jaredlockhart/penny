"""Tests for memory tools.

Each tool is exercised through its ``execute`` coroutine end-to-end against a
real Database. The embedding path uses the existing ``mock_llm`` fixture so
similarity reads and dedup have something to work with.
"""

from __future__ import annotations

import hashlib

import pytest

from penny.database import Database
from penny.llm.client import LlmClient
from penny.tools.memory_tools import (
    CollectionArchiveTool,
    CollectionCreateTool,
    CollectionDeleteEntryTool,
    CollectionGetTool,
    CollectionKeysTool,
    CollectionMoveTool,
    CollectionReadRandomTool,
    CollectionUnarchiveTool,
    CollectionUpdateTool,
    CollectionWriteTool,
    DoneTool,
    ExistsTool,
    LogAppendTool,
    LogCreateTool,
    LogReadNextTool,
    LogReadRecentTool,
    ReadLatestTool,
    ReadSimilarTool,
    build_memory_tools,
)


def _make_db(tmp_path) -> Database:
    """Empty test DB with schema only — no migrations.

    Migration 0026 seeds three system log memories; these tool tests
    exercise the tool surface in isolation and declare exactly the
    memories they need.
    """
    db_path = str(tmp_path / "test.db")
    db = Database(db_path)
    db.create_tables()
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
    """Bag-of-words deterministic embedding.  Each word picks an axis via
    SHA-256 → modulo ``dim``; the vector is L2-normalised so cosine is
    comparable across strings.  Identical strings map to identical
    vectors; strings sharing words have meaningful cosine > 0;
    fully-distinct strings map to cosine = 0."""
    vec = [0.0] * dim
    words = text.lower().split() or [text]
    for word in words:
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        axis = int.from_bytes(digest[:8], "big") % dim
        vec[axis] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


class TestCreateAndList:
    @pytest.mark.asyncio
    async def test_create_collection_persists(self, tmp_path):
        db = _make_db(tmp_path)
        result = await CollectionCreateTool(db).execute(
            name="likes", description="positive prefs", recall="relevant"
        )
        assert "Created collection 'likes'" in result
        memories = {m.name: m for m in db.memories.list_all()}
        assert memories["likes"].type == "collection"
        assert memories["likes"].recall == "relevant"
        assert memories["likes"].description == "positive prefs"

    @pytest.mark.asyncio
    async def test_create_log_persists(self, tmp_path):
        db = _make_db(tmp_path)
        await LogCreateTool(db).execute(
            name="user-messages", description="inbound", recall="recent"
        )
        memories = {m.name: m for m in db.memories.list_all()}
        assert memories["user-messages"].type == "log"
        assert memories["user-messages"].recall == "recent"


class TestCollectionWritesAndReads:
    @pytest.mark.asyncio
    async def test_write_read_roundtrip(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="likes", description="x", recall="relevant")
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        result = await write.execute(
            memory="likes",
            entries=[
                {"key": "dark roast", "content": "loves dark roast"},
                {"key": "cold brew", "content": "enjoys cold brew"},
            ],
        )
        assert "Wrote 2 entries to 'likes'" in result
        latest = await ReadLatestTool(db).execute(memory="likes")
        assert "dark roast" in latest
        assert "cold brew" in latest

    @pytest.mark.asyncio
    async def test_write_reports_duplicate_via_tcr(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="likes", description="x", recall="off")
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
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
        await CollectionWriteTool(db, _make_llm_client(mock_llm), author="test").execute(
            memory="likes", entries=[{"key": "k", "content": "hello"}]
        )
        assert "hello" in await CollectionGetTool(db).execute(memory="likes", key="k")
        missing = await CollectionGetTool(db).execute(memory="likes", key="absent")
        assert "not found" in missing

    @pytest.mark.asyncio
    async def test_keys_lists_unique_keys_in_order(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="likes", description="x", recall="off")
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        await write.execute(memory="likes", entries=[{"key": "first", "content": "1"}])
        await write.execute(memory="likes", entries=[{"key": "second", "content": "2"}])
        listing = await CollectionKeysTool(db).execute(memory="likes")
        assert listing == "- first\n- second"

    @pytest.mark.asyncio
    async def test_read_random_returns_all_when_few(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="likes", description="x", recall="off")
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        await write.execute(memory="likes", entries=[{"key": "a", "content": "1"}])
        rendered = await CollectionReadRandomTool(db).execute(memory="likes", k=5)
        assert "[a] 1" in rendered

    @pytest.mark.asyncio
    async def test_read_similar_uses_embedding(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="likes", description="x", recall="off")
        client = _make_llm_client(mock_llm)
        await CollectionWriteTool(db, client, author="test").execute(
            memory="likes", entries=[{"key": "coffee", "content": "loves coffee"}]
        )
        # Anchor shares the "coffee" word with the entry — the bag-of-words
        # mock embedding gives meaningful cosine, so the entry survives the
        # adaptive cutoff in ``read_similar``.
        rendered = await ReadSimilarTool(db, client).execute(memory="likes", anchor="coffee please")
        assert "coffee" in rendered

    @pytest.mark.asyncio
    async def test_read_similar_without_llm_client_returns_sentinel(self, tmp_path):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="likes", description="x", recall="off")
        result = await ReadSimilarTool(db, None).execute(memory="likes", anchor="whatever")
        assert "similarity search unavailable" in result


class TestCollectionMutations:
    @pytest.mark.asyncio
    async def test_update_replaces_content(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="likes", description="x", recall="off")
        await CollectionWriteTool(db, _make_llm_client(mock_llm), author="test").execute(
            memory="likes", entries=[{"key": "k", "content": "old"}]
        )
        result = await CollectionUpdateTool(db, author="test").execute(
            memory="likes", key="k", content="new"
        )
        assert "Updated 'k' in 'likes'" in result
        fetched = await CollectionGetTool(db).execute(memory="likes", key="k")
        assert "new" in fetched

    @pytest.mark.asyncio
    async def test_update_missing_reports_not_found(self, tmp_path):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="likes", description="x", recall="off")
        result = await CollectionUpdateTool(db, author="test").execute(
            memory="likes", key="k", content="new"
        )
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_move_between_collections(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="unnotified", description="x", recall="off")
        await CollectionCreateTool(db).execute(name="notified", description="x", recall="off")
        await CollectionWriteTool(db, _make_llm_client(mock_llm), author="test").execute(
            memory="unnotified", entries=[{"key": "t1", "content": "x"}]
        )
        result = await CollectionMoveTool(db, author="test").execute(
            key="t1", from_memory="unnotified", to_memory="notified"
        )
        assert "Moved 't1'" in result

    @pytest.mark.asyncio
    async def test_move_collision(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="a", description="x", recall="off")
        await CollectionCreateTool(db).execute(name="b", description="x", recall="off")
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        await write.execute(memory="a", entries=[{"key": "k", "content": "src"}])
        await write.execute(memory="b", entries=[{"key": "k", "content": "dst"}])
        result = await CollectionMoveTool(db, author="test").execute(
            key="k", from_memory="a", to_memory="b"
        )
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
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        await append.execute(memory="events", content="first")
        await append.execute(memory="events", content="second")
        rendered = await ReadLatestTool(db).execute(memory="events")
        assert rendered.splitlines() == ["- second", "- first"]

    @pytest.mark.asyncio
    async def test_read_recent_window(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await LogCreateTool(db).execute(name="events", description="x", recall="recent")
        await LogAppendTool(db, _make_llm_client(mock_llm), author="test").execute(
            memory="events", content="hello"
        )
        rendered = await LogReadRecentTool(db).execute(memory="events", window_seconds=3600)
        assert "hello" in rendered

    @pytest.mark.asyncio
    async def test_log_similar_with_client(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await LogCreateTool(db).execute(name="events", description="x", recall="relevant")
        client = _make_llm_client(mock_llm)
        await LogAppendTool(db, client, author="test").execute(
            memory="events", content="coffee is great"
        )
        # Anchor shares words with the entry so the bag-of-words mock
        # embedding gives meaningful cosine and the entry survives the
        # adaptive cutoff in ``read_similar``.
        rendered = await ReadSimilarTool(db, client).execute(
            memory="events", anchor="coffee morning"
        )
        assert "coffee is great" in rendered

    @pytest.mark.asyncio
    async def test_read_next_returns_all_entries_when_no_cursor(self, tmp_path, mock_llm):
        """Without a stored cursor, read_next returns every entry in the log."""
        db = _make_db(tmp_path)
        await LogCreateTool(db).execute(name="events", description="x", recall="recent")
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        await append.execute(memory="events", content="first")
        await append.execute(memory="events", content="second")

        read_next = LogReadNextTool(db, agent_name="extractor")
        rendered = await read_next.execute(memory="events")

        assert "first" in rendered
        assert "second" in rendered

    @pytest.mark.asyncio
    async def test_commit_pending_advances_cursor_to_max_seen(self, tmp_path, mock_llm):
        """commit_pending writes the highest timestamp seen during the run."""
        db = _make_db(tmp_path)
        await LogCreateTool(db).execute(name="events", description="x", recall="recent")
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        await append.execute(memory="events", content="first")
        await append.execute(memory="events", content="second")

        read_next = LogReadNextTool(db, agent_name="extractor")
        await read_next.execute(memory="events")
        read_next.commit_pending()

        # A new instance after commit should see no entries (cursor caught up).
        fresh = LogReadNextTool(db, agent_name="extractor")
        rendered = await fresh.execute(memory="events")
        assert rendered == "(no entries)"

    @pytest.mark.asyncio
    async def test_discard_pending_leaves_cursor_unchanged(self, tmp_path, mock_llm):
        """discard_pending drops the in-memory state without touching the DB cursor."""
        db = _make_db(tmp_path)
        await LogCreateTool(db).execute(name="events", description="x", recall="recent")
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        await append.execute(memory="events", content="first")

        read_next = LogReadNextTool(db, agent_name="extractor")
        await read_next.execute(memory="events")
        read_next.discard_pending()

        # Cursor still at None; a new read sees the same entries.
        fresh = LogReadNextTool(db, agent_name="extractor")
        rendered = await fresh.execute(memory="events")
        assert "first" in rendered

    @pytest.mark.asyncio
    async def test_per_agent_cursors_are_independent(self, tmp_path, mock_llm):
        """Two agents reading the same log have independent cursor state."""
        db = _make_db(tmp_path)
        await LogCreateTool(db).execute(name="events", description="x", recall="recent")
        await LogAppendTool(db, _make_llm_client(mock_llm), author="test").execute(
            memory="events", content="hello"
        )

        agent_a = LogReadNextTool(db, agent_name="a")
        await agent_a.execute(memory="events")
        agent_a.commit_pending()

        # Agent B has its own cursor and still sees the entry.
        agent_b = LogReadNextTool(db, agent_name="b")
        rendered = await agent_b.execute(memory="events")
        assert "hello" in rendered


class TestExistsAndDone:
    @pytest.mark.asyncio
    async def test_exists_yes_via_exact_key(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="likes", description="x", recall="off")
        client = _make_llm_client(mock_llm)
        await CollectionWriteTool(db, client, author="test").execute(
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
    async def test_writes_stamp_constructor_author(self, tmp_path, mock_llm):
        """Author is bound at tool construction (not pulled from ambient state)."""
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="likes", description="x", recall="off")
        await CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="preference-extractor"
        ).execute(memory="likes", entries=[{"key": "k", "content": "v"}])

        rows = db.memories.get_entry("likes", "k")
        assert rows[0].author == "preference-extractor"


class TestFactory:
    def test_build_memory_tools_registers_every_tool(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        tools = build_memory_tools(db, _make_llm_client(mock_llm), agent_name="test")
        names = {tool.name for tool in tools}
        expected = {
            "collection_create",
            "collection_get",
            "collection_read_random",
            "collection_keys",
            "collection_write",
            "collection_update",
            "collection_move",
            "collection_delete_entry",
            "collection_archive",
            "collection_unarchive",
            "log_create",
            "log_read_recent",
            "log_read_next",
            "log_append",
            "read_latest",
            "read_similar",
            "exists",
        }
        assert names == expected


class TestScopedFactory:
    """When ``scope`` is set the surface narrows to a collector's needs:
    metadata creation/archive, ``collection_move`` (multi-collection), and
    ``log_append`` (logs are inputs, not outputs) are excluded entirely.
    """

    def test_scoped_factory_drops_metadata_and_unscoped_writes(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        tools = build_memory_tools(
            db, _make_llm_client(mock_llm), agent_name="collector:likes", scope="likes"
        )
        names = {tool.name for tool in tools}
        # Reads + the three scoped writes — nothing else
        assert names == {
            "collection_get",
            "collection_read_random",
            "collection_keys",
            "log_read_recent",
            "log_read_next",
            "read_latest",
            "read_similar",
            "exists",
            "collection_write",
            "collection_update",
            "collection_delete_entry",
        }

    @pytest.mark.asyncio
    async def test_scoped_write_rejects_other_collection(self, tmp_path, mock_llm):
        """A scoped collector that tries to write to a different collection
        gets a clean refusal rather than silently corrupting unrelated data."""
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="likes", description="x", recall="off")
        await CollectionCreateTool(db).execute(name="dislikes", description="x", recall="off")

        write = CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="collector:likes", scope="likes"
        )
        result = await write.execute(memory="dislikes", entries=[{"key": "k", "content": "v"}])

        assert "Refused" in result and "likes" in result and "dislikes" in result
        # And nothing was actually written
        assert db.memories.get_entry("dislikes", "k") == []

    @pytest.mark.asyncio
    async def test_scoped_write_allows_target_collection(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db).execute(name="likes", description="x", recall="off")

        write = CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="collector:likes", scope="likes"
        )
        result = await write.execute(memory="likes", entries=[{"key": "k", "content": "v"}])

        assert "Wrote 1 entry" in result
        assert db.memories.get_entry("likes", "k")[0].content == "v"

    @pytest.mark.asyncio
    async def test_scoped_update_rejects_other_collection(self, tmp_path):
        db = _make_db(tmp_path)
        update = CollectionUpdateTool(db, author="collector:likes", scope="likes")
        result = await update.execute(memory="dislikes", key="k", content="v")
        assert "Refused" in result

    @pytest.mark.asyncio
    async def test_scoped_delete_rejects_other_collection(self, tmp_path):
        db = _make_db(tmp_path)
        delete = CollectionDeleteEntryTool(db, scope="likes")
        result = await delete.execute(memory="dislikes", key="k")
        assert "Refused" in result

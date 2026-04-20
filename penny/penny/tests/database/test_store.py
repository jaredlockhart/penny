"""Tests for StoreStore, CursorStore, and MediaStore.

Exercises the data layer for the task/collection framework. Dedup, type
enforcement, log append, cursor monotonicity, and the similarity-based
`exists` check all run through these tests.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from penny.database import Database
from penny.database.migrate import migrate
from penny.database.store import (
    DedupThresholds,
    EntryInput,
    LogEntryInput,
    RecallMode,
    StoreNotFoundError,
    StoreTypeError,
)


def _make_db(tmp_path) -> Database:
    db_path = str(tmp_path / "test.db")
    db = Database(db_path)
    db.create_tables()
    migrate(db_path)
    return db


def _unit_vec(idx: int, dim: int = 8) -> list[float]:
    """Return a sparse unit vector with a single 1.0 at position idx."""
    vec = [0.0] * dim
    vec[idx % dim] = 1.0
    return vec


class TestStoreMetadata:
    def test_create_collection_and_fetch(self, tmp_path):
        db = _make_db(tmp_path)
        store = db.stores.create_collection(
            "likes", "user positive preferences", RecallMode.RELEVANT
        )
        assert store.name == "likes"
        assert store.type == "collection"
        assert store.recall == "relevant"
        assert store.archived is False

        fetched = db.stores.get("likes")
        assert fetched is not None
        assert fetched.description == "user positive preferences"

    def test_create_log_and_list(self, tmp_path):
        db = _make_db(tmp_path)
        db.stores.create_log("user-messages", "inbound user messages", RecallMode.RELEVANT)
        db.stores.create_collection("dislikes", "user negative preferences", RecallMode.RELEVANT)

        names = [s.name for s in db.stores.list_all()]
        assert names == ["dislikes", "user-messages"]

    def test_archive_and_unarchive(self, tmp_path):
        db = _make_db(tmp_path)
        db.stores.create_collection("notes", "scratch", RecallMode.OFF)
        db.stores.archive("notes")
        assert db.stores.get("notes").archived is True
        db.stores.unarchive("notes")
        assert db.stores.get("notes").archived is False

    def test_archive_missing_raises(self, tmp_path):
        db = _make_db(tmp_path)
        with pytest.raises(StoreNotFoundError):
            db.stores.archive("nope")


class TestCollectionWrites:
    def test_write_returns_entry_ids(self, tmp_path):
        db = _make_db(tmp_path)
        db.stores.create_collection("likes", "positive prefs", RecallMode.RELEVANT)

        results = db.stores.write(
            "likes",
            [
                EntryInput(
                    key="dark roast coffee",
                    content="I love dark roast coffee",
                    key_embedding=_unit_vec(0),
                    content_embedding=_unit_vec(1),
                ),
                EntryInput(
                    key="cold brew",
                    content="cold brew is great",
                    key_embedding=_unit_vec(2),
                    content_embedding=_unit_vec(3),
                ),
            ],
            author="preference-extractor",
        )
        assert [r.outcome for r in results] == ["written", "written"]
        assert all(r.entry_id is not None for r in results)
        assert {r.key for r in results} == {"dark roast coffee", "cold brew"}

    def test_write_dedups_on_key_embedding(self, tmp_path):
        db = _make_db(tmp_path)
        db.stores.create_collection("likes", "positive prefs", RecallMode.RELEVANT)
        shared_key_vec = _unit_vec(0)

        db.stores.write(
            "likes",
            [
                EntryInput(
                    key="dark roast",
                    content="dark roast",
                    key_embedding=shared_key_vec,
                    content_embedding=_unit_vec(1),
                )
            ],
            author="preference-extractor",
        )
        results = db.stores.write(
            "likes",
            [
                EntryInput(
                    key="dark roast coffee",
                    content="totally different body",
                    key_embedding=shared_key_vec,
                    content_embedding=_unit_vec(5),
                )
            ],
            author="preference-extractor",
        )
        assert results[0].outcome == "duplicate"
        assert results[0].entry_id is None
        assert len(db.stores.read_all("likes")) == 1

    def test_write_dedups_on_content_embedding(self, tmp_path):
        db = _make_db(tmp_path)
        db.stores.create_collection("likes", "positive prefs", RecallMode.RELEVANT)
        shared_content = _unit_vec(4)

        db.stores.write(
            "likes",
            [
                EntryInput(
                    key="first key",
                    content="same body",
                    key_embedding=_unit_vec(0),
                    content_embedding=shared_content,
                )
            ],
            author="preference-extractor",
        )
        results = db.stores.write(
            "likes",
            [
                EntryInput(
                    key="different key entirely",
                    content="same body",
                    key_embedding=_unit_vec(7),
                    content_embedding=shared_content,
                )
            ],
            author="preference-extractor",
        )
        assert results[0].outcome == "duplicate"

    def test_write_without_embeddings_always_accepts(self, tmp_path):
        db = _make_db(tmp_path)
        db.stores.create_collection("likes", "positive prefs", RecallMode.RELEVANT)

        first = db.stores.write(
            "likes",
            [EntryInput(key="a", content="hello")],
            author="chat",
        )
        second = db.stores.write(
            "likes",
            [EntryInput(key="b", content="hello")],
            author="chat",
        )
        assert first[0].outcome == "written"
        assert second[0].outcome == "written"
        assert len(db.stores.read_all("likes")) == 2

    def test_update_replaces_content(self, tmp_path):
        db = _make_db(tmp_path)
        db.stores.create_collection("likes", "positive prefs", RecallMode.RELEVANT)
        db.stores.write(
            "likes",
            [EntryInput(key="k", content="old body")],
            author="chat",
        )

        assert db.stores.update("likes", "k", "new body", "chat") == "ok"
        entries = db.stores.get_entry("likes", "k")
        assert entries[0].content == "new body"

    def test_update_not_found(self, tmp_path):
        db = _make_db(tmp_path)
        db.stores.create_collection("likes", "positive prefs", RecallMode.RELEVANT)
        assert db.stores.update("likes", "missing", "body", "chat") == "not_found"

    def test_delete_removes_all_matching(self, tmp_path):
        db = _make_db(tmp_path)
        db.stores.create_collection("likes", "positive prefs", RecallMode.RELEVANT)
        db.stores.write("likes", [EntryInput(key="k", content="a")], author="chat")
        assert db.stores.delete("likes", "k") == 1
        assert db.stores.get_entry("likes", "k") == []

    def test_move_transfers_entry(self, tmp_path):
        db = _make_db(tmp_path)
        db.stores.create_collection("unnotified", "pending", RecallMode.OFF)
        db.stores.create_collection("notified", "done", RecallMode.RELEVANT)
        db.stores.write(
            "unnotified", [EntryInput(key="thought-1", content="x")], author="thinking-agent"
        )

        outcome = db.stores.move("thought-1", "unnotified", "notified", author="notifier")
        assert outcome == "ok"
        assert db.stores.get_entry("unnotified", "thought-1") == []
        assert len(db.stores.get_entry("notified", "thought-1")) == 1

    def test_move_collision(self, tmp_path):
        db = _make_db(tmp_path)
        db.stores.create_collection("a", "src", RecallMode.OFF)
        db.stores.create_collection("b", "dst", RecallMode.OFF)
        db.stores.write("a", [EntryInput(key="k", content="src")], author="chat")
        db.stores.write("b", [EntryInput(key="k", content="dst")], author="chat")

        assert db.stores.move("k", "a", "b", author="chat") == "collision"

    def test_move_not_found(self, tmp_path):
        db = _make_db(tmp_path)
        db.stores.create_collection("a", "src", RecallMode.OFF)
        db.stores.create_collection("b", "dst", RecallMode.OFF)
        assert db.stores.move("missing", "a", "b", author="chat") == "not_found"


class TestLogAppend:
    def test_append_multiple_entries_stored_in_order(self, tmp_path):
        db = _make_db(tmp_path)
        db.stores.create_log("user-messages", "inbound", RecallMode.RELEVANT)
        db.stores.append(
            "user-messages",
            [
                LogEntryInput(content="hello"),
                LogEntryInput(content="are you there"),
            ],
            author="user",
        )

        entries = db.stores.read_all("user-messages")
        assert [e.content for e in entries] == ["hello", "are you there"]
        assert all(e.key is None for e in entries)
        assert all(e.author == "user" for e in entries)

    def test_append_to_collection_raises(self, tmp_path):
        db = _make_db(tmp_path)
        db.stores.create_collection("likes", "x", RecallMode.RELEVANT)
        with pytest.raises(StoreTypeError):
            db.stores.append("likes", [LogEntryInput(content="nope")], author="user")

    def test_write_to_log_raises(self, tmp_path):
        db = _make_db(tmp_path)
        db.stores.create_log("events", "x", RecallMode.RECENT)
        with pytest.raises(StoreTypeError):
            db.stores.write(
                "events",
                [EntryInput(key="k", content="v")],
                author="chat",
            )


class TestReads:
    def test_read_latest(self, tmp_path):
        db = _make_db(tmp_path)
        db.stores.create_log("events", "x", RecallMode.RECENT)
        for i in range(5):
            db.stores.append("events", [LogEntryInput(content=f"msg-{i}")], author="user")

        latest = db.stores.read_latest("events", 3)
        assert [e.content for e in latest] == ["msg-4", "msg-3", "msg-2"]

    def test_read_since(self, tmp_path):
        db = _make_db(tmp_path)
        db.stores.create_log("events", "x", RecallMode.RECENT)
        db.stores.append("events", [LogEntryInput(content="early")], author="user")
        mid = datetime.now(UTC)
        db.stores.append("events", [LogEntryInput(content="late")], author="user")

        after = db.stores.read_since("events", mid)
        assert [e.content for e in after] == ["late"]

    def test_read_random_returns_all_when_k_exceeds(self, tmp_path):
        db = _make_db(tmp_path)
        db.stores.create_collection("likes", "x", RecallMode.RELEVANT)
        db.stores.write("likes", [EntryInput(key="a", content="1")], author="chat")
        db.stores.write("likes", [EntryInput(key="b", content="2")], author="chat")
        picked = db.stores.read_random("likes", 5)
        assert {e.key for e in picked} == {"a", "b"}

    def test_read_random_no_k_returns_all(self, tmp_path):
        db = _make_db(tmp_path)
        db.stores.create_collection("likes", "x", RecallMode.RELEVANT)
        db.stores.write("likes", [EntryInput(key="a", content="1")], author="chat")
        db.stores.write("likes", [EntryInput(key="b", content="2")], author="chat")
        assert {e.key for e in db.stores.read_random("likes")} == {"a", "b"}

    def test_read_random_samples_subset_deterministically(self, tmp_path, monkeypatch):
        db = _make_db(tmp_path)
        db.stores.create_collection("likes", "x", RecallMode.RELEVANT)
        for letter in ("a", "b", "c", "d"):
            db.stores.write("likes", [EntryInput(key=letter, content=letter)], author="chat")

        import penny.database.store as store_mod

        captured: dict = {}

        def fake_sample(population, count):
            captured["population_size"] = len(population)
            captured["count"] = count
            return list(population[:count])

        monkeypatch.setattr(store_mod.random, "sample", fake_sample)

        picked = db.stores.read_random("likes", 2)
        assert [e.key for e in picked] == ["a", "b"]
        assert captured == {"population_size": 4, "count": 2}

    def test_read_similar_orders_by_cosine(self, tmp_path):
        db = _make_db(tmp_path)
        db.stores.create_collection("likes", "x", RecallMode.RELEVANT)
        anchor = [1.0, 0.0, 0.0, 0.0]
        db.stores.write(
            "likes",
            [
                EntryInput(
                    key="orth",
                    content="orthogonal",
                    content_embedding=[0.0, 0.0, 0.0, 1.0],
                ),
                EntryInput(
                    key="close",
                    content="halfway to anchor",
                    content_embedding=[0.5, 0.0, 0.87, 0.0],
                ),
                EntryInput(
                    key="exact",
                    content="anchor itself",
                    content_embedding=[1.0, 0.0, 0.0, 0.0],
                ),
            ],
            author="chat",
        )

        similar = db.stores.read_similar("likes", anchor, k=2)
        assert [e.key for e in similar] == ["exact", "close"]

    def test_read_similar_respects_floor(self, tmp_path):
        db = _make_db(tmp_path)
        db.stores.create_collection("likes", "x", RecallMode.RELEVANT)
        anchor = [1.0, 0.0]
        db.stores.write(
            "likes",
            [
                EntryInput(
                    key="off-topic",
                    content="unrelated",
                    content_embedding=[0.0, 1.0],
                )
            ],
            author="chat",
        )

        assert db.stores.read_similar("likes", anchor, k=5, floor=0.5) == []

    def test_keys_returns_unique_in_insertion_order(self, tmp_path):
        db = _make_db(tmp_path)
        db.stores.create_collection("likes", "x", RecallMode.RELEVANT)
        db.stores.write("likes", [EntryInput(key="first", content="1")], author="chat")
        db.stores.write("likes", [EntryInput(key="second", content="2")], author="chat")
        assert db.stores.keys("likes") == ["first", "second"]


class TestExists:
    def test_exists_by_exact_key(self, tmp_path):
        db = _make_db(tmp_path)
        db.stores.create_collection("likes", "x", RecallMode.RELEVANT)
        db.stores.write("likes", [EntryInput(key="dark roast", content="body")], author="chat")

        assert db.stores.exists(["likes"], "dark roast", None, None) is True
        assert db.stores.exists(["likes"], "not seen", None, None) is False

    def test_exists_by_similarity_across_stores(self, tmp_path):
        db = _make_db(tmp_path)
        db.stores.create_collection("unnotified", "pending", RecallMode.OFF)
        db.stores.create_collection("notified", "done", RecallMode.RELEVANT)
        shared = _unit_vec(2)
        db.stores.write(
            "notified",
            [
                EntryInput(
                    key="t1",
                    content="already notified",
                    content_embedding=shared,
                )
            ],
            author="notifier",
        )

        assert (
            db.stores.exists(
                ["unnotified", "notified"],
                key="t2",
                key_embedding=None,
                content_embedding=shared,
            )
            is True
        )


class TestCursorStore:
    def test_advance_and_get(self, tmp_path):
        db = _make_db(tmp_path)
        db.stores.create_log("user-messages", "inbound", RecallMode.RELEVANT)
        now = datetime.now(UTC)
        db.cursors.advance_committed("preference-extractor", "user-messages", now)

        assert db.cursors.get("preference-extractor", "user-messages") == now

    def test_advance_is_monotonic(self, tmp_path):
        db = _make_db(tmp_path)
        db.stores.create_log("user-messages", "inbound", RecallMode.RELEVANT)
        later = datetime.now(UTC)
        earlier = later - timedelta(minutes=5)

        db.cursors.advance_committed("preference-extractor", "user-messages", later)
        db.cursors.advance_committed("preference-extractor", "user-messages", earlier)

        assert db.cursors.get("preference-extractor", "user-messages") == later

    def test_missing_cursor_returns_none(self, tmp_path):
        db = _make_db(tmp_path)
        assert db.cursors.get("preference-extractor", "user-messages") is None


class TestMediaStore:
    def test_put_and_get_roundtrip(self, tmp_path):
        db = _make_db(tmp_path)
        media_id = db.media.put(b"binary payload", "image/png", source_url="https://x.test/a.png")
        entry = db.media.get(media_id)

        assert entry is not None
        assert entry.data == b"binary payload"
        assert entry.mime_type == "image/png"
        assert entry.source_url == "https://x.test/a.png"

    def test_get_missing_returns_none(self, tmp_path):
        db = _make_db(tmp_path)
        assert db.media.get(99999) is None


class TestWriteTypeEnforcement:
    def test_write_requires_collection(self, tmp_path):
        db = _make_db(tmp_path)
        db.stores.create_log("events", "x", RecallMode.RECENT)
        with pytest.raises(StoreTypeError):
            db.stores.write("events", [EntryInput(key="k", content="v")], author="chat")

    def test_write_on_missing_store_raises(self, tmp_path):
        db = _make_db(tmp_path)
        with pytest.raises(StoreNotFoundError):
            db.stores.write("nope", [EntryInput(key="k", content="v")], author="chat")

    def test_dedup_thresholds_configurable(self, tmp_path):
        db = _make_db(tmp_path)
        db.stores.create_collection("likes", "x", RecallMode.RELEVANT)
        db.stores.write(
            "likes",
            [
                EntryInput(
                    key="a",
                    content="body",
                    key_embedding=_unit_vec(0),
                    content_embedding=_unit_vec(1),
                )
            ],
            author="chat",
        )

        # Very lenient thresholds keep a near-duplicate; very strict rejects it.
        strict = DedupThresholds(key_sim=0.99, content_sim=0.99, combined=0.99)
        result = db.stores.write(
            "likes",
            [
                EntryInput(
                    key="b",
                    content="slightly different body",
                    key_embedding=_unit_vec(7),
                    content_embedding=_unit_vec(6),
                )
            ],
            author="chat",
            thresholds=strict,
        )
        assert result[0].outcome == "written"

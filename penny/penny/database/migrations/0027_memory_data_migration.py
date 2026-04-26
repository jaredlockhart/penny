"""Migrate existing data from per-domain tables into the unified memory framework.

Reads ``messagelog``, ``preference``, ``thought``, and ``knowledge``; populates
``memory_entry`` rows in the corresponding system memories.  Existing
embeddings on the source rows carry over as ``content_embedding``; the
thought ``title_embedding`` becomes the new entry's ``key_embedding``.
``key_embedding`` for preferences and knowledge is left NULL — future writes
that touch those entries will compute it; the dedup logic falls back to TCR
+ content cosine in the meantime.

Source rows the migration covers:

  messagelog WHERE direction='incoming' → user-messages   (author=user)
  messagelog WHERE direction='outgoing' → penny-messages  (author=chat or notify*)
  preference WHERE valence='positive'   → likes           (author=history)
  preference WHERE valence='negative'   → dislikes        (author=history)
  thought    WHERE notified_at IS NULL  → unnotified-thoughts (key=title)
  thought    WHERE notified_at IS NOT NULL → notified-thoughts (key=title)
  knowledge                             → knowledge (key=title, content=URL+summary)

* outgoing messages with thought_id set are attributed to ``notify``;
  everything else to ``chat``.  ScheduleExecutor messages can't be
  retroactively distinguished and end up as ``chat`` — best we can do.

Each block guards on the target memory being empty, so the migration is
safe to re-run after a manual revert.  The three system log memories
(user-messages, penny-messages, browse-results) are created by 0026; the
remaining five (likes, dislikes, unnotified-thoughts, notified-thoughts,
knowledge) are created here, idempotently.

Old tables remain in place — see Stage 12 of the memory-implementation-plan.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime


def up(conn: sqlite3.Connection) -> None:
    now = datetime.now(UTC).isoformat()
    _create_collection_memories(conn, now)
    if _table_exists(conn, "messagelog"):
        _migrate_messages(conn)
    if _table_exists(conn, "preference"):
        _migrate_preferences(conn)
    if _table_exists(conn, "thought"):
        _migrate_thoughts(conn)
    if _table_exists(conn, "knowledge"):
        _migrate_knowledge(conn)
    conn.commit()


# ── Helpers ──────────────────────────────────────────────────────────────────


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    """True if the named table is present in this DB.

    Source tables may be missing when the bootstrap-skipped path is exercised
    (test_skips_already_applied) or in partial recovery scenarios; in those
    cases there's nothing to migrate from that source, but the new memories
    should still be created.
    """
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _create_collection_memories(conn: sqlite3.Connection, now: str) -> None:
    """Create the five collection memories that consume migrated entries."""
    rows = [
        ("likes", "Topics the user has expressed positive sentiment about", "relevant"),
        ("dislikes", "Topics the user has expressed negative sentiment about", "relevant"),
        ("unnotified-thoughts", "Pending thoughts to share with the user", "off"),
        ("notified-thoughts", "Thoughts already shared with the user", "relevant"),
        ("knowledge", "Summarized facts from web pages Penny has read", "relevant"),
    ]
    for name, description, recall in rows:
        conn.execute(
            "INSERT OR IGNORE INTO memory"
            " (name, type, description, recall, archived, created_at)"
            " VALUES (?, 'collection', ?, ?, 0, ?)",
            (name, description, recall, now),
        )


def _is_memory_empty(conn: sqlite3.Connection, name: str) -> bool:
    """Return True when the named memory has no entries yet."""
    row = conn.execute(
        "SELECT 1 FROM memory_entry WHERE memory_name = ? LIMIT 1", (name,)
    ).fetchone()
    return row is None


def _migrate_messages(conn: sqlite3.Connection) -> None:
    """Populate user-messages and penny-messages from messagelog rows."""
    if _is_memory_empty(conn, "user-messages"):
        for content, embedding, timestamp in conn.execute(
            "SELECT content, embedding, timestamp FROM messagelog"
            " WHERE direction = 'incoming' ORDER BY timestamp ASC"
        ).fetchall():
            conn.execute(
                "INSERT INTO memory_entry"
                " (memory_name, key, content, author,"
                "  key_embedding, content_embedding, created_at)"
                " VALUES ('user-messages', NULL, ?, 'user', NULL, ?, ?)",
                (content, embedding, timestamp),
            )

    if _is_memory_empty(conn, "penny-messages"):
        for content, embedding, thought_id, timestamp in conn.execute(
            "SELECT content, embedding, thought_id, timestamp FROM messagelog"
            " WHERE direction = 'outgoing' ORDER BY timestamp ASC"
        ).fetchall():
            author = "notify" if thought_id is not None else "chat"
            conn.execute(
                "INSERT INTO memory_entry"
                " (memory_name, key, content, author,"
                "  key_embedding, content_embedding, created_at)"
                " VALUES ('penny-messages', NULL, ?, ?, NULL, ?, ?)",
                (content, author, embedding, timestamp),
            )


def _migrate_preferences(conn: sqlite3.Connection) -> None:
    """Populate likes/dislikes from preference rows split by valence."""
    for memory_name, valence in (("likes", "positive"), ("dislikes", "negative")):
        if not _is_memory_empty(conn, memory_name):
            continue
        for content, embedding, created_at in conn.execute(
            "SELECT content, embedding, created_at FROM preference"
            " WHERE valence = ? ORDER BY created_at ASC",
            (valence,),
        ).fetchall():
            conn.execute(
                "INSERT INTO memory_entry"
                " (memory_name, key, content, author,"
                "  key_embedding, content_embedding, created_at)"
                " VALUES (?, ?, ?, 'history', NULL, ?, ?)",
                (memory_name, content, content, embedding, created_at),
            )


def _migrate_thoughts(conn: sqlite3.Connection) -> None:
    """Populate unnotified-thoughts and notified-thoughts from thought rows.

    Thoughts without a title are skipped — collections require a key, and
    the modern thinking agent always sets one.  Title embeddings carry over
    as the entry's key_embedding.
    """
    for memory_name, where_clause in (
        ("unnotified-thoughts", "notified_at IS NULL"),
        ("notified-thoughts", "notified_at IS NOT NULL"),
    ):
        if not _is_memory_empty(conn, memory_name):
            continue
        rows = conn.execute(
            "SELECT title, content, embedding, title_embedding, created_at"
            f" FROM thought WHERE {where_clause} AND title IS NOT NULL"
            " ORDER BY created_at ASC"
        ).fetchall()
        for title, content, embedding, title_embedding, created_at in rows:
            conn.execute(
                "INSERT INTO memory_entry"
                " (memory_name, key, content, author,"
                "  key_embedding, content_embedding, created_at)"
                " VALUES (?, ?, ?, 'thinking', ?, ?, ?)",
                (memory_name, title, content, title_embedding, embedding, created_at),
            )


def _migrate_knowledge(conn: sqlite3.Connection) -> None:
    """Populate the knowledge collection with title-keyed entries.

    Each entry's content has the URL on the first line and the summary
    body below — same shape the chat agent's knowledge section used to
    render so retrieval feels identical.
    """
    if not _is_memory_empty(conn, "knowledge"):
        return
    for title, url, summary, embedding, created_at in conn.execute(
        "SELECT title, url, summary, embedding, created_at FROM knowledge ORDER BY created_at ASC"
    ).fetchall():
        body = f"URL: {url}\n\n{summary}"
        conn.execute(
            "INSERT INTO memory_entry"
            " (memory_name, key, content, author,"
            "  key_embedding, content_embedding, created_at)"
            " VALUES ('knowledge', ?, ?, 'history', NULL, ?, ?)",
            (title, body, embedding, created_at),
        )

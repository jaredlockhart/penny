"""Add embedding column to thought and messagelog tables for cached embeddings."""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Add embedding BLOB columns for caching embeddings."""
    for table in ("thought", "messagelog"):
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchall()
        ]
        if not tables:
            continue
        columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        if "embedding" not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN embedding BLOB")
    conn.commit()

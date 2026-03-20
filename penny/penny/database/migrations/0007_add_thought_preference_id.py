"""Add preference_id FK column to thought for dedup scoping."""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Add preference_id FK to thought linking thoughts to their seed preference."""
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='thought'"
        ).fetchall()
    ]
    if not tables:
        conn.commit()
        return
    columns = [row[1] for row in conn.execute("PRAGMA table_info(thought)").fetchall()]
    if "preference_id" not in columns:
        conn.execute(
            "ALTER TABLE thought ADD COLUMN preference_id INTEGER REFERENCES preference(id)"
        )
    indexes = {row[1] for row in conn.execute("PRAGMA index_list(thought)").fetchall()}
    if "ix_thought_preference_id" not in indexes:
        conn.execute("CREATE INDEX ix_thought_preference_id ON thought (preference_id)")
    conn.commit()

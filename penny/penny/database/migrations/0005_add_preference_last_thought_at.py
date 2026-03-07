"""Add last_thought_at column to preference table for seed rotation."""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Add last_thought_at to track when each preference was last used as a thinking seed."""
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='preference'"
        ).fetchall()
    ]
    if not tables:
        conn.commit()
        return
    columns = [row[1] for row in conn.execute("PRAGMA table_info(preference)").fetchall()]
    if "last_thought_at" not in columns:
        conn.execute("ALTER TABLE preference ADD COLUMN last_thought_at TIMESTAMP")
    conn.commit()

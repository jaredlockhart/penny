"""Add title column to thought table for dedup and image search."""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Add title column to thought table."""
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
    if "title" not in columns:
        conn.execute("ALTER TABLE thought ADD COLUMN title TEXT")
    conn.commit()

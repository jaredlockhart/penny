"""Add notified_at column to thought table for tracking shared thoughts."""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Add notified_at column to track which thoughts have been shared."""
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
    if "notified_at" not in columns:
        conn.execute("ALTER TABLE thought ADD COLUMN notified_at TIMESTAMP")
    conn.commit()

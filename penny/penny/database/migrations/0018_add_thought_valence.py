"""Add valence column to thought table for user reaction tracking."""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Add valence column to thought (1 = positive reaction, -1 = negative, NULL = unreacted)."""
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
    if "valence" not in columns:
        conn.execute("ALTER TABLE thought ADD COLUMN valence INTEGER")
    conn.commit()

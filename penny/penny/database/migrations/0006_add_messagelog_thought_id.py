"""Add thought_id FK column to messagelog for tracking proactive message sources."""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Add thought_id FK to messagelog linking proactive messages to their source thought."""
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messagelog'"
        ).fetchall()
    ]
    if not tables:
        conn.commit()
        return
    columns = [row[1] for row in conn.execute("PRAGMA table_info(messagelog)").fetchall()]
    if "thought_id" not in columns:
        conn.execute("ALTER TABLE messagelog ADD COLUMN thought_id INTEGER REFERENCES thought(id)")
    indexes = {row[1] for row in conn.execute("PRAGMA index_list(messagelog)").fetchall()}
    if "ix_messagelog_thought_id" not in indexes:
        conn.execute("CREATE INDEX ix_messagelog_thought_id ON messagelog (thought_id)")
    conn.commit()

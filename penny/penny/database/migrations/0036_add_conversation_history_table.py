"""Add conversation history table for daily topic summaries.

Stores compacted topic summaries of past conversations so Penny retains
awareness of what was discussed beyond the recent message window.
"""

from __future__ import annotations

import sqlite3


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Check whether a column exists in the given table."""
    columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    return column in columns


def up(conn: sqlite3.Connection) -> None:
    """Create the conversationhistory table and indexes."""
    if not _has_column(conn, "conversationhistory", "id"):
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversationhistory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT NOT NULL,
                period_start TIMESTAMP NOT NULL,
                period_end TIMESTAMP NOT NULL,
                duration TEXT NOT NULL,
                topics TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT (datetime('now'))
            )
        """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_conversationhistory_user ON conversationhistory (user)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_conversationhistory_period_start "
        "ON conversationhistory (period_start)"
    )
    conn.commit()

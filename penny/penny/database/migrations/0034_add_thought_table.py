"""Add thought table for inner monologue persistence.

Stores Penny's stream-of-consciousness thoughts — an append-only log
that gets rehydrated at the start of each inner monologue cycle and
injected into message handling for conversational continuity.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Create the thought table and indexes."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS thought (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS ix_thought_user ON thought (user)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_thought_created_at ON thought (created_at)")
    conn.commit()

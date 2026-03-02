"""Add recipient column to messagelog.

Outgoing messages were logged with sender=penny_id but no recipient,
making it impossible to query "outgoing messages to user X" without
a parent_id thread join.  Autonomous messages (parent_id IS NULL)
were invisible to conversation context and recall.

This migration adds a recipient column and backfills all outgoing
messages.  Threaded replies get the parent's sender; autonomous
messages (parent_id IS NULL) get the sole incoming sender since
there is currently only one user.
"""

from __future__ import annotations

import sqlite3


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Check whether a column exists in the given table."""
    columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    return column in columns


def up(conn: sqlite3.Connection) -> None:
    """Add recipient column and backfill from parent messages."""
    if not _has_column(conn, "messagelog", "recipient"):
        conn.execute("ALTER TABLE messagelog ADD COLUMN recipient TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_messagelog_recipient ON messagelog (recipient)")
    # Backfill only when the required columns exist (test DBs may have minimal schemas)
    if _has_column(conn, "messagelog", "direction") and _has_column(conn, "messagelog", "sender"):
        _backfill_recipients(conn)
    conn.commit()


def _backfill_recipients(conn: sqlite3.Connection) -> None:
    """Backfill recipient for existing outgoing messages."""
    # Threaded replies: copy the parent (incoming) message's sender
    conn.execute("""
        UPDATE messagelog
        SET recipient = (
            SELECT parent.sender
            FROM messagelog AS parent
            WHERE parent.id = messagelog.parent_id
        )
        WHERE direction = 'outgoing'
          AND parent_id IS NOT NULL
          AND recipient IS NULL
    """)
    # Autonomous messages: assign to the sole incoming sender
    conn.execute("""
        UPDATE messagelog
        SET recipient = (
            SELECT DISTINCT sender FROM messagelog WHERE direction = 'incoming' LIMIT 1
        )
        WHERE direction = 'outgoing'
          AND parent_id IS NULL
          AND recipient IS NULL
    """)

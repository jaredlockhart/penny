"""Add heat and heat_cooldown columns to entity table.

Heat is the persistent interest score for notification entity scoring.
Heat_cooldown tracks how many notification cycles an entity must sit out.
"""

from __future__ import annotations

import sqlite3


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Check if a column exists in a table."""
    row = conn.execute(
        f"SELECT 1 FROM pragma_table_info('{table}') WHERE name='{column}'"
    ).fetchone()
    return row is not None


def up(conn: sqlite3.Connection) -> None:
    """Add heat columns and seed initial heat from engagement history."""
    if not _has_column(conn, "entity", "heat"):
        conn.execute("ALTER TABLE entity ADD COLUMN heat REAL NOT NULL DEFAULT 0.0")
    if not _has_column(conn, "entity", "heat_cooldown"):
        conn.execute("ALTER TABLE entity ADD COLUMN heat_cooldown INTEGER NOT NULL DEFAULT 0")

    # Seed initial heat from positive engagement days (capped at 5.0)
    conn.execute("""
        UPDATE entity SET heat = MIN(COALESCE((
            SELECT COUNT(DISTINCT DATE(e.created_at))
            FROM engagement e
            WHERE e.entity_id = entity.id
              AND e.valence = 'positive'
              AND e.engagement_type IN (
                  'emoji_reaction', 'explicit_statement',
                  'follow_up_question', 'message_mention'
              )
        ), 0.0), 5.0)
    """)

    conn.commit()

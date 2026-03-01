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
    """Add heat and heat_cooldown columns to entity table."""
    if not _has_column(conn, "entity", "heat"):
        conn.execute("ALTER TABLE entity ADD COLUMN heat REAL NOT NULL DEFAULT 0.0")
    if not _has_column(conn, "entity", "heat_cooldown"):
        conn.execute("ALTER TABLE entity ADD COLUMN heat_cooldown INTEGER NOT NULL DEFAULT 0")

    conn.commit()

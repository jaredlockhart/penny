"""Add time-based heat decay columns to entity table.

Converts heat decay from cycle-based to wall-clock-time-based.
- heat_decayed_at: tracks when decay was last applied (for elapsed-time computation)
- heat_cooldown_until: deadline timestamp replacing cycle-counter cooldown
- Drops deprecated heat_cooldown cycle-counter column
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
    """Add heat_decayed_at and heat_cooldown_until columns to entity table."""
    if not _has_column(conn, "entity", "heat_decayed_at"):
        conn.execute("ALTER TABLE entity ADD COLUMN heat_decayed_at DATETIME")
    if not _has_column(conn, "entity", "heat_cooldown_until"):
        conn.execute("ALTER TABLE entity ADD COLUMN heat_cooldown_until DATETIME")

    # Drop deprecated cycle-based cooldown column
    if _has_column(conn, "entity", "heat_cooldown"):
        conn.execute("ALTER TABLE entity DROP COLUMN heat_cooldown")

    conn.commit()

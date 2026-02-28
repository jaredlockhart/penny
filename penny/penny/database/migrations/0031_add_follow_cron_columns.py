"""Add cron_expression, timing_description, and user_timezone to followprompt.

Replaces the simple cadence field with cron-based scheduling, reusing
the same natural language → cron semantics from /schedule.
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
    """Add cron scheduling columns and backfill from existing cadence values."""
    if not _has_column(conn, "followprompt", "cron_expression"):
        conn.execute(
            "ALTER TABLE followprompt ADD COLUMN cron_expression TEXT NOT NULL DEFAULT '0 9 * * *'"
        )
    if not _has_column(conn, "followprompt", "timing_description"):
        conn.execute(
            "ALTER TABLE followprompt ADD COLUMN timing_description TEXT NOT NULL DEFAULT 'daily'"
        )
    if not _has_column(conn, "followprompt", "user_timezone"):
        conn.execute(
            "ALTER TABLE followprompt ADD COLUMN user_timezone TEXT NOT NULL DEFAULT 'UTC'"
        )

    # Backfill from existing cadence values
    conn.execute(
        "UPDATE followprompt SET cron_expression = '0 * * * *', timing_description = 'hourly' "
        "WHERE cadence = 'hourly'"
    )
    conn.execute(
        "UPDATE followprompt SET cron_expression = '0 9 * * *', timing_description = 'daily' "
        "WHERE cadence = 'daily'"
    )
    conn.execute(
        "UPDATE followprompt SET cron_expression = '0 9 * * 1', timing_description = 'weekly' "
        "WHERE cadence = 'weekly'"
    )

    conn.commit()

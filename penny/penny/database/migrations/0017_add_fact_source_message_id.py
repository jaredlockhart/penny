"""Add source_message_id column to fact table for message-sourced facts."""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Add source_message_id to fact table."""
    # Check if fact table exists
    has_fact = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fact'"
    ).fetchone()
    if not has_fact:
        return

    # Check if column already exists
    has_col = conn.execute(
        "SELECT 1 FROM pragma_table_info('fact') WHERE name='source_message_id'"
    ).fetchone()
    if not has_col:
        conn.execute(
            "ALTER TABLE fact ADD COLUMN source_message_id INTEGER REFERENCES messagelog(id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_fact_source_message_id ON fact (source_message_id)"
        )

    conn.commit()

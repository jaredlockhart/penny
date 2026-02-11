"""Add Schedule table for recurring background tasks.

Type: schema
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Apply the migration."""
    # Create schedule table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schedule (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            user_timezone TEXT NOT NULL DEFAULT 'UTC',
            cron_expression TEXT NOT NULL,
            prompt_text TEXT NOT NULL,
            timing_description TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Add index on user_id for fast lookups
    conn.execute("""
        CREATE INDEX IF NOT EXISTS ix_schedule_user_id ON schedule (user_id)
    """)

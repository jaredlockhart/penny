"""Add Preference table for structured likes/dislikes.

Type: schema
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Apply the migration."""
    # Create preference table with columns: id, user, topic, type, created_at
    conn.execute("""
        CREATE TABLE IF NOT EXISTS preference (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            topic TEXT NOT NULL,
            type TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL
        )
    """)

    # Add index on user column for fast lookups
    conn.execute("""
        CREATE INDEX IF NOT EXISTS ix_preference_user ON preference (user)
    """)

    # Add unique constraint on (user, topic, type) to prevent duplicates
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ix_preference_user_topic_type
        ON preference (user, topic, type)
    """)

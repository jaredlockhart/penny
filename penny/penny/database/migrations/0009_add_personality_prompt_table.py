"""Add PersonalityPrompt table for storing user custom personality prompts.

Type: schema
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Apply the migration."""
    conn.execute("""
        CREATE TABLE personalityprompt (
            user_id TEXT PRIMARY KEY,
            prompt_text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    # Add index on user_id for quick lookups
    conn.execute("CREATE INDEX idx_personalityprompt_user_id ON personalityprompt(user_id)")

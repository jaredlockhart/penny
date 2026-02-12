"""Add user_style_profile table for adaptive speaking style.

Type: schema
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Apply the migration."""
    cursor = conn.cursor()

    # Check if table already exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='user_style_profile'"
    )
    table_exists = cursor.fetchone() is not None

    if not table_exists:
        conn.execute(
            """
            CREATE TABLE user_style_profile (
                user_id TEXT PRIMARY KEY,
                style_prompt TEXT NOT NULL,
                message_count INTEGER NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX idx_user_style_enabled ON user_style_profile(user_id, enabled)")

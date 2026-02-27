"""Add FollowPrompt table for ongoing event monitoring subscriptions.

FollowPrompt is like LearnPrompt but never-ending — it tracks topics the
user wants to monitor for ongoing events via the news API.

Type: schema
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Apply the migration."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS followprompt (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            prompt_text TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            query_terms TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL,
            last_polled_at TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS ix_followprompt_user ON followprompt (user)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_followprompt_status ON followprompt (status)")
    conn.commit()

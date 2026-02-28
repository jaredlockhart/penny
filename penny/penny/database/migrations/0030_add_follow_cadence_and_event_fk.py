"""Add cadence and last_notified_at to FollowPrompt, follow_prompt_id FK to Event.

Per-follow-prompt notification cadence decouples event notifications from
fact discovery, preventing event starvation of fact notifications.

Type: schema
"""

import sqlite3


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Check if a column exists in a table."""
    row = conn.execute(
        f"SELECT 1 FROM pragma_table_info('{table}') WHERE name='{column}'"
    ).fetchone()
    return row is not None


def up(conn: sqlite3.Connection) -> None:
    """Apply the migration."""
    if not _has_column(conn, "followprompt", "cadence"):
        conn.execute("ALTER TABLE followprompt ADD COLUMN cadence TEXT NOT NULL DEFAULT 'daily'")
    if not _has_column(conn, "followprompt", "last_notified_at"):
        conn.execute("ALTER TABLE followprompt ADD COLUMN last_notified_at TIMESTAMP")
    if not _has_column(conn, "event", "follow_prompt_id"):
        conn.execute(
            "ALTER TABLE event ADD COLUMN follow_prompt_id INTEGER REFERENCES followprompt(id)"
        )
    conn.execute("CREATE INDEX IF NOT EXISTS ix_event_follow_prompt_id ON event (follow_prompt_id)")
    conn.commit()

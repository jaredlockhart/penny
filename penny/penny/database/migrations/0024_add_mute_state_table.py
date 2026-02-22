"""Add MuteState table for per-user notification muting.

Row exists = muted. Delete row = unmuted.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Apply the migration."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mutestate (
            user TEXT PRIMARY KEY,
            muted_at TIMESTAMP NOT NULL
        )
    """)

"""Add announced_at column to learnprompt table.

Tracks when the learn completion announcement was sent to the user.
NULL means no announcement has been sent yet. Once set, the learn prompt
is hidden from /learn status display.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Apply the migration."""
    has_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='learnprompt'"
    ).fetchone()
    if not has_table:
        return

    has_col = conn.execute(
        "SELECT 1 FROM pragma_table_info('learnprompt') WHERE name='announced_at'"
    ).fetchone()
    if not has_col:
        conn.execute("ALTER TABLE learnprompt ADD COLUMN announced_at TIMESTAMP")

    conn.commit()

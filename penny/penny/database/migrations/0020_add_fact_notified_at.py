"""Add notified_at column to fact table.

Tracks when each fact was communicated to the user via notification.
NULL means not yet notified. Existing facts are backfilled with their
learned_at value so the notification agent doesn't re-notify on deploy.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Apply the migration."""
    has_fact = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fact'"
    ).fetchone()
    if not has_fact:
        return

    has_col = conn.execute(
        "SELECT 1 FROM pragma_table_info('fact') WHERE name='notified_at'"
    ).fetchone()
    if not has_col:
        conn.execute("ALTER TABLE fact ADD COLUMN notified_at TIMESTAMP")
        # Backfill: treat all existing facts as already notified
        conn.execute("UPDATE fact SET notified_at = learned_at")

    conn.commit()

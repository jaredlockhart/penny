"""Add last_notified_at column to entity table.

Tracks when each entity was last included in a notification so the
NotificationAgent can enforce a per-entity cooldown and rotate
notifications across more entities.

Type: schema
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Apply the migration."""
    has_col = conn.execute(
        "SELECT 1 FROM pragma_table_info('entity') WHERE name='last_notified_at'"
    ).fetchone()
    if not has_col:
        conn.execute("ALTER TABLE entity ADD COLUMN last_notified_at TIMESTAMP DEFAULT NULL")
    conn.commit()

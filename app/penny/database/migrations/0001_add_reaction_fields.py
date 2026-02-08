"""Add reaction tracking fields to messagelog.

Type: schema
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Add is_reaction and external_id columns to messagelog."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(messagelog)")
    existing = {row[1] for row in cursor.fetchall()}

    if "is_reaction" not in existing:
        conn.execute("ALTER TABLE messagelog ADD COLUMN is_reaction BOOLEAN DEFAULT 0")
    if "external_id" not in existing:
        conn.execute("ALTER TABLE messagelog ADD COLUMN external_id VARCHAR DEFAULT NULL")

"""Add processed field to MessageLog for tracking processed reactions.

Type: schema
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Apply the migration."""
    # Check if processed column already exists (may be present in new databases from SQLModel)
    cursor = conn.execute("PRAGMA table_info(messagelog)")
    columns = {row[1] for row in cursor.fetchall()}

    if "processed" not in columns:
        # Add processed column to track which reactions have been analyzed
        conn.execute("ALTER TABLE messagelog ADD COLUMN processed INTEGER DEFAULT 0")

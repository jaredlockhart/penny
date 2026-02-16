"""Add options column to research_tasks table.
Type: schema
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Apply the migration."""
    cursor = conn.execute("PRAGMA table_info(research_tasks)")
    columns = {row[1] for row in cursor.fetchall()}

    if not columns:
        return  # Table doesn't exist yet; SQLModel will create it with the column

    if "options" not in columns:
        conn.execute("ALTER TABLE research_tasks ADD COLUMN options TEXT DEFAULT NULL")

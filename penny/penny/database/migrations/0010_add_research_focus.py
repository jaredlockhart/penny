"""Add focus column to research_tasks for clarification step.

Type: schema
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Apply the migration."""
    # Check if table exists (old DBs created before research feature won't have it)
    cursor = conn.execute("PRAGMA table_info(research_tasks)")
    columns = {row[1] for row in cursor.fetchall()}

    if not columns:
        return  # Table doesn't exist yet; SQLModel will create it with the column

    if "focus" not in columns:
        conn.execute("ALTER TABLE research_tasks ADD COLUMN focus TEXT DEFAULT NULL")

"""Add embedding BLOB columns to entity, fact, and preference tables."""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Add embedding columns for vector storage."""
    cursor = conn.cursor()

    # Add embedding column to each table (nullable, NULL = not yet computed)
    for table in ("entity", "fact", "preference"):
        # Check if column already exists
        columns = [row[1] for row in cursor.execute(f"PRAGMA table_info({table})").fetchall()]
        if "embedding" not in columns:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN embedding BLOB")

    conn.commit()

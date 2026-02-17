"""Add Entity table and extraction cursor for knowledge base.

Type: schema
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Apply the migration."""
    # Create entity table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            name TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            facts TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL
        )
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS ix_entity_user ON entity (user)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS ix_entity_entity_type ON entity (entity_type)
    """)

    # Unique constraint on (user, name, entity_type) to prevent duplicates
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ix_entity_user_name_type
        ON entity (user, name, entity_type)
    """)

    # Cursor table for tracking entity extraction progress per source type
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entity_extraction_cursor (
            source_type TEXT PRIMARY KEY,
            last_processed_id INTEGER NOT NULL DEFAULT 0
        )
    """)

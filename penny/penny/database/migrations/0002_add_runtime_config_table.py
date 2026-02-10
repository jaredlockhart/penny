"""Add runtime_config table for user-configurable settings.

Type: schema
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Create runtime_config table."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runtime_config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            description TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

"""Add domain_permission table for server-side domain allowlist."""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Create domain_permission table."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS domain_permission (
            id INTEGER PRIMARY KEY,
            domain TEXT NOT NULL UNIQUE,
            permission TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_domain_permission_domain "
        "ON domain_permission (domain)"
    )
    conn.commit()

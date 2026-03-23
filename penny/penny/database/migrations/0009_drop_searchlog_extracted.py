"""Drop vestigial extracted column from searchlog — entity system removed."""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Drop the extracted column from searchlog if it exists."""
    columns = [row[1] for row in conn.execute("PRAGMA table_info(searchlog)").fetchall()]
    if "extracted" in columns:
        conn.execute("ALTER TABLE searchlog DROP COLUMN extracted")
    conn.commit()

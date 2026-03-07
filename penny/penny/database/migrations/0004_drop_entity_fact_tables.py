"""Drop entity and fact tables — knowledge system removed."""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Drop entity and fact tables."""
    conn.execute("DROP TABLE IF EXISTS fact")
    conn.execute("DROP TABLE IF EXISTS entity")
    conn.commit()

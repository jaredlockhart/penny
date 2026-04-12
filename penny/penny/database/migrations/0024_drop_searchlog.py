"""Drop the legacy searchlog table — never written to since the browser-based search migration."""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Drop the searchlog table and its indexes if they exist."""
    conn.execute("DROP INDEX IF EXISTS ix_searchlog_timestamp")
    conn.execute("DROP INDEX IF EXISTS ix_searchlog_query")
    conn.execute("DROP INDEX IF EXISTS ix_searchlog_trigger")
    conn.execute("DROP TABLE IF EXISTS searchlog")
    conn.commit()

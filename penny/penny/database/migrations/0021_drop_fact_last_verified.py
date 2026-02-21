"""Drop last_verified column from fact table.

The fact verification mechanism was never used effectively.
Priority scoring now uses interest × (1/fact_count) without staleness.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Drop last_verified column from fact table."""
    has_fact = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fact'"
    ).fetchone()
    if not has_fact:
        return

    has_col = conn.execute(
        "SELECT 1 FROM pragma_table_info('fact') WHERE name='last_verified'"
    ).fetchone()
    if has_col:
        conn.execute("ALTER TABLE fact DROP COLUMN last_verified")

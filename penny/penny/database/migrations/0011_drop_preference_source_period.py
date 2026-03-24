"""Drop vestigial source_period_start/end columns from preference table.

These fields tracked which conversation period a preference was extracted from,
but extraction now uses the processed flag on messages instead of date ranges.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Drop source_period_start and source_period_end from preference."""
    columns = [row[1] for row in conn.execute("PRAGMA table_info(preference)").fetchall()]
    if "source_period_start" in columns:
        conn.execute("ALTER TABLE preference DROP COLUMN source_period_start")
    if "source_period_end" in columns:
        conn.execute("ALTER TABLE preference DROP COLUMN source_period_end")
    conn.commit()

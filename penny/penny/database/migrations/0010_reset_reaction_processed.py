"""Reset processed flag on reaction messages so the new reaction pipeline picks them up.

Reactions were previously marked processed alongside text messages, but the old
pipeline never actually extracted preferences from them. Resetting to unprocessed
lets the new deterministic reaction preference pipeline catch up on history.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Set processed=0 on all reaction messages."""
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    if "messagelog" in tables:
        conn.execute("UPDATE messagelog SET processed = 0 WHERE is_reaction = 1")
    conn.commit()

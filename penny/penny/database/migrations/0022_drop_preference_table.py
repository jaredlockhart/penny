"""Drop preference table and engagement.preference_id column.

The /like and /dislike commands have been removed — organic engagement
signals (learn, reactions, mentions, searches) now cover all use cases.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Drop preference_id from engagement, then drop the preference table."""
    # Drop preference_id column from engagement table
    has_engagement = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='engagement'"
    ).fetchone()
    if has_engagement:
        has_col = conn.execute(
            "SELECT 1 FROM pragma_table_info('engagement') WHERE name='preference_id'"
        ).fetchone()
        if has_col:
            conn.execute("ALTER TABLE engagement DROP COLUMN preference_id")

    # Drop the preference table
    conn.execute("DROP TABLE IF EXISTS preference")

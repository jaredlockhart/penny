"""Reset processed flag on non-reaction messages for PreferenceAgent reprocessing.

The PreferenceAgent was rewritten to use a two-pass approach (likes then dislikes)
instead of a single combined pass. Reset all non-reaction messages so they get
reprocessed with the new extraction logic.

Type: data
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Apply the migration."""
    cursor = conn.execute("PRAGMA table_info(messagelog)")
    columns = {row[1] for row in cursor.fetchall()}
    if "is_reaction" in columns and "processed" in columns:
        conn.execute("UPDATE messagelog SET processed = 0 WHERE is_reaction = 0 AND processed = 1")

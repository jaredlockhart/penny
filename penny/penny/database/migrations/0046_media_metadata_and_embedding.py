"""Add metadata + embedding columns to the media table.

The image side-channel matches outgoing message text against stored images by
embedding distance, so each media row gains a ``title`` and an ``embedding`` of
its title+URL metadata.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(media)").fetchall()}

    if "title" not in columns:
        conn.execute("ALTER TABLE media ADD COLUMN title TEXT")
    if "embedding" not in columns:
        conn.execute("ALTER TABLE media ADD COLUMN embedding BLOB")

    conn.commit()

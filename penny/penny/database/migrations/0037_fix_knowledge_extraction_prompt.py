"""Fix knowledge extraction_prompt: replace collection_update with update_entry.

Type: data

The prompt seeded by migration 0031 instructs the Collector to call
``collection_update`` when merging new page content into an existing entry.
``collection_update`` is a chat-only lifecycle tool (modifies collection
metadata); it is not in the Collector's scoped tool surface.  The correct
tool is ``update_entry``, which replaces an entry's content by key and IS
available to collectors.

This migration replaces only the exact wrong call string — it is idempotent
if run on a DB where 0031 never ran or where the user has since customised
the prompt.
"""

from __future__ import annotations

import sqlite3

_OLD = 'call collection_update("knowledge", key=<title>,'
_NEW = 'call update_entry("knowledge", key=<title>,'


def up(conn: sqlite3.Connection) -> None:
    """Replace collection_update with update_entry in the knowledge extraction_prompt."""
    conn.execute(
        "UPDATE memory SET extraction_prompt = REPLACE(extraction_prompt, ?, ?) "
        "WHERE name = 'knowledge' AND extraction_prompt LIKE ?",
        (_OLD, _NEW, f"%{_OLD}%"),
    )
    conn.commit()

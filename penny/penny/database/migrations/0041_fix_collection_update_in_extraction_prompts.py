"""Fix extraction_prompts: replace collection_update with update_entry.

Type: data

``collection_update`` is a chat-only lifecycle tool that modifies collection
*metadata* (description, recall mode, extraction_prompt, interval).  It is not
available in the Collector's scoped tool surface.  The correct tool for
replacing an existing entry's content in a collector is ``update_entry``.

Migration 0037 fixed this for the system ``knowledge`` collection.  This
migration generalises the fix to *all* collections — covering user-created
collections whose extraction_prompts were written with the same mistake (e.g.
``supplement-routine`` which contains "update the entry via collection_update").

The replacement is safe: in an extraction_prompt context, every reference to
``collection_update`` is an error — no collector can call it, and the intent
is always to update an entry's content (``update_entry``).
"""

from __future__ import annotations

import sqlite3

_OLD = "collection_update"
_NEW = "update_entry"


def up(conn: sqlite3.Connection) -> None:
    """Replace collection_update with update_entry in all extraction_prompts."""
    conn.execute(
        "UPDATE memory "
        "SET extraction_prompt = REPLACE(extraction_prompt, ?, ?) "
        "WHERE extraction_prompt LIKE ?",
        (_OLD, _NEW, f"%{_OLD}%"),
    )
    conn.commit()

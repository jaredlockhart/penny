"""Fix stale extraction_prompts: replace collection_read_latest with read_latest.

Type: data

The tool collection_read_latest was unified into the shape-agnostic read_latest
(which works for both collections and logs).  Stored extraction_prompt values
written before the rename still reference the old name, causing ToolExecutor to
return a "Tool not found" error when the Collector runs those cycles.

This migration replaces every occurrence of the old tool name with the new one
across all stored extraction_prompts.  Safe to re-run (REPLACE is idempotent
when the target string is already absent).
"""

from __future__ import annotations

import sqlite3

_OLD = "collection_read_latest"
_NEW = "read_latest"


def up(conn: sqlite3.Connection) -> None:
    """Replace collection_read_latest with read_latest in all extraction_prompts."""
    conn.execute(
        "UPDATE memory SET extraction_prompt = REPLACE(extraction_prompt, ?, ?) "
        "WHERE extraction_prompt LIKE ?",
        (_OLD, _NEW, f"%{_OLD}%"),
    )
    conn.commit()

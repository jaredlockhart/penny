"""Add read_similar guidance to the knowledge extraction_prompt.

Type: data

The Collector LLM was hallucinating a non-existent tool ``collection_search``
when trying to find existing knowledge entries by similarity after a
duplicate-rejection error.  The correct tool is ``read_similar``.

The knowledge extraction prompt (seeded by 0031, patched by 0037) only
describes ``collection_get`` for exact-key lookup.  When ``collection_get``
returns nothing but ``collection_write`` fails with a duplicate rejection,
the model had no guidance and invented ``collection_search``.

This migration patches step 3 of the knowledge extraction_prompt to:
  1. Explicitly name ``read_similar`` as the similarity-search tool.
  2. Note that no ``collection_search`` tool exists.
"""

from __future__ import annotations

import sqlite3

_OLD = (
    'call collection_get("knowledge", key=<page '
    "title>) to see whether you already have a summary.  If one is "
    'returned, call update_entry("knowledge", key=<title>, '
    "content=<merged paragraph>) — integrate any new details from "
    "this fetch while preserving existing ones.  Otherwise, call "
    'collection_write("knowledge", entries=[{key: <title>, '
    "content: <new paragraph>}])."
)

_NEW = (
    'call collection_get("knowledge", key=<page '
    "title>) to see whether you already have a summary.  If one is "
    'returned, call update_entry("knowledge", key=<title>, '
    "content=<merged paragraph>) — integrate any new details from "
    "this fetch while preserving existing ones.  If collection_get "
    "returns nothing but collection_write fails with a duplicate "
    'error, call read_similar("knowledge", anchor=<page title>) to '
    "find the closest existing entry by similarity, then call "
    "update_entry on that key instead.  (There is no "
    "collection_search tool — use read_similar for any "
    "similarity-based lookup.)  Otherwise, call "
    'collection_write("knowledge", entries=[{key: <title>, '
    "content: <new paragraph>}])."
)


def up(conn: sqlite3.Connection) -> None:
    """Add read_similar guidance to the knowledge extraction_prompt."""
    conn.execute(
        "UPDATE memory SET extraction_prompt = REPLACE(extraction_prompt, ?, ?) "
        "WHERE name = 'knowledge' AND extraction_prompt LIKE ?",
        (_OLD, _NEW, f"%{_OLD}%"),
    )
    conn.commit()

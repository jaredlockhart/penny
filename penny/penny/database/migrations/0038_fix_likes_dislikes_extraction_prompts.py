"""Fix likes and dislikes extraction_prompts: use explicit tool names for corrections.

Type: data

The prompts seeded by migration 0031 instruct the Collector to "update or delete
that entry" for corrections without naming which tools to call.  This vague phrasing
causes the LLM to hallucinate ``collection_update_entry`` — a blend of the real
``collection_delete_entry`` (visible in the runtime rules) and ``update_entry``
(the correct tool for content updates).

This migration replaces the vague phrases with explicit tool calls, matching the
style already used by the knowledge extraction_prompt.

Idempotent — the REPLACE is a no-op when the old substring is absent.
"""

from __future__ import annotations

import sqlite3

_LIKES_OLD = "accurate (e.g. 'I don't actually like X anymore'), update or delete that entry."
_LIKES_NEW = (
    "accurate (e.g. 'I don't actually like X anymore'), call "
    'update_entry("likes", key=<topic>, content=<updated message>) '
    'or collection_delete_entry("likes", key=<topic>).'
)

_DISLIKES_OLD = "applies, update or delete that entry."
_DISLIKES_NEW = (
    'applies, call update_entry("dislikes", key=<topic>, '
    'content=<updated message>) or collection_delete_entry("dislikes", key=<topic>).'
)


def up(conn: sqlite3.Connection) -> None:
    """Replace vague 'update or delete' instructions with explicit tool calls."""
    conn.execute(
        "UPDATE memory SET extraction_prompt = REPLACE(extraction_prompt, ?, ?) "
        "WHERE name = 'likes' AND extraction_prompt LIKE ?",
        (_LIKES_OLD, _LIKES_NEW, f"%{_LIKES_OLD}%"),
    )
    conn.execute(
        "UPDATE memory SET extraction_prompt = REPLACE(extraction_prompt, ?, ?) "
        "WHERE name = 'dislikes' AND extraction_prompt LIKE ?",
        (_DISLIKES_OLD, _DISLIKES_NEW, f"%{_DISLIKES_OLD}%"),
    )
    conn.commit()

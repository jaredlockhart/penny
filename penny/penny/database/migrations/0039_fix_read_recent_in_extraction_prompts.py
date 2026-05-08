"""Fix extraction prompts that reference the non-existent tool 'read_recent'.

Type: data

The correct tool name is 'log_read_recent'.  User-created or LLM-generated
extraction prompts may contain 'read_recent(' instead, causing the Collector
agent to produce a "Tool not found: read_recent" error on every cycle.

This migration replaces every occurrence of 'read_recent(' with 'log_read_recent('
across all non-NULL extraction_prompt values.  The parenthesis anchor ensures
only the tool call form is rewritten and not any prose description that happens
to use the phrase 'read recent'.
"""

from __future__ import annotations

import sqlite3

_OLD = "read_recent("
_NEW = "log_read_recent("


def up(conn: sqlite3.Connection) -> None:
    """Replace read_recent( with log_read_recent( in all extraction prompts."""
    conn.execute(
        "UPDATE memory SET extraction_prompt = REPLACE(extraction_prompt, ?, ?) "
        "WHERE extraction_prompt LIKE ?",
        (_OLD, _NEW, f"%{_OLD}%"),
    )
    conn.commit()

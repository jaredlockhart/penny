"""Fix extraction prompts that reference the non-existent tool 'log_read_log'.

Type: data

The correct tool name is 'log_read_next'.  User-created or LLM-generated
extraction prompts may contain 'log_read_log(' instead, causing the Collector
agent to produce a "Tool not found: log_read_log" error on every cycle.

This migration replaces every occurrence of 'log_read_log(' with 'log_read_next('
across all non-NULL extraction_prompt values.  The parenthesis anchor ensures
only the tool call form is rewritten and not any prose description that happens
to use the phrase 'log read log'.
"""

from __future__ import annotations

import sqlite3

_OLD = "log_read_log("
_NEW = "log_read_next("


def up(conn: sqlite3.Connection) -> None:
    """Replace log_read_log( with log_read_next( in all extraction prompts."""
    conn.execute(
        "UPDATE memory SET extraction_prompt = REPLACE(extraction_prompt, ?, ?) "
        "WHERE extraction_prompt LIKE ?",
        (_OLD, _NEW, f"%{_OLD}%"),
    )
    conn.commit()

"""Rename read_latest( → log_read_latest( in stored extraction_prompts.

Type: data

ReadLatestTool was registered as ``read_latest`` but the original plan and
naming convention (log_read_next, log_read_recent) call for ``log_read_latest``.
Migration 0033 wrote extraction_prompts that call ``read_latest(...)``; this
migration rewrites those stored prompts to use the correct tool name so running
collector cycles no longer hit "Tool not found: log_read_latest".
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Replace read_latest( with log_read_latest( in all extraction_prompts."""
    conn.execute(
        "UPDATE memory "
        "SET extraction_prompt = REPLACE(extraction_prompt, 'read_latest(', 'log_read_latest(') "
        "WHERE extraction_prompt LIKE '%read_latest(%'"
    )
    conn.commit()

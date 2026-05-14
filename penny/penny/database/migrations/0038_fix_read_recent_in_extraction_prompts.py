"""Fix bare read_recent( → log_read_recent( in stored extraction_prompts.

Type: data

The model occasionally calls read_recent (without the log_ prefix) because
system prompts listed read_latest / read_similar / etc., allowing it to infer
a read_recent by analogy.  Any extraction_prompt rows that were written with
the bare form will feed the wrong tool name into every collector cycle.

Uses Python-level regex (negative lookbehind) so log_read_recent( occurrences
are not double-prefixed.  Idempotent — rows that already use log_read_recent
are unchanged.
"""

from __future__ import annotations

import re
import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Replace bare read_recent( with log_read_recent( in extraction_prompts."""
    rows = conn.execute(
        "SELECT name, extraction_prompt FROM memory WHERE extraction_prompt LIKE '%read_recent(%'"
    ).fetchall()
    for name, prompt in rows:
        fixed = re.sub(r"(?<!log_)read_recent\(", "log_read_recent(", prompt)
        if fixed != prompt:
            conn.execute(
                "UPDATE memory SET extraction_prompt = ? WHERE name = ?",
                (fixed, name),
            )
    conn.commit()

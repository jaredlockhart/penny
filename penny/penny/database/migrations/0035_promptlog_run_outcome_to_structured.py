"""Restructure ``promptlog`` run outcome from a free-text string into a
typed (success, reason, target) triple.

The browser addon's prompts tab renders a green/red tag per run.  Up
through PR #1021 only the legacy ``run_outcome`` string carried this
signal, with the convention "starts with ``Stored``" → green.  Now that
the unified Collector dispatcher (``agent_name="collector"``) drives
every per-collection cycle, the tag needs to carry: did the cycle
succeed, what's the reason text, and which collection was the target.

Schema change:
  - rename ``run_outcome`` → ``run_reason`` (free-text reason text)
  - add ``run_success BOOLEAN`` (NULL for non-collector agents)
  - add ``run_target TEXT`` (collection name; NULL for non-collector agents)

No data backfill needed — ``set_run_outcome`` was wired but never called
from production code (only tests), so every row has ``run_outcome IS
NULL`` in the live DB.  ``ALTER TABLE ... RENAME COLUMN`` requires
SQLite 3.25+, which Python 3.14 ships.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "promptlog" not in tables:
        return

    columns = {row[1] for row in conn.execute("PRAGMA table_info(promptlog)").fetchall()}
    if "run_outcome" in columns and "run_reason" not in columns:
        conn.execute("ALTER TABLE promptlog RENAME COLUMN run_outcome TO run_reason")
    if "run_success" not in columns:
        conn.execute("ALTER TABLE promptlog ADD COLUMN run_success BOOLEAN")
    if "run_target" not in columns:
        conn.execute("ALTER TABLE promptlog ADD COLUMN run_target TEXT")
    conn.commit()

"""Add ``extraction_prompt`` column to the ``memory`` table.

The ``extraction_prompt`` is the body of instructions given to a per-collection
collector subagent each cycle: read recent entries from input log streams,
extract structured records relevant to the collection's scope, write them back.

Nullable.  Collections that aren't populated by collectors — logs, system
collections like ``notified-thoughts`` / ``unnotified-thoughts`` (populated by
thinking + notify), ``user-profile`` (populated by chat), and any user
collection not yet wired with a prompt — leave this NULL.  The scheduler only
registers collectors for collections where ``extraction_prompt IS NOT NULL``.

Idempotent — safe to re-run.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(memory)").fetchall()}
    if "extraction_prompt" not in columns:
        conn.execute("ALTER TABLE memory ADD COLUMN extraction_prompt TEXT")
    conn.commit()

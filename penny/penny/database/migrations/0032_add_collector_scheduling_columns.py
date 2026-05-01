"""Add per-collection scheduling columns to ``memory``.

Phase 6 of the collector refactor introduces a single dispatcher
``Collector`` agent that wakes idle-gated on a fast tick and runs
whichever collection is most overdue.  Two new nullable columns drive
the dispatch:

  - ``collector_interval_seconds`` — how long between this collection's
    runs.  NULL means "use the global default" (``COLLECTOR_DEFAULT_INTERVAL``).
  - ``last_collected_at`` — when the dispatcher last ran this collection
    (regardless of whether the agent did real work or just ``done()``).
    NULL means "never collected" — first cycle picks it up immediately.

Per-collection backfill of these columns + extraction_prompt for
``unnotified-thoughts`` and ``notified-thoughts`` happens in 0033.

Idempotent — schema only.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(memory)").fetchall()}
    if "collector_interval_seconds" not in columns:
        conn.execute("ALTER TABLE memory ADD COLUMN collector_interval_seconds INTEGER")
    if "last_collected_at" not in columns:
        conn.execute("ALTER TABLE memory ADD COLUMN last_collected_at TIMESTAMP")
    conn.commit()

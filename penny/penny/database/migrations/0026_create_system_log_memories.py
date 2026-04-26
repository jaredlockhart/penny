"""Create the three system log memories that side-effect writes target.

- ``user-messages`` (log, recall=relevant) — every incoming user message
- ``penny-messages`` (log, recall=recent)  — every outgoing Penny reply
- ``browse-results`` (log, recall=off)     — every browse-tool fetch result

These are the system-level memories the channel adapter and browse tool
write to automatically (Stage 9 of the memory framework rollout). They
must exist before the side-effect writes can succeed; this migration
creates them with the recall modes the recall assembler expects.

Idempotent — uses ``INSERT OR IGNORE`` so re-running is a no-op.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime


def up(conn: sqlite3.Connection) -> None:
    now = datetime.now(UTC).isoformat()
    rows = [
        ("user-messages", "log", "Every incoming user message", "relevant"),
        ("penny-messages", "log", "Every outgoing Penny reply", "recent"),
        ("browse-results", "log", "Every browse-tool fetch result", "off"),
    ]
    for name, mtype, description, recall in rows:
        conn.execute(
            "INSERT OR IGNORE INTO memory (name, type, description, recall, archived, created_at)"
            " VALUES (?, ?, ?, ?, 0, ?)",
            (name, mtype, description, recall, now),
        )
    conn.commit()

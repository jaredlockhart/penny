"""Create the ``collector-runs`` log memory.

One entry per Collector cycle, written by the Collector after the agent
loop exits.  Content captures the target collection name, a success
marker (``✅`` / ``❌``), and the prose ``summary`` arg from ``done()``
— so Penny can audit collector behaviour from chat by reading the log
and see at a glance which collectors are doing real work, which are
quiet, and which are persistently failing.

``recall=off`` because surfacing every collector cycle in ambient
context would be overwhelming noise — Penny pulls the log explicitly
via ``log_read_recent`` / ``read_latest`` when she wants to inspect
collector activity.

Idempotent — INSERT OR IGNORE.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime


def up(conn: sqlite3.Connection) -> None:
    now = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO memory "
        "(name, type, description, recall, archived, created_at) "
        "VALUES (?, ?, ?, ?, 0, ?)",
        (
            "collector-runs",
            "log",
            "One entry per Collector cycle: target, marker, done() summary",
            "off",
            now,
        ),
    )
    conn.commit()

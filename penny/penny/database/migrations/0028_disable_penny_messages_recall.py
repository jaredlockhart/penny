"""Disable ambient recall for the ``penny-messages`` log.

The chat agent already passes the recent conversation as alternating
user/assistant turns via ``history=`` (built from ``db.messages``).
Surfacing ``penny-messages`` in the recall block re-injected the same
content as a system-prompt section — pure duplication.  Flip recall to
``off`` so the log is still written and tool-readable but no longer
contributes to ambient context.

Idempotent — UPDATE is a no-op if the row is already ``off`` or missing.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    conn.execute(
        "UPDATE memory SET recall = 'off' WHERE name = 'penny-messages'",
    )
    conn.commit()

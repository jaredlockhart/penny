"""Add source and mention_count columns to preference for mention threshold gating."""

from __future__ import annotations

import sqlite3

# Default threshold — used to grandfather existing extracted preferences
DEFAULT_MENTION_THRESHOLD = 3


def up(conn: sqlite3.Connection) -> None:
    """Add source and mention_count to preference, infer source from timestamps."""
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='preference'"
        ).fetchall()
    ]
    if not tables:
        conn.commit()
        return

    columns = [row[1] for row in conn.execute("PRAGMA table_info(preference)").fetchall()]

    if "source" not in columns:
        conn.execute("ALTER TABLE preference ADD COLUMN source TEXT DEFAULT 'extracted'")

    if "mention_count" not in columns:
        conn.execute("ALTER TABLE preference ADD COLUMN mention_count INTEGER DEFAULT 1")

    # Infer source from timestamps: manual prefs have identical start/end
    conn.execute(
        "UPDATE preference SET source = 'manual' WHERE source_period_start = source_period_end"
    )

    # Read configured threshold from runtime_config, fall back to default
    threshold = _get_threshold(conn)

    # Grandfather existing extracted preferences so they remain thinking candidates
    conn.execute(
        "UPDATE preference SET mention_count = ? WHERE source = 'extracted'",
        (threshold,),
    )

    # Create index on source
    indexes = {row[1] for row in conn.execute("PRAGMA index_list(preference)").fetchall()}
    if "ix_preference_source" not in indexes:
        conn.execute("CREATE INDEX ix_preference_source ON preference (source)")

    # Mark all existing messages as processed so only new messages are extracted
    msg_tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messagelog'"
        ).fetchall()
    ]
    if msg_tables:
        conn.execute("UPDATE messagelog SET processed = 1 WHERE processed = 0")

    conn.commit()


def _get_threshold(conn: sqlite3.Connection) -> int:
    """Read PREFERENCE_MENTION_THRESHOLD from runtime_config, or use default."""
    row = conn.execute(
        "SELECT value FROM runtime_config WHERE key = 'PREFERENCE_MENTION_THRESHOLD'"
    ).fetchone()
    if row:
        try:
            return int(row[0])
        except Exception:
            pass
    return DEFAULT_MENTION_THRESHOLD

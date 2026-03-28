"""Add device table and device_id FK on messagelog."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime


def up(conn: sqlite3.Connection) -> None:
    """Create device table, seed from existing messages, add device_id FK to messagelog."""
    _create_device_table(conn)
    _seed_device_from_messages(conn)
    _add_device_id_to_messagelog(conn)
    conn.commit()


def _create_device_table(conn: sqlite3.Connection) -> None:
    """Create the device table if it doesn't exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS device (
            id INTEGER PRIMARY KEY,
            channel_type TEXT NOT NULL,
            identifier TEXT NOT NULL UNIQUE,
            label TEXT NOT NULL,
            is_default INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS ix_device_channel_type ON device (channel_type)")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_device_identifier ON device (identifier)")


def _seed_device_from_messages(conn: sqlite3.Connection) -> None:
    """Seed a device row from existing incoming messages."""
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messagelog'"
        ).fetchall()
    ]
    if not tables:
        return

    row = conn.execute(
        "SELECT DISTINCT sender FROM messagelog WHERE direction='incoming' LIMIT 1"
    ).fetchone()
    if not row:
        return

    sender = row[0]
    # Determine channel type from sender format
    if sender.startswith("+"):
        channel_type = "signal"
        label = "Signal"
    elif "#" in sender:
        channel_type = "discord"
        label = "Discord"
    else:
        channel_type = "signal"
        label = "Signal"

    now = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO device (channel_type, identifier, label, is_default, created_at) "
        "VALUES (?, ?, ?, 1, ?)",
        (channel_type, sender, label, now),
    )


def _add_device_id_to_messagelog(conn: sqlite3.Connection) -> None:
    """Add device_id column to messagelog and backfill from device table."""
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messagelog'"
        ).fetchall()
    ]
    if not tables:
        return

    columns = [row[1] for row in conn.execute("PRAGMA table_info(messagelog)").fetchall()]
    if "device_id" not in columns:
        conn.execute("ALTER TABLE messagelog ADD COLUMN device_id INTEGER REFERENCES device(id)")

    # Backfill device_id from sender → device.identifier lookup
    conn.execute(
        """
        UPDATE messagelog
        SET device_id = (SELECT id FROM device WHERE identifier = messagelog.sender)
        WHERE device_id IS NULL
          AND sender IN (SELECT identifier FROM device)
        """
    )

    # Also backfill outgoing messages via recipient → device.identifier
    conn.execute(
        """
        UPDATE messagelog
        SET device_id = (SELECT id FROM device WHERE identifier = messagelog.recipient)
        WHERE device_id IS NULL
          AND recipient IN (SELECT identifier FROM device)
        """
    )

    indexes = {row[1] for row in conn.execute("PRAGMA index_list(messagelog)").fetchall()}
    if "ix_messagelog_device_id" not in indexes:
        conn.execute("CREATE INDEX ix_messagelog_device_id ON messagelog (device_id)")

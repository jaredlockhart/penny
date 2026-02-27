"""Add Event and EventEntity tables for time-aware knowledge.

Events are time-stamped occurrences that can involve multiple entities
(M2M via event_entity junction table). This enables tracking news,
developments, and changes about topics the user follows.

Type: schema
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Apply the migration."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS event (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            headline TEXT NOT NULL,
            summary TEXT NOT NULL,
            occurred_at TIMESTAMP NOT NULL,
            discovered_at TIMESTAMP NOT NULL,
            source_url TEXT,
            source_type TEXT NOT NULL,
            external_id TEXT,
            notified_at TIMESTAMP,
            embedding BLOB
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS ix_event_user ON event (user)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_event_occurred_at ON event (occurred_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_event_source_type ON event (source_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_event_external_id ON event (external_id)")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS event_entity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL REFERENCES event(id),
            entity_id INTEGER NOT NULL REFERENCES entity(id),
            UNIQUE(event_id, entity_id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS ix_event_entity_event_id ON event_entity (event_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_event_entity_entity_id ON event_entity (entity_id)")
    conn.commit()

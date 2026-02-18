"""Add engagement table for interest tracking."""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Create the engagement table."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS engagement (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            entity_id INTEGER REFERENCES entity(id),
            preference_id INTEGER REFERENCES preference(id),
            engagement_type TEXT NOT NULL,
            valence TEXT NOT NULL,
            strength REAL NOT NULL,
            source_message_id INTEGER REFERENCES messagelog(id),
            created_at TIMESTAMP NOT NULL
        )
    """)

    conn.execute("CREATE INDEX IF NOT EXISTS ix_engagement_user ON engagement (user)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_engagement_entity_id ON engagement (entity_id)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_engagement_preference_id ON engagement (preference_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_engagement_engagement_type ON engagement (engagement_type)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS ix_engagement_created_at ON engagement (created_at)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_engagement_source_message_id"
        " ON engagement (source_message_id)"
    )

    conn.commit()

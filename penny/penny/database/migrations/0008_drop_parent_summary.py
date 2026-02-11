"""Drop parent_summary column from MessageLog table.

Type: schema
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Apply the migration."""
    # Check if parent_summary column exists
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(messagelog)")
    existing = {row[1] for row in cursor.fetchall()}

    # If parent_summary doesn't exist, nothing to do
    if "parent_summary" not in existing:
        return

    # SQLite doesn't support ALTER TABLE DROP COLUMN directly for older versions
    # We need to recreate the table without the column

    # 1. Create new table without parent_summary
    conn.execute("""
        CREATE TABLE messagelog_new (
            id INTEGER PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            direction VARCHAR NOT NULL,
            sender VARCHAR NOT NULL,
            content VARCHAR NOT NULL,
            parent_id INTEGER,
            signal_timestamp INTEGER,
            external_id VARCHAR,
            is_reaction BOOLEAN NOT NULL DEFAULT 0,
            processed BOOLEAN NOT NULL DEFAULT 0,
            FOREIGN KEY (parent_id) REFERENCES messagelog (id)
        )
    """)

    # 2. Copy data from old table to new (excluding parent_summary)
    conn.execute("""
        INSERT INTO messagelog_new (
            id, timestamp, direction, sender, content, parent_id,
            signal_timestamp, external_id, is_reaction, processed
        )
        SELECT
            id, timestamp, direction, sender, content, parent_id,
            signal_timestamp, external_id, is_reaction, processed
        FROM messagelog
    """)

    # 3. Drop old table
    conn.execute("DROP TABLE messagelog")

    # 4. Rename new table to old name
    conn.execute("ALTER TABLE messagelog_new RENAME TO messagelog")

    # 5. Recreate indexes
    conn.execute("CREATE INDEX ix_messagelog_timestamp ON messagelog (timestamp)")
    conn.execute("CREATE INDEX ix_messagelog_direction ON messagelog (direction)")
    conn.execute("CREATE INDEX ix_messagelog_sender ON messagelog (sender)")
    conn.execute("CREATE INDEX ix_messagelog_parent_id ON messagelog (parent_id)")
    conn.execute("CREATE INDEX ix_messagelog_external_id ON messagelog (external_id)")
    conn.execute("CREATE INDEX ix_messagelog_is_reaction ON messagelog (is_reaction)")

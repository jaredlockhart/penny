"""Replace entity_extraction_cursor with entity_search_log join table.

Enables most-recent-first processing and tracks entity-to-search provenance.

Type: schema
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Apply the migration."""
    # Create join table linking entities to search logs
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entity_search_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER REFERENCES entity(id),
            search_log_id INTEGER NOT NULL REFERENCES searchlog(id),
            created_at TIMESTAMP NOT NULL
        )
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS ix_entity_search_log_search
        ON entity_search_log (search_log_id)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS ix_entity_search_log_entity
        ON entity_search_log (entity_id)
    """)

    # Backfill: mark previously-processed search logs as processed
    # by inserting sentinel rows (entity_id=NULL).
    # Guard: searchlog and entity_extraction_cursor are only present in production
    # databases (searchlog is created by SQLModel, not a migration).
    has_searchlog = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='searchlog'"
    ).fetchone()
    has_cursor = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='entity_extraction_cursor'"
    ).fetchone()

    if has_searchlog and has_cursor:
        conn.execute("""
            INSERT INTO entity_search_log (entity_id, search_log_id, created_at)
            SELECT NULL, s.id, CURRENT_TIMESTAMP
            FROM searchlog s
            WHERE s.id <= COALESCE(
                (SELECT last_processed_id FROM entity_extraction_cursor
                 WHERE source_type = 'search'),
                0
            )
        """)

    # Drop the old cursor table
    conn.execute("DROP TABLE IF EXISTS entity_extraction_cursor")

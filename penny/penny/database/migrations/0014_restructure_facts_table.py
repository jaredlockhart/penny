"""Restructure facts into dedicated table with source tracking.

Replaces the Entity.facts text blob with individual Fact rows.
Replaces EntitySearchLog join table with extracted flag on SearchLog.

Type: schema + data
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Apply the migration."""
    # Guard: only run if entity table exists (created by migration 0012 or create_tables).
    # Minimal test databases (just messagelog) won't have it.
    has_entity = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='entity'"
    ).fetchone()
    if not has_entity:
        return

    # 1. Create fact table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fact (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL REFERENCES entity(id),
            content TEXT NOT NULL,
            source_url TEXT,
            source_search_log_id INTEGER REFERENCES searchlog(id),
            learned_at TIMESTAMP NOT NULL,
            last_verified TIMESTAMP
        )
    """)

    conn.execute("CREATE INDEX IF NOT EXISTS ix_fact_entity ON fact (entity_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_fact_search ON fact (source_search_log_id)")

    # 2. Migrate existing entity facts text blobs into individual Fact rows.
    # Each line starting with "- " becomes a Fact row. Source is unknown for
    # migrated data, so source_url and source_search_log_id are left NULL.
    # Guard: entity.facts column only exists in pre-migration databases.
    has_facts_col = conn.execute(
        "SELECT 1 FROM pragma_table_info('entity') WHERE name='facts'"
    ).fetchone()
    if has_facts_col:
        entities = conn.execute(
            "SELECT id, facts, created_at FROM entity WHERE facts != ''"
        ).fetchall()
        for entity_id, facts_text, created_at in entities:
            for line in facts_text.strip().split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    content = line[2:].strip()
                elif line:
                    content = line
                else:
                    continue
                if content:
                    conn.execute(
                        "INSERT INTO fact (entity_id, content, learned_at) VALUES (?, ?, ?)",
                        (entity_id, content, created_at),
                    )

    # 3. Add extracted column to searchlog (skip if already present or table missing)
    has_searchlog = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='searchlog'"
    ).fetchone()
    if has_searchlog:
        has_extracted_col = conn.execute(
            "SELECT 1 FROM pragma_table_info('searchlog') WHERE name='extracted'"
        ).fetchone()
        if not has_extracted_col:
            conn.execute("ALTER TABLE searchlog ADD COLUMN extracted BOOLEAN NOT NULL DEFAULT 0")

    # 4. Backfill extracted from entity_search_log:
    # any search_log_id that appears in entity_search_log has been processed.
    # Both tables must exist (searchlog may not exist in minimal test databases).
    has_esl = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='entity_search_log'"
    ).fetchone()
    if has_esl and has_searchlog:
        conn.execute("""
            UPDATE searchlog SET extracted = 1
            WHERE id IN (SELECT DISTINCT search_log_id FROM entity_search_log)
        """)

    # 5. Drop entity_search_log table
    conn.execute("DROP TABLE IF EXISTS entity_search_log")

    # 6. Drop facts column from entity (skip if already absent from create_tables)
    if has_facts_col:
        conn.execute("ALTER TABLE entity DROP COLUMN facts")

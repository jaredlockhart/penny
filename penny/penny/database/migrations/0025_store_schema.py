"""Add store, store_entry, agent_cursor, and media tables.

Foundation for the task/collection framework: collections and logs are
unified in a single `store` table (type-discriminated) with entries in
`store_entry`. `agent_cursor` tracks per-agent read progress through logs.
`media` stores binary blobs referenced by `<media:ID>` tokens in entry content.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }

    if "store" not in tables:
        conn.execute("""
            CREATE TABLE store (
                name TEXT PRIMARY KEY,
                type TEXT NOT NULL CHECK (type IN ('collection', 'log')),
                description TEXT NOT NULL,
                recall TEXT NOT NULL CHECK (recall IN ('off', 'recent', 'relevant', 'all')),
                archived INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP NOT NULL
            )
        """)
        conn.execute("CREATE INDEX ix_store_archived ON store (archived)")

    if "store_entry" not in tables:
        conn.execute("""
            CREATE TABLE store_entry (
                id INTEGER PRIMARY KEY,
                store_name TEXT NOT NULL REFERENCES store(name),
                key TEXT,
                content TEXT NOT NULL,
                author TEXT NOT NULL,
                key_embedding BLOB,
                content_embedding BLOB,
                created_at TIMESTAMP NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX ix_store_entry_by_created ON store_entry (store_name, created_at)"
        )
        conn.execute("CREATE INDEX ix_store_entry_by_key ON store_entry (store_name, key)")
        conn.execute("CREATE INDEX ix_store_entry_author ON store_entry (author)")

    if "agent_cursor" not in tables:
        conn.execute("""
            CREATE TABLE agent_cursor (
                agent_name TEXT NOT NULL,
                store_name TEXT NOT NULL REFERENCES store(name),
                last_read_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                PRIMARY KEY (agent_name, store_name)
            )
        """)

    if "media" not in tables:
        conn.execute("""
            CREATE TABLE media (
                id INTEGER PRIMARY KEY,
                mime_type TEXT NOT NULL,
                data BLOB NOT NULL,
                source_url TEXT,
                created_at TIMESTAMP NOT NULL
            )
        """)

    conn.commit()

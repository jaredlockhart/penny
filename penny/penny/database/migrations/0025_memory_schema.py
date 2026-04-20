"""Add memory, memory_entry, agent_cursor, and media tables.

Foundation for the task/memory framework: collections and logs are unified
in a single `memory` table (type-discriminated) with entries in
`memory_entry`. `agent_cursor` tracks per-agent read progress through logs.
`media` stores binary blobs referenced by `<media:ID>` tokens in entry content.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }

    if "memory" not in tables:
        conn.execute("""
            CREATE TABLE memory (
                name TEXT PRIMARY KEY,
                type TEXT NOT NULL CHECK (type IN ('collection', 'log')),
                description TEXT NOT NULL,
                recall TEXT NOT NULL CHECK (recall IN ('off', 'recent', 'relevant', 'all')),
                archived INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP NOT NULL
            )
        """)
        conn.execute("CREATE INDEX ix_memory_archived ON memory (archived)")

    if "memory_entry" not in tables:
        conn.execute("""
            CREATE TABLE memory_entry (
                id INTEGER PRIMARY KEY,
                memory_name TEXT NOT NULL REFERENCES memory(name),
                key TEXT,
                content TEXT NOT NULL,
                author TEXT NOT NULL,
                key_embedding BLOB,
                content_embedding BLOB,
                created_at TIMESTAMP NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX ix_memory_entry_by_created ON memory_entry (memory_name, created_at)"
        )
        conn.execute("CREATE INDEX ix_memory_entry_by_key ON memory_entry (memory_name, key)")
        conn.execute("CREATE INDEX ix_memory_entry_author ON memory_entry (author)")

    if "agent_cursor" not in tables:
        conn.execute("""
            CREATE TABLE agent_cursor (
                agent_name TEXT NOT NULL,
                memory_name TEXT NOT NULL REFERENCES memory(name),
                last_read_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                PRIMARY KEY (agent_name, memory_name)
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

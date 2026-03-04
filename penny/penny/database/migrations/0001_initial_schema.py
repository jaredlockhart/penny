"""Initial schema — all tables in their final state.

Replaces 38 incremental migrations with a single bootstrap.
Tables are created via CREATE TABLE IF NOT EXISTS so this is safe
to run against databases already created by SQLModel.create_tables().
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes."""
    _create_promptlog(conn)
    _create_searchlog(conn)
    _create_messagelog(conn)
    _create_userinfo(conn)
    _create_command_logs(conn)
    _create_runtime_config(conn)
    _create_schedule(conn)
    _create_entity(conn)
    _create_mutestate(conn)
    _create_fact(conn)
    _create_thought(conn)
    _create_preference(conn)
    _create_conversationhistory(conn)
    conn.commit()


def _create_promptlog(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS promptlog (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP NOT NULL,
            model TEXT NOT NULL,
            messages TEXT NOT NULL,
            tools TEXT,
            response TEXT NOT NULL,
            thinking TEXT,
            duration_ms INTEGER
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS ix_promptlog_timestamp ON promptlog (timestamp)")


def _create_searchlog(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS searchlog (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP NOT NULL,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            duration_ms INTEGER,
            extracted BOOLEAN NOT NULL DEFAULT 0,
            trigger TEXT NOT NULL DEFAULT 'user_message'
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS ix_searchlog_timestamp ON searchlog (timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_searchlog_query ON searchlog (query)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_searchlog_trigger ON searchlog (trigger)")


def _create_messagelog(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messagelog (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP NOT NULL,
            direction TEXT NOT NULL,
            sender TEXT NOT NULL,
            content TEXT NOT NULL,
            parent_id INTEGER REFERENCES messagelog(id),
            signal_timestamp INTEGER,
            recipient TEXT,
            external_id TEXT,
            is_reaction BOOLEAN NOT NULL DEFAULT 0,
            processed BOOLEAN NOT NULL DEFAULT 0
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS ix_messagelog_timestamp ON messagelog (timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_messagelog_direction ON messagelog (direction)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_messagelog_sender ON messagelog (sender)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_messagelog_parent_id ON messagelog (parent_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_messagelog_recipient ON messagelog (recipient)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_messagelog_external_id ON messagelog (external_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_messagelog_is_reaction ON messagelog (is_reaction)")


def _create_userinfo(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS userinfo (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            location TEXT NOT NULL,
            timezone TEXT NOT NULL,
            date_of_birth TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL
        )
    """)
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_userinfo_sender ON userinfo (sender)")


def _create_command_logs(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS command_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP NOT NULL,
            user TEXT NOT NULL,
            channel_type TEXT NOT NULL,
            command_name TEXT NOT NULL,
            command_args TEXT NOT NULL,
            response TEXT NOT NULL,
            error TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS ix_command_logs_timestamp ON command_logs (timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_command_logs_user ON command_logs (user)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_command_logs_command_name ON command_logs (command_name)"
    )


def _create_runtime_config(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runtime_config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            description TEXT NOT NULL,
            updated_at TIMESTAMP NOT NULL
        )
    """)


def _create_schedule(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schedule (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            user_timezone TEXT NOT NULL DEFAULT 'UTC',
            cron_expression TEXT NOT NULL,
            prompt_text TEXT NOT NULL,
            timing_description TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS ix_schedule_user_id ON schedule (user_id)")


def _create_entity(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL,
            embedding BLOB,
            tagline TEXT,
            last_enriched_at TIMESTAMP,
            last_notified_at TIMESTAMP,
            heat REAL NOT NULL DEFAULT 0.0,
            heat_decayed_at TIMESTAMP,
            heat_cooldown_until TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS ix_entity_user ON entity (user)")


def _create_mutestate(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mutestate (
            user TEXT PRIMARY KEY,
            muted_at TIMESTAMP NOT NULL
        )
    """)


def _create_fact(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fact (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL REFERENCES entity(id),
            content TEXT NOT NULL,
            source_url TEXT,
            source_search_log_id INTEGER REFERENCES searchlog(id),
            source_message_id INTEGER REFERENCES messagelog(id),
            learned_at TIMESTAMP NOT NULL,
            notified_at TIMESTAMP,
            embedding BLOB
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS ix_fact_entity_id ON fact (entity_id)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_fact_source_search_log_id ON fact (source_search_log_id)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS ix_fact_source_message_id ON fact (source_message_id)")


def _create_thought(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS thought (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS ix_thought_user ON thought (user)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_thought_created_at ON thought (created_at)")


def _create_preference(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS preference (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            content TEXT NOT NULL,
            valence TEXT NOT NULL,
            embedding BLOB,
            source_period_start TIMESTAMP NOT NULL,
            source_period_end TIMESTAMP NOT NULL,
            created_at TIMESTAMP NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS ix_preference_user ON preference (user)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_preference_valence ON preference (valence)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_preference_created_at ON preference (created_at)")


def _create_conversationhistory(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversationhistory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            period_start TIMESTAMP NOT NULL,
            period_end TIMESTAMP NOT NULL,
            duration TEXT NOT NULL,
            topics TEXT NOT NULL,
            embedding BLOB,
            created_at TIMESTAMP NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_conversationhistory_user ON conversationhistory (user)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_conversationhistory_period_start"
        " ON conversationhistory (period_start)"
    )

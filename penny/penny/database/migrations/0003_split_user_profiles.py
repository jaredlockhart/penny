"""Split user profiles into UserInfo (basic info) and UserTopics (interests).

Type: schema
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Create UserInfo table and rename UserProfile to UserTopics."""
    cursor = conn.cursor()

    # Check if userprofile table exists (it should)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='userprofile'")
    has_userprofile = cursor.fetchone() is not None

    # Create UserInfo table
    cursor.execute("PRAGMA table_info(userinfo)")
    userinfo_exists = len(cursor.fetchall()) > 0

    if not userinfo_exists:
        conn.execute("""
            CREATE TABLE userinfo (
                id INTEGER PRIMARY KEY,
                sender VARCHAR NOT NULL UNIQUE,
                name VARCHAR NOT NULL,
                location VARCHAR NOT NULL,
                timezone VARCHAR NOT NULL,
                date_of_birth VARCHAR NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX ix_userinfo_sender ON userinfo (sender)")

    # Rename userprofile to usertopics
    if has_userprofile:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='usertopics'")
        has_usertopics = cursor.fetchone() is not None

        if not has_usertopics:
            conn.execute("ALTER TABLE userprofile RENAME TO usertopics")

"""Add LearnPrompt table and trigger/learn_prompt_id columns to SearchLog.

Phase 1 of Knowledge System v2: search trigger tracking for two-mode
extraction and LearnPrompt lifecycle tracking for /learn command.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Apply the migration."""
    # 1. Create learnprompt table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS learnprompt (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            prompt_text TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            searches_remaining INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS ix_learnprompt_user ON learnprompt (user)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_learnprompt_status ON learnprompt (status)")

    # 2. Add trigger column to searchlog
    has_searchlog = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='searchlog'"
    ).fetchone()
    if has_searchlog:
        has_trigger = conn.execute(
            "SELECT 1 FROM pragma_table_info('searchlog') WHERE name='trigger'"
        ).fetchone()
        if not has_trigger:
            conn.execute(
                "ALTER TABLE searchlog ADD COLUMN trigger TEXT NOT NULL DEFAULT 'user_message'"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS ix_searchlog_trigger ON searchlog (trigger)")

        # 3. Add learn_prompt_id column to searchlog
        has_lp_id = conn.execute(
            "SELECT 1 FROM pragma_table_info('searchlog') WHERE name='learn_prompt_id'"
        ).fetchone()
        if not has_lp_id:
            conn.execute(
                "ALTER TABLE searchlog ADD COLUMN"
                " learn_prompt_id INTEGER REFERENCES learnprompt(id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_searchlog_learn_prompt_id"
                " ON searchlog (learn_prompt_id)"
            )

    conn.commit()

"""Add prompt_type column to promptlog.

Identifies which flow within an agent produced the prompt
(e.g., user_message, free, daily_summary).
"""


def up(conn):
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "promptlog" not in tables:
        return
    columns = [row[1] for row in conn.execute("PRAGMA table_info(promptlog)").fetchall()]
    if "prompt_type" not in columns:
        conn.execute("ALTER TABLE promptlog ADD COLUMN prompt_type TEXT")

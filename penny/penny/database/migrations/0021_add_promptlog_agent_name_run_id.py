"""Add agent_name and run_id columns to promptlog.

Enables grouping prompt logs by which agent produced them and which
agentic loop invocation they belong to.
"""


def up(conn):
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "promptlog" not in tables:
        return
    columns = [row[1] for row in conn.execute("PRAGMA table_info(promptlog)").fetchall()]
    if "agent_name" not in columns:
        conn.execute("ALTER TABLE promptlog ADD COLUMN agent_name TEXT")
    if "run_id" not in columns:
        conn.execute("ALTER TABLE promptlog ADD COLUMN run_id TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_promptlog_run_id ON promptlog (run_id)")

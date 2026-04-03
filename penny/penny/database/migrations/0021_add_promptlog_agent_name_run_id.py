"""Add agent_name, prompt_type, and run_id columns to promptlog.

Enables a three-part taxonomy (agent_name, prompt_type, run_id) for
grouping prompt logs by agent, flow type, and agentic loop invocation.
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
    if "prompt_type" not in columns:
        conn.execute("ALTER TABLE promptlog ADD COLUMN prompt_type TEXT")
    if "run_id" not in columns:
        conn.execute("ALTER TABLE promptlog ADD COLUMN run_id TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_promptlog_run_id ON promptlog (run_id)")

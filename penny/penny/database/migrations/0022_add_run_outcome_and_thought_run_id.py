"""Add run_outcome to promptlog and run_id to thought.

run_outcome: free-text outcome of a thinking run (stored, discard reason).
thought.run_id: links a stored thought back to the thinking run that created it.
"""


def up(conn):
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }

    if "promptlog" in tables:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(promptlog)").fetchall()]
        if "run_outcome" not in columns:
            conn.execute("ALTER TABLE promptlog ADD COLUMN run_outcome TEXT")

    if "thought" in tables:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(thought)").fetchall()]
        if "run_id" not in columns:
            conn.execute("ALTER TABLE thought ADD COLUMN run_id TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS ix_thought_run_id ON thought (run_id)")

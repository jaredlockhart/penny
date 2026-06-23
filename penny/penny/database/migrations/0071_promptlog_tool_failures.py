"""Add ``promptlog.tool_failures`` — the per-run count of failed tool calls.

Type: schema

A collector cycle's authoritative tool-failure signal (``ToolCallRecord.failed``,
``= not ToolResult.success``) is computed in the agent loop but was never
persisted — the only render-time trace of a failure was the framed ``role:"tool"``
result text, which is heuristic to parse.  This column records the count
structurally so the run-health classifier (``classify_run``) can read it instead
of guessing from text.

Stamped on the run's last prompt row alongside ``run_outcome``/``run_reason`` (see
``MessageStore.set_run_outcome``), so it is NULL for old rows and for non-collector
runs that never get tagged — read as "unknown / not measured", which the
classifier treats as zero failures.
"""


def up(conn):
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "promptlog" not in tables:
        return
    columns = [row[1] for row in conn.execute("PRAGMA table_info(promptlog)").fetchall()]
    if "tool_failures" not in columns:
        conn.execute("ALTER TABLE promptlog ADD COLUMN tool_failures INTEGER")
    conn.commit()

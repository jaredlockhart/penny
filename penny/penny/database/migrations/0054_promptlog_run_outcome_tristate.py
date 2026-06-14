"""Replace promptlog.run_success (bool) with run_outcome (tri-state enum).

Type: schema

``run_success`` couldn't tell a clean no-op apart from real work — a quiet
"no new matches" collector cycle was ``success=True`` just like a cycle that
wrote entries.  ``run_outcome`` makes the cycle's result first-class:
``failed`` / ``no_work`` / ``worked`` / ``cancelled`` (see ``RunOutcome``).

Backfill is best-effort for existing rows: ``success=0`` → ``failed`` and
``success=1`` → ``worked``.  The work/no-work split isn't recoverable for
history (old successes didn't record whether they changed anything), so old
quiet cycles will read as ``worked``; new cycles are precise.
"""


def up(conn):
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "promptlog" not in tables:
        return
    columns = [row[1] for row in conn.execute("PRAGMA table_info(promptlog)").fetchall()]

    if "run_outcome" not in columns:
        conn.execute("ALTER TABLE promptlog ADD COLUMN run_outcome TEXT")
    if "run_success" in columns:
        conn.execute(
            "UPDATE promptlog SET run_outcome = "
            "CASE WHEN run_success = 1 THEN 'worked' WHEN run_success = 0 THEN 'failed' END "
            "WHERE run_success IS NOT NULL AND run_outcome IS NULL"
        )
        conn.execute("ALTER TABLE promptlog DROP COLUMN run_success")
    conn.commit()

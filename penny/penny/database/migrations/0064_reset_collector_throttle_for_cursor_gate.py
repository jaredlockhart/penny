"""Reset auto-throttle state now that the cursor gate governs log-driven collectors.

Type: data

Log-driven collections (those that read a log via ``log_read``) are no longer
throttled — a pre-model cursor gate skips their idle ticks instead, and they
stay pinned at ``base_interval_seconds``.  But instances upgrading from the
throttle-only regime carry collections whose ``collector_interval_seconds`` was
doubled well past their base during the recent quiet stretch (e.g. a 300s
collector stretched to 38400s).  Left as-is, such a collection would wait out
that stretched floor before its first post-upgrade run even once its input log
moves again — exactly the catch-up lag the gate removes.

So snap every collector's current interval back to its base and clear the idle
counter, giving the new regime a clean slate.  Generative / collection-driven
collections (no log cursor) simply re-accumulate idle runs and re-throttle from
base as before; this only un-does throttle state that no longer applies.
"""


def up(conn):
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "memory" not in tables:
        return
    conn.execute(
        "UPDATE memory SET collector_interval_seconds = base_interval_seconds,"
        " consecutive_idle_runs = 0"
        " WHERE extraction_prompt IS NOT NULL AND base_interval_seconds IS NOT NULL"
    )
    conn.commit()

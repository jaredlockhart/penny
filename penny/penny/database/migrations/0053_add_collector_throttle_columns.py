"""Add auto-throttle bookkeeping columns to ``memory``.

Type: schema

``base_interval_seconds`` is the user's intended collector cadence — the value
a collection's ``collector_interval_seconds`` snaps back to when a cycle
produces work.  ``consecutive_idle_runs`` counts cycles that produced nothing;
at ``COLLECTOR_THROTTLE_AFTER`` the collector doubles its interval (capped at
``COLLECTOR_MAX_INTERVAL``) and resets the counter.

Existing collections backfill ``base_interval_seconds`` from their current
``collector_interval_seconds`` (their cadence so far becomes the snap-back
target), and start with ``consecutive_idle_runs = 0``.
"""


def up(conn):
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "memory" not in tables:
        return
    columns = [row[1] for row in conn.execute("PRAGMA table_info(memory)").fetchall()]
    if "base_interval_seconds" not in columns:
        conn.execute("ALTER TABLE memory ADD COLUMN base_interval_seconds INTEGER")
        conn.execute(
            "UPDATE memory SET base_interval_seconds = collector_interval_seconds"
            " WHERE base_interval_seconds IS NULL"
        )
    if "consecutive_idle_runs" not in columns:
        conn.execute(
            "ALTER TABLE memory ADD COLUMN consecutive_idle_runs INTEGER NOT NULL DEFAULT 0"
        )
    conn.commit()

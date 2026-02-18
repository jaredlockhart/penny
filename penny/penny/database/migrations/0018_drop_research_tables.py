"""Drop research_tasks and research_iterations tables.

These tables were used by the /research command and ResearchAgent, which have
been replaced by the /learn command and LearnLoopAgent.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Drop research tables (iterations first due to FK constraint)."""
    conn.execute("DROP TABLE IF EXISTS research_iterations")
    conn.execute("DROP TABLE IF EXISTS research_tasks")
    conn.commit()

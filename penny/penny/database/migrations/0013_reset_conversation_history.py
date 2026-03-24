"""Reset conversation history so it rebuilds from user messages only.

History summaries were dominated by Penny's proactive notifications
(news, thinking output). Now that get_messages_in_range returns only
user messages, delete all existing entries so the backfill rebuilds
them from scratch.
"""


def up(conn):
    # Table may not exist in fresh DBs (created later by SQLModel)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "conversationhistory" in tables:
        conn.execute("DELETE FROM conversationhistory")

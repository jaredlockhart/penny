"""Rebuild promptlog_fts to drop the ``messages`` column.

Type: schema

0051 originally indexed ``messages`` too.  But the input ``messages`` blob is
mostly shared scaffolding (system prompt, recall block, the whole
user-messages log) replicated across every run, so a search for a word that
appears once in the user-messages log matched hundreds of unrelated collector
runs whose input merely embedded it.  The fix is to index only ``response`` +
``thinking`` — what each run actually produced.

0051's file definition was updated to the two-column form for fresh installs,
but instances that already applied the three-column 0051 won't re-run it — so
this migration rebuilds the index in place for them.  It's conditional: if the
index is already two-column (a fresh install), it's a no-op.
"""


def up(conn):
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "promptlog_fts" not in tables:
        return
    columns = [row[1] for row in conn.execute("PRAGMA table_info(promptlog_fts)").fetchall()]
    if "messages" not in columns:
        return  # already the two-column index — nothing to do

    for trigger in ("promptlog_fts_ai", "promptlog_fts_ad", "promptlog_fts_au"):
        conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    conn.execute("DROP TABLE promptlog_fts")

    conn.execute(
        "CREATE VIRTUAL TABLE promptlog_fts USING fts5("
        "response, thinking, content='promptlog', content_rowid='id')"
    )
    conn.execute(
        "CREATE TRIGGER promptlog_fts_ai AFTER INSERT ON promptlog BEGIN "
        "INSERT INTO promptlog_fts(rowid, response, thinking) "
        "VALUES (new.id, new.response, new.thinking); END"
    )
    conn.execute(
        "CREATE TRIGGER promptlog_fts_ad AFTER DELETE ON promptlog BEGIN "
        "INSERT INTO promptlog_fts(promptlog_fts, rowid, response, thinking) "
        "VALUES ('delete', old.id, old.response, old.thinking); END"
    )
    conn.execute(
        "CREATE TRIGGER promptlog_fts_au AFTER UPDATE ON promptlog "
        "WHEN old.response IS NOT new.response OR old.thinking IS NOT new.thinking BEGIN "
        "INSERT INTO promptlog_fts(promptlog_fts, rowid, response, thinking) "
        "VALUES ('delete', old.id, old.response, old.thinking); "
        "INSERT INTO promptlog_fts(rowid, response, thinking) "
        "VALUES (new.id, new.response, new.thinking); END"
    )
    conn.execute("INSERT INTO promptlog_fts(promptlog_fts) VALUES ('rebuild')")
    conn.commit()

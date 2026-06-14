"""Full-text index over promptlog for the addon's prompt search.

Type: schema

A ``LIKE '%text%'`` scan over the (multi-GB) promptlog JSON columns is a full
table scan — no B-tree index can serve a leading-wildcard match — so the
addon's prompt search was very slow.  This adds an FTS5 full-text index and
the triggers that keep it in sync, then rebuilds it from the existing rows.

The index covers ``response`` and ``thinking`` only — NOT ``messages``.  The
input ``messages`` blob is mostly shared scaffolding (system prompt, recall
block, the whole user-messages log) replicated across every run, so indexing
it made searches match that boilerplate: a search for a word that appears once
in the user-messages log returned hundreds of unrelated collector runs whose
input merely embedded it.  ``response`` + ``thinking`` are what each run
actually produced, which is what a prompt search is looking for.

External-content mode (``content='promptlog'``) means the index stores only
the inverted index, not a second copy of the text.  The one-time ``rebuild``
reads every existing prompt once; later inserts are maintained by the AFTER
INSERT trigger.  Run-outcome updates touch only the ``run_*`` columns, so the
UPDATE trigger is guarded to fire only when the indexed text changes.
"""


def up(conn):
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "promptlog" not in tables:
        return
    if "promptlog_fts" in tables:
        return

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
    # Backfill the index from every existing prompt.
    conn.execute("INSERT INTO promptlog_fts(promptlog_fts) VALUES ('rebuild')")
    conn.commit()

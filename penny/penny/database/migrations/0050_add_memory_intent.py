"""Add the ``intent`` column to ``memory``.

Type: schema

``intent`` records the user's expressed goal when a collection was created,
in their own words.  It is the spec a quality collector judges the
``extraction_prompt`` and observed behavior against — set once at creation
(an arg on ``collection_create``, never on ``collection_update``) and NULL
for system / migration-seeded collections.
"""


def up(conn):
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "memory" not in tables:
        return
    columns = [row[1] for row in conn.execute("PRAGMA table_info(memory)").fetchall()]
    if "intent" not in columns:
        conn.execute("ALTER TABLE memory ADD COLUMN intent TEXT")

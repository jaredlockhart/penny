"""Add knowledge table for storing summarized web page content.

Stores prose summaries of browsed pages with embeddings for semantic retrieval.
One entry per URL (upsert on revisit). source_prompt_id tracks extraction progress.
"""


def up(conn):
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }

    if "knowledge" not in tables:
        conn.execute("""
            CREATE TABLE knowledge (
                id INTEGER PRIMARY KEY,
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                embedding BLOB,
                source_prompt_id INTEGER REFERENCES promptlog(id),
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        """)
        conn.execute("CREATE UNIQUE INDEX ix_knowledge_url ON knowledge (url)")
        conn.execute("CREATE INDEX ix_knowledge_source_prompt_id ON knowledge (source_prompt_id)")

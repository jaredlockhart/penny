"""Rename thought.image_url to thought.image.

The column now stores base64 data URIs instead of URLs.
"""


def up(conn):
    columns = [row[1] for row in conn.execute("PRAGMA table_info(thought)").fetchall()]
    if "image_url" in columns and "image" not in columns:
        conn.execute("ALTER TABLE thought RENAME COLUMN image_url TO image")
    elif "image_url" in columns and "image" in columns:
        # Fresh DB created both columns — migrate data and drop old column
        conn.execute(
            "UPDATE thought SET image = image_url WHERE image IS NULL AND image_url IS NOT NULL"
        )
        conn.execute("ALTER TABLE thought DROP COLUMN image_url")

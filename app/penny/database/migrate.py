"""Database migrations for Penny.

SQLite doesn't support full ALTER TABLE, but adding columns works fine.
Each migration is idempotent — safe to run multiple times.

Usage:
    python -m penny.database.migrate          # Run all pending migrations
    docker compose run --rm penny python -m penny.database.migrate
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# Each migration: (name, list of SQL statements)
# Migrations run in order. Each checks whether the change already exists.
MIGRATIONS: list[tuple[str, list[str]]] = [
    (
        "add_reaction_fields",
        [
            "ALTER TABLE messagelog ADD COLUMN is_reaction BOOLEAN DEFAULT 0",
            "ALTER TABLE messagelog ADD COLUMN external_id VARCHAR DEFAULT NULL",
        ],
    ),
]


def _column_exists(cursor: sqlite3.Cursor, table: str, column: str) -> bool:
    cursor.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def migrate(db_path: str) -> None:
    """Run all pending migrations against the given SQLite database."""
    if not Path(db_path).exists():
        logger.info("Database does not exist yet, skipping migrations")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for name, statements in MIGRATIONS:
        applied = False
        for stmt in statements:
            # Extract column name from ALTER TABLE ... ADD COLUMN <name> ...
            parts = stmt.upper().split("ADD COLUMN")
            if len(parts) == 2:
                col_name = parts[1].strip().split()[0].lower()
                table = parts[0].split("ALTER TABLE")[1].strip().lower()
                if _column_exists(cursor, table, col_name):
                    logger.debug("Column %s.%s already exists, skipping", table, col_name)
                    continue

            cursor.execute(stmt)
            applied = True

        if applied:
            conn.commit()
            logger.info("Applied migration: %s", name)
        else:
            logger.debug("Migration already applied: %s", name)

    conn.close()


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    db_path = sys.argv[1] if len(sys.argv) > 1 else "/app/data/penny.db"
    migrate(db_path)

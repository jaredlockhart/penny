"""Database migration runner for Penny.

Discovers and applies numbered migration files from the migrations/ directory.
Each migration runs once and is tracked in the _migrations table.

Usage:
    python -m penny.database.migrate                  # Run pending migrations
    python -m penny.database.migrate --test           # Test against copy of prod DB
    python -m penny.database.migrate --test /path/db  # Test against copy of specific DB
    python -m penny.database.migrate --validate       # Check for duplicate migration numbers
"""

from __future__ import annotations

import importlib.util
import logging
import shutil
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

_MIGRATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS _migrations (
    name TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL
)
"""


def _discover_migrations() -> list[tuple[str, Path]]:
    """Find all migration files, sorted by filename."""
    files = sorted(MIGRATIONS_DIR.glob("[0-9]*.py"))
    return [(f.stem, f) for f in files]


def _get_number_prefix(name: str) -> str:
    """Extract the numeric prefix from a migration name (e.g., '0001' from '0001_add_fields')."""
    return name.split("_", 1)[0]


def validate_migrations() -> None:
    """Check for duplicate migration number prefixes.

    Raises ValueError if any two migration files share the same numeric prefix.
    This enforces the rebase-increment policy: if a migration number conflict
    occurs after rebasing, the branch must renumber its migration.
    """
    migrations = _discover_migrations()
    seen: dict[str, str] = {}

    for name, _path in migrations:
        prefix = _get_number_prefix(name)
        if prefix in seen:
            msg = (
                f"Migration number conflict: {seen[prefix]} and {name} "
                f"share prefix {prefix}. Rebase and renumber the migration."
            )
            raise ValueError(msg)
        seen[prefix] = name

    logger.info("Migration validation passed: %d migration(s), no conflicts", len(migrations))


def _get_applied(conn: sqlite3.Connection) -> set[str]:
    """Get set of already-applied migration names."""
    cursor = conn.execute("SELECT name FROM _migrations")
    return {row[0] for row in cursor.fetchall()}


def _load_module(name: str, path: Path) -> ModuleType:
    """Dynamically import a migration module from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def migrate(db_path: str) -> int:
    """Run all pending migrations against the given SQLite database.

    Returns the number of migrations applied.
    """
    if not Path(db_path).exists():
        logger.info("Database does not exist yet, skipping migrations")
        return 0

    validate_migrations()

    conn = sqlite3.connect(db_path)
    conn.execute(_MIGRATIONS_TABLE_SQL)
    conn.commit()

    applied = _get_applied(conn)
    migrations = _discover_migrations()
    count = 0

    for name, path in migrations:
        if name in applied:
            logger.debug("Migration already applied: %s", name)
            continue

        module = _load_module(name, path)
        if not hasattr(module, "up"):
            logger.warning("Migration %s has no up() function, skipping", name)
            continue

        try:
            conn.execute("BEGIN IMMEDIATE")
            module.up(conn)
            conn.execute(
                "INSERT INTO _migrations (name, applied_at) VALUES (?, ?)",
                (name, datetime.now(UTC).isoformat()),
            )
            conn.commit()
            count += 1
            logger.info("Applied migration: %s", name)
        except Exception:
            conn.rollback()
            logger.exception("Migration failed: %s", name)
            raise
        finally:
            pass

    conn.close()
    return count


def migrate_test(db_path: str) -> bool:
    """Test migrations against a copy of the given database.

    Copies the database to a temp directory, runs all pending migrations,
    and reports success or failure. Cleans up the copy afterward.

    Returns True if successful, False otherwise.
    """
    source = Path(db_path)
    temp_dir = tempfile.mkdtemp()
    test_path = Path(temp_dir) / "migrate_test.db"

    try:
        if source.exists():
            shutil.copy2(source, test_path)
            logger.info("Copied %s to %s for testing", source, test_path)
        else:
            # Create a fresh DB with current schema for testing
            from sqlmodel import SQLModel, create_engine

            import penny.database.models  # noqa: F401 — registers SQLModel tables

            engine = create_engine(f"sqlite:///{test_path}")
            SQLModel.metadata.create_all(engine)
            engine.dispose()
            logger.info("Created fresh database at %s for testing", test_path)

        count = migrate(str(test_path))
        logger.info("Migration test passed: %d migration(s) applied", count)
        return True
    except Exception:
        logger.exception("Migration test FAILED")
        return False
    finally:
        test_path.unlink(missing_ok=True)
        Path(temp_dir).rmdir()


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = sys.argv[1:]

    if "--validate" in args:
        try:
            validate_migrations()
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)
        sys.exit(0)

    if "--test" in args:
        args.remove("--test")
        db_path = args[0] if args else "/penny/data/penny.db"
        success = migrate_test(db_path)
        sys.exit(0 if success else 1)

    db_path = args[0] if args else "/penny/data/penny.db"
    count = migrate(db_path)
    logger.info("Done. %d migration(s) applied.", count)

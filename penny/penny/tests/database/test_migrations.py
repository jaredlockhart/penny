"""Tests for the database migration system."""

import sqlite3
from pathlib import Path

import pytest

from penny.database.migrate import (
    _discover_migrations,
    _get_number_prefix,
    migrate,
    validate_migrations,
)


class TestDiscovery:
    """Tests for migration file discovery."""

    def test_discover_finds_migrations(self):
        migrations = _discover_migrations()
        assert len(migrations) >= 1
        assert migrations[0][0] == "0001_initial_schema"

    def test_discover_returns_sorted(self):
        migrations = _discover_migrations()
        names = [name for name, _path in migrations]
        assert names == sorted(names)

    def test_get_number_prefix(self):
        assert _get_number_prefix("0001_add_fields") == "0001"
        assert _get_number_prefix("0042_something") == "0042"


class TestValidation:
    """Tests for migration number validation."""

    def test_validate_passes_with_no_duplicates(self):
        validate_migrations()

    def test_validate_detects_duplicates(self, tmp_path):
        """Create temp migration files with duplicate prefixes and verify detection."""
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        (migrations_dir / "0001_first.py").write_text(
            "import sqlite3\ndef up(conn: sqlite3.Connection) -> None: pass\n"
        )
        (migrations_dir / "0001_second.py").write_text(
            "import sqlite3\ndef up(conn: sqlite3.Connection) -> None: pass\n"
        )

        # Monkeypatch MIGRATIONS_DIR to use our temp dir
        import penny.database.migrate as mod

        original = mod.MIGRATIONS_DIR
        mod.MIGRATIONS_DIR = migrations_dir
        try:
            with pytest.raises(ValueError, match="Migration number conflict"):
                validate_migrations()
        finally:
            mod.MIGRATIONS_DIR = original


class TestMigrate:
    """Tests for the migration runner."""

    def test_skips_if_db_does_not_exist(self, tmp_path):
        db_path = str(tmp_path / "nonexistent.db")
        count = migrate(db_path)
        assert count == 0
        assert not Path(db_path).exists()

    def test_applies_to_existing_db(self, tmp_path):
        """Migration 0001 should create all tables in a bare database."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        # Create a minimal table so the DB file exists
        conn.execute("CREATE TABLE _bootstrap (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        count = migrate(db_path)
        assert count == 38

        conn = sqlite3.connect(db_path)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master"
                " WHERE type='table' AND name NOT LIKE '\\_%' ESCAPE '\\'"
            ).fetchall()
        }
        expected = {
            "promptlog",
            "messagelog",
            "userinfo",
            "command_logs",
            "runtime_config",
            "schedule",
            "mutestate",
            "thought",
            "preference",
            "device",
        }
        assert expected.issubset(tables)
        # entity and fact tables should NOT exist (dropped by 0004)
        assert "entity" not in tables
        assert "fact" not in tables
        # conversationhistory should NOT exist (dropped by 0024)
        assert "conversationhistory" not in tables
        conn.close()

    def test_idempotent(self, tmp_path):
        """Running migrate twice should not fail or re-apply."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE _bootstrap (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        count1 = migrate(db_path)
        count2 = migrate(db_path)
        assert count1 == 38
        assert count2 == 0

    def test_tracks_in_migrations_table(self, tmp_path):
        """Applied migrations should be recorded in _migrations."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE _bootstrap (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        migrate(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT name FROM _migrations")
        applied = {row[0] for row in cursor.fetchall()}
        assert "0001_initial_schema" in applied
        conn.close()

    def test_skips_already_applied(self, tmp_path):
        """If _migrations already records a migration, it should not be re-run."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE _bootstrap (id INTEGER PRIMARY KEY)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _migrations (
                name TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
        """)
        conn.execute(
            "INSERT INTO _migrations (name, applied_at) VALUES (?, ?)",
            ("0001_initial_schema", "2025-01-01T00:00:00"),
        )
        conn.commit()
        conn.close()

        count = migrate(db_path)
        # 0001 is skipped; 0002 through 0038 run = 37 migrations
        assert count == 37

    def test_bootstrap_with_tables_already_present(self, tmp_path):
        """If tables already exist (from SQLModel.create_tables), migration should succeed."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        # Simulate a table already created by SQLModel.create_tables() with full schema
        conn.execute("""
            CREATE TABLE messagelog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP NOT NULL,
                direction TEXT NOT NULL,
                sender TEXT NOT NULL,
                content TEXT NOT NULL,
                parent_id INTEGER REFERENCES messagelog(id),
                signal_timestamp INTEGER,
                recipient TEXT,
                external_id TEXT,
                is_reaction BOOLEAN NOT NULL DEFAULT 0,
                processed BOOLEAN NOT NULL DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()

        count = migrate(db_path)
        assert count == 38  # all migrations applied

        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT name FROM _migrations")
        applied = {row[0] for row in cursor.fetchall()}
        assert "0001_initial_schema" in applied
        conn.close()

    def test_0037_fixes_knowledge_extraction_prompt(self, tmp_path):
        """Migration 0037 replaces collection_update with update_entry in the
        knowledge extraction_prompt for databases seeded by migration 0031."""
        import importlib.util
        from pathlib import Path

        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE memory (name TEXT PRIMARY KEY, extraction_prompt TEXT)")
        broken_prompt = (
            'call collection_update("knowledge", key=<title>, content=<merged paragraph>)'
        )
        conn.execute(
            "INSERT INTO memory (name, extraction_prompt) VALUES (?, ?)",
            ("knowledge", broken_prompt),
        )
        conn.commit()
        conn.close()

        migration_path = (
            Path(__file__).parents[3]
            / "penny"
            / "database"
            / "migrations"
            / "0037_fix_knowledge_extraction_prompt.py"
        )
        spec = importlib.util.spec_from_file_location("m0037", migration_path)
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

        conn = sqlite3.connect(db_path)
        mod.up(conn)
        conn.close()

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT extraction_prompt FROM memory WHERE name = 'knowledge'"
        ).fetchone()
        assert row is not None
        prompt = row[0]
        assert "collection_update" not in prompt
        assert 'update_entry("knowledge", key=<title>,' in prompt
        conn.close()

    def test_0038_fixes_browse_in_thinking_prompt(self, tmp_path):
        """Migration 0038 replaces ambiguous 'browse —' with explicit browse(queries=[...])
        in the unnotified-thoughts extraction_prompt seeded by migration 0033."""
        import importlib.util
        from pathlib import Path

        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE memory (name TEXT PRIMARY KEY, extraction_prompt TEXT)")
        old_prompt = (
            "Sequence:\n"
            '1. collection_read_random("likes", 1) — pick one seed topic.\n'
            '2. read_latest("dislikes") — see dislikes.\n'
            "3. browse — search the web and read one or two pages to find something.\n"
            "4. Draft a thought.\n"
            "8. done().\n"
        )
        conn.execute(
            "INSERT INTO memory (name, extraction_prompt) VALUES (?, ?)",
            ("unnotified-thoughts", old_prompt),
        )
        conn.commit()
        conn.close()

        migration_path = (
            Path(__file__).parents[3]
            / "penny"
            / "database"
            / "migrations"
            / "0038_fix_thinking_prompt_browse_tool_name.py"
        )
        spec = importlib.util.spec_from_file_location("m0038", migration_path)
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

        conn = sqlite3.connect(db_path)
        mod.up(conn)
        conn.close()

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT extraction_prompt FROM memory WHERE name = 'unnotified-thoughts'"
        ).fetchone()
        assert row is not None
        prompt = row[0]
        # Old ambiguous form replaced
        assert "3. browse — search the web" not in prompt
        # New explicit function-call form present
        assert 'browse(queries=["<seed topic>"]) — search the web' in prompt
        # Other steps unchanged
        assert 'collection_read_random("likes", 1)' in prompt
        assert "done()" in prompt
        conn.close()

    def test_0038_is_idempotent_on_already_fixed_prompt(self, tmp_path):
        """Migration 0038 is a no-op when the prompt has already been corrected."""
        import importlib.util
        from pathlib import Path

        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE memory (name TEXT PRIMARY KEY, extraction_prompt TEXT)")
        already_fixed = (
            '3. browse(queries=["<seed topic>"]) — search the web and read one or two pages.\n'
        )
        conn.execute(
            "INSERT INTO memory (name, extraction_prompt) VALUES (?, ?)",
            ("unnotified-thoughts", already_fixed),
        )
        conn.commit()
        conn.close()

        migration_path = (
            Path(__file__).parents[3]
            / "penny"
            / "database"
            / "migrations"
            / "0038_fix_thinking_prompt_browse_tool_name.py"
        )
        spec = importlib.util.spec_from_file_location("m0038b", migration_path)
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

        conn = sqlite3.connect(db_path)
        mod.up(conn)
        conn.close()

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT extraction_prompt FROM memory WHERE name = 'unnotified-thoughts'"
        ).fetchone()
        assert row is not None
        assert row[0] == already_fixed
        conn.close()

    def test_0038_does_not_touch_other_collections(self, tmp_path):
        """Migration 0038 only updates unnotified-thoughts, not other collections."""
        import importlib.util
        from pathlib import Path

        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE memory (name TEXT PRIMARY KEY, extraction_prompt TEXT)")
        other_prompt = "3. browse — search the web for other stuff.\n"
        conn.executemany(
            "INSERT INTO memory (name, extraction_prompt) VALUES (?, ?)",
            [
                ("unnotified-thoughts", "3. browse — search the web and read pages.\n"),
                ("other-collection", other_prompt),
            ],
        )
        conn.commit()
        conn.close()

        migration_path = (
            Path(__file__).parents[3]
            / "penny"
            / "database"
            / "migrations"
            / "0038_fix_thinking_prompt_browse_tool_name.py"
        )
        spec = importlib.util.spec_from_file_location("m0038c", migration_path)
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

        conn = sqlite3.connect(db_path)
        mod.up(conn)
        conn.close()

        conn = sqlite3.connect(db_path)
        rows = {
            row[0]: row[1]
            for row in conn.execute("SELECT name, extraction_prompt FROM memory").fetchall()
        }
        conn.close()

        # unnotified-thoughts should be updated
        assert "3. browse — search the web" not in rows["unnotified-thoughts"]
        assert 'browse(queries=["<seed topic>"])' in rows["unnotified-thoughts"]
        # other-collection should be unchanged
        assert rows["other-collection"] == other_prompt

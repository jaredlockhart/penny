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
        assert migrations[0][0] == "0001_add_reaction_fields"

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
        """Migration 0001 should add columns to an existing messagelog table."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            """CREATE TABLE messagelog (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                direction TEXT,
                sender TEXT,
                content TEXT,
                parent_id INTEGER,
                parent_summary TEXT,
                signal_timestamp INTEGER
            )"""
        )
        conn.commit()
        conn.close()

        count = migrate(db_path)
        assert count == 28  # 0001 through 0028

        conn = sqlite3.connect(db_path)
        cursor = conn.execute("PRAGMA table_info(messagelog)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "is_reaction" in columns
        assert "external_id" in columns
        assert "processed" in columns
        assert "parent_summary" not in columns  # Should be removed by migration 0008

        # Verify learnprompt table created by 0019
        has_learnprompt = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='learnprompt'"
        ).fetchone()
        assert has_learnprompt is not None
        conn.close()

    def test_idempotent(self, tmp_path):
        """Running migrate twice should not fail or re-apply."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE messagelog (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        count1 = migrate(db_path)
        count2 = migrate(db_path)
        assert count1 >= 1
        assert count2 == 0

    def test_tracks_in_migrations_table(self, tmp_path):
        """Applied migrations should be recorded in _migrations."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE messagelog (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        migrate(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT name FROM _migrations")
        applied = {row[0] for row in cursor.fetchall()}
        assert "0001_add_reaction_fields" in applied
        conn.close()

    def test_skips_already_applied(self, tmp_path):
        """If _migrations already records a migration, it should not be re-run."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE messagelog (id INTEGER PRIMARY KEY)")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS _migrations (
                name TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            "INSERT INTO _migrations (name, applied_at) VALUES (?, ?)",
            ("0001_add_reaction_fields", "2025-01-01T00:00:00"),
        )
        conn.commit()
        conn.close()

        count = migrate(db_path)
        assert count == 27  # 0002 through 0028 are applied

    def test_bootstrap_with_columns_already_present(self, tmp_path):
        """If columns already exist (from old migration system), 0001 should succeed."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            """CREATE TABLE messagelog (
                id INTEGER PRIMARY KEY,
                is_reaction BOOLEAN DEFAULT 0,
                external_id VARCHAR DEFAULT NULL
            )"""
        )
        conn.commit()
        conn.close()

        count = migrate(db_path)
        assert count == 28  # All migrations (0001 through 0028) recorded as applied

        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT name FROM _migrations")
        applied = {row[0] for row in cursor.fetchall()}
        assert "0001_add_reaction_fields" in applied
        assert "0002_add_runtime_config_table" in applied
        assert "0003_split_user_profiles" in applied
        assert "0004_add_preference_table" in applied
        assert "0005_add_reaction_processed_field" in applied
        assert "0006_reset_message_processed_flags" in applied
        assert "0007_add_schedule_table" in applied
        assert "0008_drop_parent_summary" in applied
        assert "0010_add_research_focus" in applied
        assert "0011_add_research_options" in applied
        assert "0018_drop_research_tables" in applied
        conn.close()

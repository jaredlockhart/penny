"""Tests for CODEOWNERS parsing."""

from __future__ import annotations

from penny_team.utils.codeowners import parse_codeowners


class TestParseCODEOWNERS:
    def test_parse_from_github_dir(self, tmp_path):
        """Standard .github/CODEOWNERS location."""
        github_dir = tmp_path / ".github"
        github_dir.mkdir()
        (github_dir / "CODEOWNERS").write_text("* @alice @bob\n")

        assert parse_codeowners(tmp_path) == {"alice", "bob"}

    def test_parse_from_root(self, tmp_path):
        """Fallback to root CODEOWNERS."""
        (tmp_path / "CODEOWNERS").write_text("* @charlie\n")

        assert parse_codeowners(tmp_path) == {"charlie"}

    def test_parse_from_docs(self, tmp_path):
        """Fallback to docs/CODEOWNERS."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "CODEOWNERS").write_text("* @dave\n")

        assert parse_codeowners(tmp_path) == {"dave"}

    def test_github_dir_takes_priority(self, tmp_path):
        """.github/CODEOWNERS is checked first."""
        github_dir = tmp_path / ".github"
        github_dir.mkdir()
        (github_dir / "CODEOWNERS").write_text("* @alice\n")
        (tmp_path / "CODEOWNERS").write_text("* @bob\n")

        assert parse_codeowners(tmp_path) == {"alice"}

    def test_no_codeowners_returns_empty(self, tmp_path):
        """Missing CODEOWNERS returns empty set."""
        assert parse_codeowners(tmp_path) == set()

    def test_ignores_comments_and_blanks(self, tmp_path):
        github_dir = tmp_path / ".github"
        github_dir.mkdir()
        (github_dir / "CODEOWNERS").write_text(
            "# This is a comment\n"
            "\n"
            "*.py @alice\n"
            "  # Another comment\n"
            "*.js @bob\n"
        )

        assert parse_codeowners(tmp_path) == {"alice", "bob"}

    def test_ignores_org_team_tokens(self, tmp_path):
        """@org/team references (with slash) are skipped."""
        github_dir = tmp_path / ".github"
        github_dir.mkdir()
        (github_dir / "CODEOWNERS").write_text("* @alice @myorg/myteam @bob\n")

        assert parse_codeowners(tmp_path) == {"alice", "bob"}

    def test_multiple_users_per_line(self, tmp_path):
        github_dir = tmp_path / ".github"
        github_dir.mkdir()
        (github_dir / "CODEOWNERS").write_text(
            "*.py @alice @bob @charlie\n" "*.js @dave @eve\n"
        )

        assert parse_codeowners(tmp_path) == {"alice", "bob", "charlie", "dave", "eve"}

    def test_deduplication(self, tmp_path):
        """Same user on multiple lines is deduplicated."""
        github_dir = tmp_path / ".github"
        github_dir.mkdir()
        (github_dir / "CODEOWNERS").write_text("*.py @alice\n" "*.js @alice @bob\n")

        assert parse_codeowners(tmp_path) == {"alice", "bob"}

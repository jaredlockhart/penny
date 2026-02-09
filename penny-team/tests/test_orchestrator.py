"""Tests for the orchestrator module."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from penny_team.orchestrator import get_agents, load_github_app, save_agent_log


class TestGetAgents:
    def test_returns_three_agents(self, project_root, monkeypatch):
        monkeypatch.setattr("penny_team.orchestrator.PROJECT_ROOT", project_root)
        agents = get_agents()
        assert len(agents) == 3

    def test_agent_names(self, project_root, monkeypatch):
        monkeypatch.setattr("penny_team.orchestrator.PROJECT_ROOT", project_root)
        agents = get_agents()
        names = {a.name for a in agents}
        assert names == {"product-manager", "architect", "worker"}

    def test_agent_labels(self, project_root, monkeypatch):
        monkeypatch.setattr("penny_team.orchestrator.PROJECT_ROOT", project_root)
        agents = get_agents()
        labels_by_name = {a.name: a.required_labels for a in agents}
        assert labels_by_name["product-manager"] == ["requirements"]
        assert labels_by_name["architect"] == ["specification"]
        assert labels_by_name["worker"] == ["in-progress", "in-review", "bug"]

    def test_agent_intervals(self, project_root, monkeypatch):
        monkeypatch.setattr("penny_team.orchestrator.PROJECT_ROOT", project_root)
        agents = get_agents()
        intervals = {a.name: a.interval_seconds for a in agents}
        assert intervals["product-manager"] == 300
        assert intervals["architect"] == 300
        assert intervals["worker"] == 300

    def test_worker_has_longer_timeout(self, project_root, monkeypatch):
        monkeypatch.setattr("penny_team.orchestrator.PROJECT_ROOT", project_root)
        agents = get_agents()
        timeouts = {a.name: a.timeout_seconds for a in agents}
        assert timeouts["worker"] == 1800
        assert timeouts["product-manager"] == 600

    def test_trusted_users_from_codeowners(self, project_root, monkeypatch):
        monkeypatch.setattr("penny_team.orchestrator.PROJECT_ROOT", project_root)
        agents = get_agents()
        # All agents share the same trusted_users parsed from CODEOWNERS
        for agent in agents:
            assert agent.trusted_users == {"alice", "bob"}

    def test_no_codeowners_sets_none(self, tmp_path, monkeypatch):
        """Without CODEOWNERS, trusted_users should be None (no filtering)."""
        monkeypatch.setattr("penny_team.orchestrator.PROJECT_ROOT", tmp_path)
        agents = get_agents()
        for agent in agents:
            assert agent.trusted_users is None

    def test_trusted_users_includes_both_bot_login_forms(
        self, project_root, monkeypatch, mock_github_app
    ):
        """When github_app is configured, both bot login forms are trusted.

        Bug fix: GitHub API returns bot author as both "slug" (e.g. "penny-team")
        and "slug[bot]" (e.g. "penny-team[bot]") depending on context. Both forms
        must be in trusted_users so bot-authored comments aren't filtered out.
        """
        monkeypatch.setattr("penny_team.orchestrator.PROJECT_ROOT", project_root)
        agents = get_agents(mock_github_app)
        for agent in agents:
            assert "penny-team" in agent.trusted_users  # slug form
            assert "penny-team[bot]" in agent.trusted_users  # [bot] suffix form
            # CODEOWNERS users should still be there too
            assert "alice" in agent.trusted_users
            assert "bob" in agent.trusted_users


class TestSaveAgentLog:
    def test_writes_log_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("penny_team.orchestrator.LOG_DIR", tmp_path)

        save_agent_log(
            agent_name="test-agent",
            run_number=1,
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            duration=42.5,
            success=True,
            output="Agent completed successfully",
        )

        log_path = tmp_path / "test-agent.log"
        assert log_path.exists()
        content = log_path.read_text()
        assert "Run #1" in content
        assert "2024-01-15 10:30:00" in content
        assert "42.5s" in content
        assert "Success: True" in content
        assert "Agent completed successfully" in content

    def test_appends_to_existing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("penny_team.orchestrator.LOG_DIR", tmp_path)

        for i in range(2):
            save_agent_log("test-agent", i + 1, datetime.now(), 1.0, True, f"Run {i + 1}")

        content = (tmp_path / "test-agent.log").read_text()
        assert "Run #1" in content
        assert "Run #2" in content


class TestLoadGitHubApp:
    def test_returns_none_when_not_configured(self, monkeypatch):
        monkeypatch.delenv("GITHUB_APP_ID", raising=False)
        monkeypatch.delenv("GITHUB_APP_PRIVATE_KEY_PATH", raising=False)
        monkeypatch.delenv("GITHUB_APP_INSTALLATION_ID", raising=False)

        assert load_github_app() is None

    def test_returns_none_with_partial_config(self, monkeypatch):
        monkeypatch.setenv("GITHUB_APP_ID", "12345")
        monkeypatch.delenv("GITHUB_APP_PRIVATE_KEY_PATH", raising=False)
        monkeypatch.delenv("GITHUB_APP_INSTALLATION_ID", raising=False)

        assert load_github_app() is None

    def test_returns_app_with_full_config(self, tmp_path, monkeypatch):
        key_path = tmp_path / "key.pem"
        key_path.write_text("fake-key")
        monkeypatch.setattr("penny_team.orchestrator.PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("GITHUB_APP_ID", "12345")
        monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY_PATH", str(key_path))
        monkeypatch.setenv("GITHUB_APP_INSTALLATION_ID", "67890")

        app = load_github_app()
        assert app is not None
        assert app.app_id == 12345
        assert app.installation_id == 67890

"""Tests for the Agent base class."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from penny_team.base import Agent


def _make_agent(
    tmp_path: Path,
    name: str = "test-agent",
    interval: int = 300,
    required_labels: list[str] | None = None,
    github_app=None,
    trusted_users: set[str] | None = None,
) -> Agent:
    """Create an Agent with a temporary prompt file and data directory."""
    # Create prompt file
    agent_dir = tmp_path / "penny_team" / name
    agent_dir.mkdir(parents=True)
    (agent_dir / "CLAUDE.md").write_text("Test prompt for agent.")

    agent = Agent(
        name=name,
        interval_seconds=interval,
        working_dir=tmp_path,
        required_labels=required_labels,
        github_app=github_app,
        trusted_users=trusted_users,
    )
    # Override paths to use tmp_path
    agent.prompt_path = agent_dir / "CLAUDE.md"
    return agent


# --- is_due ---


class TestIsDue:
    def test_first_run_is_due(self, tmp_path):
        agent = _make_agent(tmp_path)
        assert agent.is_due() is True

    def test_not_due_before_interval(self, tmp_path):
        agent = _make_agent(tmp_path, interval=300)
        agent.last_run = datetime.now()
        assert agent.is_due() is False

    def test_due_after_interval(self, tmp_path):
        agent = _make_agent(tmp_path, interval=300)
        agent.last_run = datetime.now() - timedelta(seconds=301)
        assert agent.is_due() is True


# --- _load_state / _save_state ---


class TestState:
    def test_round_trip(self, tmp_path, monkeypatch):
        agent = _make_agent(tmp_path)
        monkeypatch.setattr(
            type(agent), "_state_path", property(lambda self: tmp_path / "state.json")
        )

        timestamps = {"1": "2024-01-01T00:00:00Z", "2": "2024-01-02T00:00:00Z"}
        agent._save_state(timestamps)
        assert agent._load_state() == timestamps

    def test_missing_file_returns_empty(self, tmp_path, monkeypatch):
        agent = _make_agent(tmp_path)
        monkeypatch.setattr(
            type(agent),
            "_state_path",
            property(lambda self: tmp_path / "nonexistent.json"),
        )
        assert agent._load_state() == {}

    def test_corrupt_json_returns_empty(self, tmp_path, monkeypatch):
        state_path = tmp_path / "corrupt.json"
        state_path.write_text("not json{{{")
        agent = _make_agent(tmp_path)
        monkeypatch.setattr(type(agent), "_state_path", property(lambda self: state_path))
        assert agent._load_state() == {}


# --- _build_command ---


class TestBuildCommand:
    def test_basic_command(self, tmp_path):
        agent = _make_agent(tmp_path)
        cmd = agent._build_command("hello world")

        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "hello world" in cmd
        assert "--dangerously-skip-permissions" in cmd
        assert "--verbose" in cmd
        assert "stream-json" in cmd

    def test_with_model(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.model = "claude-sonnet-4-5-20250929"
        cmd = agent._build_command("test")

        assert "--model" in cmd
        assert "claude-sonnet-4-5-20250929" in cmd

    def test_with_allowed_tools(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.allowed_tools = ["Bash", "Read"]
        cmd = agent._build_command("test")

        assert "--allowedTools" in cmd
        assert "Bash" in cmd
        assert "Read" in cmd


# --- has_work ---


class TestHasWork:
    def test_no_labels_always_has_work(self, tmp_path):
        agent = _make_agent(tmp_path, required_labels=None)
        assert agent.has_work() is True

    def test_empty_labels_always_has_work(self, tmp_path):
        agent = _make_agent(tmp_path, required_labels=[])
        assert agent.has_work() is True

    def test_no_issues_returns_false(self, tmp_path, mock_subprocess):
        agent = _make_agent(tmp_path, required_labels=["requirements"])
        mock_subprocess.add_response("issue list", stdout="[]")

        assert agent.has_work() is False

    def test_new_issue_returns_true(self, tmp_path, mock_subprocess, monkeypatch):
        agent = _make_agent(tmp_path, required_labels=["requirements"])
        monkeypatch.setattr(
            type(agent), "_state_path", property(lambda self: tmp_path / "state.json")
        )
        # No saved state, one issue exists
        mock_subprocess.add_response(
            "issue list",
            stdout=json.dumps([{"number": 1, "updatedAt": "2024-01-01T00:00:00Z"}]),
        )

        assert agent.has_work() is True

    def test_unchanged_issues_returns_false(self, tmp_path, mock_subprocess, monkeypatch):
        agent = _make_agent(tmp_path, required_labels=["requirements"])
        state_path = tmp_path / "state.json"
        monkeypatch.setattr(type(agent), "_state_path", property(lambda self: state_path))

        timestamps = {"1": "2024-01-01T00:00:00Z"}
        state_path.write_text(json.dumps(timestamps))

        mock_subprocess.add_response(
            "issue list",
            stdout=json.dumps([{"number": 1, "updatedAt": "2024-01-01T00:00:00Z"}]),
        )

        assert agent.has_work() is False

    def test_gh_failure_returns_true(self, tmp_path, mock_subprocess):
        """Fail-open: if gh fails, assume there's work."""
        agent = _make_agent(tmp_path, required_labels=["requirements"])
        mock_subprocess.add_response("issue list", returncode=1, stderr="auth error")

        assert agent.has_work() is True


# --- run ---


class TestRun:
    def test_timeout_kills_process(self, tmp_path, mock_subprocess, mock_popen, monkeypatch):
        import subprocess as sp

        agent = _make_agent(tmp_path, required_labels=None)
        agent.timeout_seconds = 0

        # Make Popen.wait() raise TimeoutExpired
        popen_instance = mock_popen(stdout_lines=[], returncode=0)
        popen_instance.wait = lambda timeout=None: (_ for _ in ()).throw(
            sp.TimeoutExpired(cmd="claude", timeout=0)
        )

        result = agent.run()
        assert result.success is False
        assert "timed out" in result.output.lower()

"""Shared test fixtures and helpers for penny-team tests.

Provides subprocess mocking for gh CLI and Claude CLI interactions,
agent factories, and data builders used across all test files.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from penny_team.base import Agent

# Ensure penny-team package is importable (matches PYTHONPATH in Dockerfile)
PENNY_TEAM_ROOT = Path(__file__).parent.parent
if str(PENNY_TEAM_ROOT) not in sys.path:
    sys.path.insert(0, str(PENNY_TEAM_ROOT))

# --- Constants ---

CODEOWNERS_CONTENT = "* @alice @bob\n"

BOT_SLUG = "penny-team"
BOT_LOGIN = "penny-team[bot]"
BOT_LOGINS = {BOT_SLUG, BOT_LOGIN}
TRUSTED_USERS = {"alice", "bob", BOT_SLUG, BOT_LOGIN}
CODEOWNERS_USERS = {"alice", "bob"}


# --- Agent factory ---


def make_agent(
    tmp_path: Path,
    name: str = "test-agent",
    required_labels: list[str] | None = None,
    interval: int = 300,
    timeout: int = 600,
    model: str | None = None,
    allowed_tools: list[str] | None = None,
    github_app: MagicMock | None = None,
    trusted_users: set[str] | None = TRUSTED_USERS,
) -> Agent:
    """Create an agent with a temporary prompt file for integration testing."""
    agent_dir = tmp_path / "penny_team" / name
    agent_dir.mkdir(parents=True, exist_ok=True)
    prompt_marker = f"# {name.title().replace('-', ' ')} Agent Prompt"
    (agent_dir / "CLAUDE.md").write_text(f"{prompt_marker}\n\nYou are the {name} agent.\n")

    agent = Agent(
        name=name,
        interval_seconds=interval,
        working_dir=tmp_path,
        timeout_seconds=timeout,
        model=model,
        allowed_tools=allowed_tools,
        required_labels=required_labels,
        github_app=github_app,
        trusted_users=trusted_users,
    )
    agent.prompt_path = agent_dir / "CLAUDE.md"
    return agent


# --- Data builders ---


def result_event(text: str = "Task completed") -> str:
    """Create a stream-json result event line."""
    return json.dumps({"type": "result", "result": text})


def issue_list_response(*numbers: int) -> str:
    """Create a gh issue list JSON response."""
    return json.dumps([{"number": n} for n in numbers])


def issue_view_response(
    number: int = 42,
    title: str = "Add reminders feature",
    body: str = "Users should be able to set reminders via natural language.",
    author: str = "alice",
    labels: list[str] | None = None,
    comments: list[dict] | None = None,
) -> str:
    """Create a gh issue view JSON response."""
    return json.dumps({
        "title": title,
        "body": body,
        "author": {"login": author},
        "labels": [{"name": label} for label in (labels or ["requirements"])],
        "comments": comments or [],
    })


def make_pr_response(
    number: int,
    branch: str,
    checks: list[dict] | None = None,
    mergeable: str = "MERGEABLE",
    reviews: list[dict] | None = None,
) -> dict:
    """Create a PR data dict matching gh pr list --json output."""
    return {
        "number": number,
        "headRefName": branch,
        "statusCheckRollup": checks or [],
        "mergeable": mergeable,
        "reviews": reviews or [],
    }


# --- Prompt extraction ---


def extract_prompt(calls: list[tuple[tuple, dict]]) -> str:
    """Extract the prompt string from captured Popen calls."""
    assert calls, "Expected Popen to be called, but it was not"
    cmd = calls[0][0][0]  # First call, positional args, first arg (command list)
    p_index = cmd.index("-p")
    return cmd[p_index + 1]


# --- Mock classes ---


class MockSubprocess:
    """Intercepts subprocess.run() calls with canned responses by command pattern.

    Register responses with add_response(pattern, ...). When subprocess.run()
    is called, the command is joined into a string and matched against
    registered patterns. First match wins. Unmatched commands return
    returncode=0, stdout="[]".
    """

    def __init__(self):
        self.calls: list[tuple[list[str], dict]] = []
        self._responses: list[tuple[str, subprocess.CompletedProcess]] = []

    def add_response(
        self,
        pattern: str,
        stdout: str = "",
        returncode: int = 0,
        stderr: str = "",
    ) -> None:
        """Register a canned response for commands containing pattern."""
        self._responses.append(
            (
                pattern,
                subprocess.CompletedProcess(
                    args=[],
                    returncode=returncode,
                    stdout=stdout,
                    stderr=stderr,
                ),
            )
        )

    def __call__(self, cmd, **kwargs):
        self.calls.append((list(cmd), kwargs))
        cmd_str = " ".join(str(c) for c in cmd)
        for pattern, response in self._responses:
            if pattern in cmd_str:
                return response
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="[]")


class MockPopen:
    """Mock subprocess.Popen for Claude CLI stream-json output.

    Provides iterable stdout yielding JSON event lines,
    and standard process control methods.
    """

    def __init__(self, stdout_lines: list[str] | None = None, returncode: int = 0):
        self.stdout = iter(line + "\n" for line in (stdout_lines or []))
        self.returncode = returncode
        self.pid = 12345

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        pass

    def terminate(self):
        pass

    def poll(self):
        return self.returncode


# --- Fixtures ---


@pytest.fixture(autouse=True)
def isolate_state_dir(tmp_path, monkeypatch):
    """Isolate agent state files to tmp_path for all tests.

    Prevents _mark_processed and _save_state from writing to the real
    data directory, which would leak state between tests.
    """
    monkeypatch.setattr("penny_team.base.DATA_DIR", tmp_path)


@pytest.fixture
def mock_subprocess(monkeypatch):
    """Monkeypatch subprocess.run with a MockSubprocess instance.

    Returns the MockSubprocess so tests can register responses and inspect calls.
    """
    mock = MockSubprocess()
    monkeypatch.setattr(subprocess, "run", mock)
    return mock


@pytest.fixture
def mock_popen(monkeypatch):
    """Provide a factory to create MockPopen instances and monkeypatch subprocess.Popen.

    Usage:
        popen = mock_popen(stdout_lines=['{"type":"result","result":"done"}'])
        # Now subprocess.Popen() returns the mock
    """
    mock_instance = None

    def factory(stdout_lines=None, returncode=0):
        nonlocal mock_instance
        mock_instance = MockPopen(stdout_lines=stdout_lines, returncode=returncode)
        monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: mock_instance)
        return mock_instance

    return factory


@pytest.fixture
def capture_popen(monkeypatch):
    """Mock Popen that captures call args and returns canned stream-json output.

    Usage:
        calls = capture_popen(stdout_lines=['{"type":"result","result":"done"}'])
        agent.run()
        cmd = calls[0][0][0]  # First call, positional args, first arg (the command list)
        prompt = cmd[cmd.index("-p") + 1]
    """
    calls: list[tuple[tuple, dict]] = []

    def factory(stdout_lines=None, returncode=0):
        def popen_spy(*args, **kwargs):
            calls.append((args, kwargs))
            return MockPopen(stdout_lines=stdout_lines, returncode=returncode)

        monkeypatch.setattr(subprocess, "Popen", popen_spy)
        return calls

    return factory


@pytest.fixture
def project_root(tmp_path):
    """Create a temporary project root with a .github/CODEOWNERS file."""
    github_dir = tmp_path / ".github"
    github_dir.mkdir()
    (github_dir / "CODEOWNERS").write_text(CODEOWNERS_CONTENT)
    return tmp_path


@pytest.fixture
def mock_github_app():
    """Create a mock GitHubApp that doesn't make real API calls."""
    app = MagicMock()
    app.app_id = 12345
    app.installation_id = 67890
    app._fetch_slug.return_value = "penny-team"
    app.bot_name = "penny-team[bot]"
    app.bot_email = "12345+penny-team[bot]@users.noreply.github.com"
    app.get_token.return_value = "ghs_fake_token"
    app.get_env.return_value = {
        "GH_TOKEN": "ghs_fake_token",
        "GIT_AUTHOR_NAME": "penny-team[bot]",
        "GIT_AUTHOR_EMAIL": "12345+penny-team[bot]@users.noreply.github.com",
        "GIT_COMMITTER_NAME": "penny-team[bot]",
        "GIT_COMMITTER_EMAIL": "12345+penny-team[bot]@users.noreply.github.com",
    }
    return app

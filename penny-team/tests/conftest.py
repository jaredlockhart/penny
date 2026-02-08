"""Shared test fixtures for penny-team tests.

Provides subprocess mocking for gh CLI and Claude CLI interactions,
plus common test data fixtures.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure penny-team package is importable (matches PYTHONPATH in Dockerfile)
PENNY_TEAM_ROOT = Path(__file__).parent.parent
if str(PENNY_TEAM_ROOT) not in sys.path:
    sys.path.insert(0, str(PENNY_TEAM_ROOT))

# Test CODEOWNERS content
CODEOWNERS_CONTENT = "* @alice @bob\n"
TRUSTED_USERS = {"alice", "bob"}


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

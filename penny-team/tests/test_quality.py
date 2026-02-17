"""Tests for the Quality agent.

The Quality agent reads message pairs from Penny's database, evaluates
response quality using Ollama, and files bug issues for bad responses.
Unlike other agents, it reads from SQLite and calls Ollama directly
(no Claude CLI).

Unit tests cover privacy validation and data reading.
Integration tests verify the full flow through has_work() and run().
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

from penny_team.quality import QualityAgent, validate_privacy

from tests.conftest import (
    MockGitHubAPI,
    TRUSTED_USERS,
)


# =============================================================================
# Helper: create a QualityAgent with an in-memory test database
# =============================================================================


def _create_test_db(db_path: Path) -> None:
    """Create a minimal penny database with messagelog and promptlog tables."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE messagelog ("
        "  id INTEGER PRIMARY KEY,"
        "  timestamp TEXT NOT NULL,"
        "  direction TEXT NOT NULL,"
        "  sender TEXT NOT NULL,"
        "  content TEXT NOT NULL,"
        "  parent_id INTEGER,"
        "  signal_timestamp INTEGER,"
        "  external_id TEXT,"
        "  is_reaction INTEGER DEFAULT 0,"
        "  processed INTEGER DEFAULT 0"
        ")"
    )
    conn.execute(
        "CREATE TABLE promptlog ("
        "  id INTEGER PRIMARY KEY,"
        "  timestamp TEXT NOT NULL,"
        "  model TEXT NOT NULL,"
        "  messages TEXT NOT NULL,"
        "  tools TEXT,"
        "  response TEXT NOT NULL,"
        "  thinking TEXT,"
        "  duration_ms INTEGER"
        ")"
    )
    conn.commit()
    conn.close()


def _insert_message_pair(
    db_path: Path,
    user_message: str = "What is the weather?",
    response: str = "The weather is sunny today.",
    sender: str = "+1234567890",
    timestamp: datetime | None = None,
) -> tuple[int, int]:
    """Insert an incoming/outgoing message pair and return (incoming_id, outgoing_id)."""
    ts = timestamp or datetime.now()
    ts_str = ts.isoformat()
    response_ts = (ts + timedelta(seconds=1)).isoformat()

    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(
        "INSERT INTO messagelog (timestamp, direction, sender, content) VALUES (?, 'incoming', ?, ?)",
        (ts_str, sender, user_message),
    )
    incoming_id = cursor.lastrowid
    cursor = conn.execute(
        "INSERT INTO messagelog (timestamp, direction, sender, content, parent_id) "
        "VALUES (?, 'outgoing', 'agent', ?, ?)",
        (response_ts, response, incoming_id),
    )
    outgoing_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return incoming_id, outgoing_id


def _insert_prompt_log(
    db_path: Path,
    timestamp: datetime | None = None,
    thinking: str | None = "I should check the weather",
) -> None:
    """Insert a prompt log entry."""
    ts = (timestamp or datetime.now()).isoformat()
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO promptlog (timestamp, model, messages, response, thinking) "
        "VALUES (?, 'test-model', '[]', '{}', ?)",
        (ts, thinking),
    )
    conn.commit()
    conn.close()


def make_quality_agent(
    tmp_path: Path,
    create_db: bool = True,
    github_api: MockGitHubAPI | None = None,
) -> tuple[QualityAgent, Path]:
    """Create a QualityAgent with a temporary database."""
    db_path = tmp_path / "penny.db"
    if create_db:
        _create_test_db(db_path)

    agent = QualityAgent(
        name="quality",
        db_path=str(db_path),
        ollama_url="http://localhost:11434",
        ollama_model="test-model",
        interval_seconds=3600,
        timeout_seconds=600,
        working_dir=tmp_path,
        trusted_users=TRUSTED_USERS,
        github_api=github_api,
    )

    return agent, db_path


# =============================================================================
# Helper: mock Ollama responses
# =============================================================================


class MockOllamaResponse:
    """Mock urllib response for Ollama API calls."""

    def __init__(self, content: dict):
        self._data = json.dumps(
            {"message": {"role": "assistant", "content": json.dumps(content)}}
        ).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def mock_ollama_factory(responses: list[dict]):
    """Create a mock urlopen that returns canned Ollama responses in order."""
    call_index = 0

    def mock_urlopen(req, timeout=None):
        nonlocal call_index
        if call_index < len(responses):
            resp = MockOllamaResponse(responses[call_index])
            call_index += 1
            return resp
        return MockOllamaResponse({"issues": []})

    return mock_urlopen


# =============================================================================
# validate_privacy — unit tests
# =============================================================================


class TestValidatePrivacy:
    def test_safe_body_passes(self):
        originals = ["What is the weather in Portland?", "The weather is sunny and warm today."]
        body = "Penny sometimes returns malformed responses when asked about general topics."
        assert validate_privacy(originals, body) is True

    def test_leaked_user_message_fails(self):
        originals = ["What is the weather in Portland?", "The weather is sunny."]
        body = "Example: user asked 'What is the weather in Portland?' and got a bad response."
        assert validate_privacy(originals, body) is False

    def test_leaked_response_fails(self):
        originals = ["Tell me a joke", "The weather is sunny and warm today my friend."]
        body = "Penny responded with 'the weather is sunny and warm today my friend.' which was off-topic."
        assert validate_privacy(originals, body) is False

    def test_short_messages_skipped(self):
        """Messages under 20 chars are skipped to avoid false positives."""
        originals = ["hi", "Hello!", "The weather is sunny and warm today."]
        body = "When a user says hi, Penny should respond with a greeting."
        assert validate_privacy(originals, body) is True

    def test_case_insensitive_check(self):
        originals = ["What Is The Weather In Portland Today?"]
        body = "User asked 'what is the weather in portland today?' — leaked in body."
        assert validate_privacy(originals, body) is False

    def test_empty_originals_passes(self):
        assert validate_privacy([], "Any body text") is True

    def test_empty_body_passes(self):
        originals = ["A sufficiently long original message here"]
        assert validate_privacy(originals, "") is True


# =============================================================================
# has_work() — integration tests
# =============================================================================


class TestQualityHasWork:
    def test_no_database_returns_false(self, tmp_path):
        agent, _ = make_quality_agent(tmp_path, create_db=False)
        assert agent.has_work() is False

    def test_empty_database_returns_false(self, tmp_path):
        agent, _ = make_quality_agent(tmp_path)
        assert agent.has_work() is False

    def test_new_messages_returns_true(self, tmp_path):
        agent, db_path = make_quality_agent(tmp_path)
        _insert_message_pair(db_path)
        assert agent.has_work() is True

    def test_no_new_messages_after_timestamp_returns_false(self, tmp_path):
        agent, db_path = make_quality_agent(tmp_path)
        ts = datetime.now()
        _insert_message_pair(db_path, timestamp=ts)
        # Save a timestamp after the message
        agent._save_last_timestamp((ts + timedelta(seconds=5)).isoformat())
        assert agent.has_work() is False

    def test_new_messages_after_timestamp_returns_true(self, tmp_path):
        agent, db_path = make_quality_agent(tmp_path)
        old_ts = datetime.now() - timedelta(hours=2)
        _insert_message_pair(db_path, timestamp=old_ts)
        agent._save_last_timestamp(old_ts.isoformat())

        # Insert a newer message
        _insert_message_pair(db_path, timestamp=datetime.now())
        assert agent.has_work() is True


# =============================================================================
# _read_message_pairs() — unit tests
# =============================================================================


class TestReadMessagePairs:
    def test_reads_message_pairs(self, tmp_path):
        agent, db_path = make_quality_agent(tmp_path)
        _insert_message_pair(db_path, user_message="Hello", response="Hi there!")

        pairs = agent._read_message_pairs()
        assert len(pairs) == 1
        assert pairs[0].user_message == "Hello"
        assert pairs[0].response == "Hi there!"

    def test_respects_timestamp_filter(self, tmp_path):
        agent, db_path = make_quality_agent(tmp_path)
        old_ts = datetime.now() - timedelta(hours=2)
        _insert_message_pair(db_path, user_message="Old", response="Old reply", timestamp=old_ts)
        # Save timestamp after the outgoing message (which is old_ts + 1s)
        agent._save_last_timestamp((old_ts + timedelta(seconds=2)).isoformat())

        new_ts = datetime.now()
        _insert_message_pair(db_path, user_message="New", response="New reply", timestamp=new_ts)

        pairs = agent._read_message_pairs()
        assert len(pairs) == 1
        assert pairs[0].user_message == "New"

    def test_includes_prompt_log_data(self, tmp_path):
        agent, db_path = make_quality_agent(tmp_path)
        ts = datetime.now()
        _insert_message_pair(db_path, timestamp=ts)
        _insert_prompt_log(db_path, timestamp=ts, thinking="Let me think about this")

        pairs = agent._read_message_pairs()
        assert len(pairs) == 1
        assert pairs[0].thinking == "Let me think about this"

    def test_empty_db_returns_empty_list(self, tmp_path):
        agent, _ = make_quality_agent(tmp_path)
        pairs = agent._read_message_pairs()
        assert pairs == []


# =============================================================================
# run() — integration tests
# =============================================================================


class TestQualityRun:
    def test_no_messages_skips_evaluation(self, tmp_path):
        """Empty database -> no pairs -> no Ollama call."""
        agent, _ = make_quality_agent(tmp_path)

        result = agent.run()

        assert result.success is True
        assert result.output == "No message pairs to evaluate"

    def test_all_good_messages_no_issues_filed(self, tmp_path, monkeypatch):
        """All messages pass quality check -> no issues filed."""
        api = MockGitHubAPI()
        agent, db_path = make_quality_agent(tmp_path, github_api=api)
        _insert_message_pair(db_path)

        mock_urlopen = mock_ollama_factory([{"is_bad": False}])
        monkeypatch.setattr("penny_team.quality.urllib.request.urlopen", mock_urlopen)

        result = agent.run()

        assert result.success is True
        assert result.output == "All responses passed quality check"
        # No create_issue calls
        assert not any(c[0] == "create_issue" for c in api.calls)

    def test_bad_message_files_issue(self, tmp_path, monkeypatch):
        """Bad response detected -> issue filed with correct labels."""
        api = MockGitHubAPI()
        api.create_issue = MagicMock(return_value="https://github.com/test/issues/1")
        agent, db_path = make_quality_agent(tmp_path, github_api=api)
        _insert_message_pair(
            db_path,
            user_message="What time is it?",
            response="<function=get_time>{}</function> The time is 3pm",
        )

        evaluation_response = {
            "is_bad": True,
            "category": "exposed_function_call",
            "reason": "Function call syntax leaked to user",
        }
        bug_description = {
            "title": "bug: Function call syntax leaked in response",
            "body": (
                "*[Quality Agent]*\n\n"
                "## Quality Issue\n\n"
                "Penny exposed internal function call syntax to the user.\n\n"
                "## Example (Contrived)\n\n"
                "A user asked about the date and received a response containing "
                "raw function markup.\n\n"
                "## Suggested Fix\n\n"
                "Check response sanitization in the agent pipeline."
            ),
        }
        mock_urlopen = mock_ollama_factory([evaluation_response, bug_description])
        monkeypatch.setattr("penny_team.quality.urllib.request.urlopen", mock_urlopen)

        result = agent.run()

        assert result.success is True
        assert "Filed 1 issue(s)" in result.output
        api.create_issue.assert_called_once()
        call_args = api.create_issue.call_args
        assert call_args[0][2] == ["bug", "quality"]  # labels

    def test_privacy_validation_blocks_leaky_description(self, tmp_path, monkeypatch):
        """If bug description contains original content, issue is NOT filed."""
        api = MockGitHubAPI()
        api.create_issue = MagicMock(return_value="https://github.com/test/issues/1")
        agent, db_path = make_quality_agent(tmp_path, github_api=api)
        _insert_message_pair(
            db_path,
            user_message="What is the weather in Portland?",
            response="Error: connection timeout to weather service",
        )

        evaluation_response = {
            "is_bad": True,
            "category": "leaked_error",
            "reason": "Error message exposed to user",
        }
        # Bug description leaks the original user message
        leaky_description = {
            "title": "bug: Error message leaked",
            "body": (
                "*[Quality Agent]*\n\n"
                "## Quality Issue\n\n"
                "User asked 'What is the weather in Portland?' and got an error.\n\n"
                "## Example\n\nSame as above.\n\n"
                "## Suggested Fix\n\nHandle timeouts gracefully."
            ),
        }
        mock_urlopen = mock_ollama_factory([evaluation_response, leaky_description])
        monkeypatch.setattr("penny_team.quality.urllib.request.urlopen", mock_urlopen)

        result = agent.run()

        assert result.success is True
        assert "Filed 0 issue(s)" in result.output
        api.create_issue.assert_not_called()

    def test_ollama_failure_returns_no_issues(self, tmp_path, monkeypatch):
        """If Ollama fails, no issues are filed (graceful degradation)."""
        api = MockGitHubAPI()
        agent, db_path = make_quality_agent(tmp_path, github_api=api)
        _insert_message_pair(db_path)

        def fail_urlopen(req, timeout=None):
            raise OSError("Connection refused")

        monkeypatch.setattr("penny_team.quality.urllib.request.urlopen", fail_urlopen)

        result = agent.run()

        assert result.success is True
        assert result.output == "All responses passed quality check"

    def test_timestamp_advances_after_run(self, tmp_path, monkeypatch):
        """After run(), the timestamp advances so same messages aren't re-evaluated."""
        agent, db_path = make_quality_agent(tmp_path)
        _insert_message_pair(db_path)

        mock_urlopen = mock_ollama_factory([{"is_bad": False}])
        monkeypatch.setattr("penny_team.quality.urllib.request.urlopen", mock_urlopen)

        agent.run()

        # Timestamp should be saved
        saved_ts = agent._load_last_timestamp()
        assert saved_ts != ""

        # No new work
        assert agent.has_work() is False

    def test_max_issues_per_cycle_cap(self, tmp_path, monkeypatch):
        """At most QUALITY_MAX_ISSUES_PER_CYCLE issues are filed per run."""
        api = MockGitHubAPI()
        api.create_issue = MagicMock(return_value="https://github.com/test/issues/1")
        agent, db_path = make_quality_agent(tmp_path, github_api=api)

        # Insert 5 message pairs
        for i in range(5):
            _insert_message_pair(
                db_path,
                user_message=f"Question {i}",
                response=f"<function=tool{i}> Bad response {i}",
                timestamp=datetime.now() + timedelta(seconds=i * 2),
            )

        # Each pair gets its own evaluation call (all bad) + bug description call.
        # After 3 issues are filed, the loop stops (cap = 3), so only 3 eval + 3 desc = 6 calls.
        responses: list[dict] = []
        for i in range(3):
            responses.append(
                {"is_bad": True, "category": "exposed_function_call", "reason": f"Issue {i}"}
            )
            responses.append(
                {
                    "title": f"bug: Quality issue {i}",
                    "body": f"*[Quality Agent]*\n\n## Quality Issue\n\nProblem {i}\n\n"
                    f"## Example (Contrived)\n\nExample {i}\n\n## Suggested Fix\n\nFix {i}",
                }
            )
        mock_urlopen = mock_ollama_factory(responses)
        monkeypatch.setattr("penny_team.quality.urllib.request.urlopen", mock_urlopen)

        result = agent.run()

        assert result.success is True
        assert api.create_issue.call_count == 3

    def test_no_github_api_skips_filing(self, tmp_path, monkeypatch):
        """Without GitHub API configured, issues are detected but not filed."""
        agent, db_path = make_quality_agent(tmp_path, github_api=None)
        _insert_message_pair(
            db_path,
            user_message="Tell me something",
            response="<function=broken> oops",
        )

        evaluation_response = {
            "is_bad": True,
            "category": "exposed_function_call",
            "reason": "Leaked",
        }
        bug_description = {
            "title": "bug: Leaked function call",
            "body": "*[Quality Agent]*\n\n## Quality Issue\n\nLeaked.\n\n"
            "## Example (Contrived)\n\nExample.\n\n## Suggested Fix\n\nFix.",
        }
        mock_urlopen = mock_ollama_factory([evaluation_response, bug_description])
        monkeypatch.setattr("penny_team.quality.urllib.request.urlopen", mock_urlopen)

        result = agent.run()

        assert result.success is True
        assert "Filed 0 issue(s)" in result.output

    def test_database_error_returns_failure(self, tmp_path):
        """If database read fails, run returns failure."""
        agent, db_path = make_quality_agent(tmp_path, create_db=False)
        # Create a non-database file so the path exists but can't be read as SQLite
        db_path.write_text("not a database")

        result = agent.run()

        assert result.success is False
        assert "Failed to read database" in result.output

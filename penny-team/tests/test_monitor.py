"""Tests for the Monitor agent.

The Monitor agent reads penny's production logs, extracts errors,
and uses Claude CLI to analyze and file bug issues. Unlike other
agents, it reads files instead of GitHub issues.

Unit tests cover the pure extraction functions. Integration tests
verify the full flow through has_work() and run().
"""

from __future__ import annotations

from pathlib import Path

from penny_team.monitor import ErrorBlock, MonitorAgent, extract_errors, format_errors_for_prompt

from tests.conftest import (
    TRUSTED_USERS,
    extract_prompt,
    result_event,
)


# =============================================================================
# Helper: create a MonitorAgent with a temp log file
# =============================================================================


def make_monitor_agent(
    tmp_path: Path,
    log_content: str = "",
) -> tuple[MonitorAgent, Path]:
    """Create a MonitorAgent with a temporary log file and prompt."""
    log_file = tmp_path / "penny.log"
    if log_content:
        log_file.write_text(log_content)

    agent_dir = tmp_path / "penny_team" / "monitor"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "CLAUDE.md").write_text("# Monitor Agent Prompt\n\nYou are the monitor agent.\n")

    agent = MonitorAgent(
        name="monitor",
        log_path=str(log_file),
        interval_seconds=300,
        timeout_seconds=600,
        working_dir=tmp_path,
        trusted_users=TRUSTED_USERS,
    )
    agent.prompt_path = agent_dir / "CLAUDE.md"

    # Override state path to use tmp_path
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "monitor.state.json"
    type(agent)._state_path = property(lambda self, p=state_path: p)

    return agent, log_file


# =============================================================================
# extract_errors — unit tests for pure function
# =============================================================================


class TestExtractErrors:
    def test_empty_log_returns_empty(self):
        assert extract_errors("") == []

    def test_info_lines_ignored(self):
        log = "2024-01-15 14:23:45 - penny.agent - INFO - All good\n"
        assert extract_errors(log) == []

    def test_single_error_no_traceback(self):
        log = "2024-01-15 14:23:45 - penny.agent - ERROR - Something broke\n"
        errors = extract_errors(log)
        assert len(errors) == 1
        assert errors[0].timestamp == "2024-01-15 14:23:45"
        assert errors[0].module == "penny.agent"
        assert errors[0].level == "ERROR"
        assert errors[0].message == "Something broke"
        assert errors[0].traceback == ""

    def test_error_with_traceback(self):
        log = (
            "2024-01-15 14:23:45 - penny.agent - ERROR - Unhandled exception\n"
            "Traceback (most recent call last):\n"
            '  File "penny/agent/base.py", line 42, in run\n'
            "    result = await self.execute()\n"
            "ValueError: invalid input\n"
            "2024-01-15 14:23:46 - penny.agent - INFO - Recovered\n"
        )
        errors = extract_errors(log)
        assert len(errors) == 1
        assert "Traceback" in errors[0].traceback
        assert "ValueError: invalid input" in errors[0].traceback

    def test_critical_level_extracted(self):
        log = "2024-01-15 14:23:45 - penny.database - CRITICAL - DB corrupted\n"
        errors = extract_errors(log)
        assert len(errors) == 1
        assert errors[0].level == "CRITICAL"

    def test_multiple_errors(self):
        log = (
            "2024-01-15 14:23:45 - penny.agent - ERROR - First error\n"
            "2024-01-15 14:23:46 - penny.agent - INFO - Some info\n"
            "2024-01-15 14:23:47 - penny.tools - ERROR - Second error\n"
        )
        errors = extract_errors(log)
        assert len(errors) == 2
        assert errors[0].message == "First error"
        assert errors[1].message == "Second error"

    def test_warning_level_ignored(self):
        log = "2024-01-15 14:23:45 - penny.agent - WARNING - Just a warning\n"
        assert extract_errors(log) == []


# =============================================================================
# format_errors_for_prompt — unit test
# =============================================================================


class TestFormatErrors:
    def test_no_errors_returns_no_errors_message(self):
        result = format_errors_for_prompt([])
        assert "No errors found" in result

    def test_formats_error_with_traceback(self):
        error = ErrorBlock(
            timestamp="2024-01-15 14:23:45",
            module="penny.agent",
            level="ERROR",
            message="Something broke",
            traceback="Traceback (most recent call last):\n  ValueError: bad",
        )
        result = format_errors_for_prompt([error])
        assert "penny.agent" in result
        assert "Something broke" in result
        assert "ValueError: bad" in result
        assert "Error 1" in result


# =============================================================================
# has_work() — integration tests
# =============================================================================


class TestMonitorHasWork:
    def test_no_log_file_returns_false(self, tmp_path):
        agent, log_file = make_monitor_agent(tmp_path)
        # No content was written, so file wasn't created
        assert agent.has_work() is False

    def test_empty_log_file_returns_false(self, tmp_path):
        agent, log_file = make_monitor_agent(tmp_path)
        log_file.write_text("")
        assert agent.has_work() is False

    def test_first_run_with_content_returns_true(self, tmp_path):
        agent, _ = make_monitor_agent(
            tmp_path,
            log_content="2024-01-15 14:23:45 - penny.agent - ERROR - test\n",
        )
        assert agent.has_work() is True

    def test_no_new_content_returns_false(self, tmp_path):
        agent, log_file = make_monitor_agent(
            tmp_path,
            log_content="2024-01-15 14:23:45 - penny.agent - ERROR - test\n",
        )
        file_size = log_file.stat().st_size
        agent._save_offset(file_size)
        assert agent.has_work() is False

    def test_new_content_appended_returns_true(self, tmp_path):
        agent, log_file = make_monitor_agent(
            tmp_path,
            log_content="2024-01-15 14:23:45 - penny.agent - INFO - old\n",
        )
        old_size = log_file.stat().st_size
        agent._save_offset(old_size)

        with open(log_file, "a") as f:
            f.write("2024-01-15 14:24:00 - penny.agent - ERROR - new error\n")

        assert agent.has_work() is True

    def test_log_rotation_detected_returns_true(self, tmp_path):
        agent, _ = make_monitor_agent(
            tmp_path,
            log_content="short",
        )
        agent._save_offset(999999)
        assert agent.has_work() is True


# =============================================================================
# run() — integration tests through agent.run()
# =============================================================================


class TestMonitorRun:
    def test_no_errors_in_log_skips_claude(self, tmp_path, capture_popen):
        """Log with only INFO lines -> no errors -> Claude CLI not called."""
        agent, _ = make_monitor_agent(
            tmp_path,
            log_content="2024-01-15 14:23:45 - penny.agent - INFO - All good\n",
        )

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        assert result.output == "No errors in logs"
        assert len(calls) == 0

    def test_errors_in_log_triggers_claude(self, tmp_path, capture_popen):
        """Log with ERROR lines -> errors extracted -> Claude CLI called."""
        agent, _ = make_monitor_agent(
            tmp_path,
            log_content=(
                "2024-01-15 14:23:45 - penny.agent - ERROR - Unhandled exception\n"
                "Traceback (most recent call last):\n"
                '  File "penny/agent/base.py", line 42, in run\n'
                "    result = await self.execute()\n"
                "ValueError: invalid input\n"
            ),
        )

        calls = capture_popen(stdout_lines=[result_event("Filed issue #100")], returncode=0)

        result = agent.run()

        assert result.success is True
        prompt = extract_prompt(calls)
        assert "Monitor Agent Prompt" in prompt
        assert "Unhandled exception" in prompt
        assert "ValueError: invalid input" in prompt

    def test_offset_advances_after_run(self, tmp_path, capture_popen):
        """After run(), the byte offset advances so same content isn't re-read."""
        agent, log_file = make_monitor_agent(
            tmp_path,
            log_content="2024-01-15 14:23:45 - penny.agent - ERROR - test error\n",
        )

        capture_popen(stdout_lines=[result_event()], returncode=0)
        agent.run()

        assert agent._load_offset() == log_file.stat().st_size
        assert agent.has_work() is False

    def test_offset_advances_even_without_errors(self, tmp_path, capture_popen):
        """After run() with no errors, offset still advances past INFO lines."""
        agent, log_file = make_monitor_agent(
            tmp_path,
            log_content="2024-01-15 14:23:45 - penny.agent - INFO - All good\n",
        )

        capture_popen(stdout_lines=[result_event()], returncode=0)
        agent.run()

        assert agent._load_offset() == log_file.stat().st_size

    def test_log_rotation_resets_offset(self, tmp_path, capture_popen):
        """If log file is smaller than saved offset, reads from start."""
        agent, _ = make_monitor_agent(
            tmp_path,
            log_content="2024-01-15 14:23:45 - penny.agent - ERROR - after rotation\n",
        )
        agent._save_offset(999999)

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)
        result = agent.run()

        assert result.success is True
        prompt = extract_prompt(calls)
        assert "after rotation" in prompt

    def test_first_run_reads_tail_of_large_log(self, tmp_path, capture_popen):
        """On first run with a large log, only reads last 100KB."""
        filler = "2024-01-15 14:23:45 - penny.agent - INFO - filler line here\n"
        # Each filler line is ~56 bytes, need ~2000 lines to exceed 100KB
        large_content = filler * 2000
        error_line = "2024-01-15 14:23:45 - penny.agent - ERROR - recent error\n"
        agent, _ = make_monitor_agent(
            tmp_path,
            log_content=large_content + error_line,
        )

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)
        result = agent.run()

        assert result.success is True
        assert len(calls) == 1
        prompt = extract_prompt(calls)
        assert "recent error" in prompt

    def test_missing_log_file_returns_failure(self, tmp_path, capture_popen):
        """If log file doesn't exist at run time, returns failure."""
        agent, log_file = make_monitor_agent(
            tmp_path,
            log_content="some content",
        )
        log_file.unlink()

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)
        result = agent.run()

        assert result.success is False
        assert "Failed to read log" in result.output
        assert len(calls) == 0

    def test_claude_cli_failure_returns_failure(self, tmp_path, capture_popen):
        """If Claude CLI returns non-zero, result.success is False."""
        agent, _ = make_monitor_agent(
            tmp_path,
            log_content="2024-01-15 14:23:45 - penny.agent - ERROR - test\n",
        )

        capture_popen(stdout_lines=[result_event("error")], returncode=1)
        result = agent.run()

        assert result.success is False

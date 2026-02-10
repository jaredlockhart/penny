"""Monitor agent: reads penny production logs, detects errors, files bug issues.

Reads new content from penny's log file since the last run, extracts
ERROR/CRITICAL lines with their tracebacks, and uses Claude CLI to
analyze errors, deduplicate against existing bug issues, and create
new bug issues for the Worker agent to fix.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from penny_team.base import Agent, AgentRun
from penny_team.constants import (
    LOG_LEVELS_ERROR,
    MONITOR_FIRST_RUN_MAX_BYTES,
    MONITOR_MAX_ERROR_CONTEXT,
    MONITOR_STATE_OFFSET,
)
from penny_team.utils.github_api import GitHubAPI
from penny_team.utils.github_app import GitHubApp

logger = logging.getLogger(__name__)


@dataclass
class ErrorBlock:
    """A single error extracted from logs: the ERROR line plus any traceback."""

    timestamp: str
    module: str
    level: str
    message: str
    traceback: str


# Pattern to match standard Python log lines:
# "2024-01-15 14:23:45 - penny.module - ERROR - Error message here"
_LOG_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"  # timestamp
    r" - "
    r"([\w.]+)"  # module
    r" - "
    r"(\w+)"  # level
    r" - "
    r"(.*)$",  # message
)


def extract_errors(log_text: str) -> list[ErrorBlock]:
    """Extract ERROR/CRITICAL log entries with their tracebacks.

    Scans log lines for ERROR or CRITICAL level entries. When found,
    captures the line and any subsequent lines that are part of a
    traceback (indented lines, "Traceback" header, exception lines)
    until the next timestamped log line.
    """
    errors: list[ErrorBlock] = []
    lines = log_text.split("\n")
    i = 0

    while i < len(lines):
        match = _LOG_LINE_RE.match(lines[i])
        if match:
            timestamp, module, level, message = match.groups()
            if level in LOG_LEVELS_ERROR:
                # Collect traceback lines that follow
                traceback_lines: list[str] = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j]
                    # Stop at the next timestamped log line
                    if _LOG_LINE_RE.match(next_line):
                        break
                    traceback_lines.append(next_line)
                    j += 1

                errors.append(
                    ErrorBlock(
                        timestamp=timestamp,
                        module=module,
                        level=level,
                        message=message,
                        traceback="\n".join(traceback_lines).strip(),
                    )
                )
                i = j
                continue
        i += 1

    return errors


def format_errors_for_prompt(errors: list[ErrorBlock]) -> str:
    """Format extracted errors into a section for the Claude prompt."""
    if not errors:
        return "\n\n# Log Errors\n\nNo errors found in recent logs.\n"

    parts = [
        "\n\n# Log Errors\n\n"
        "The following errors were extracted from penny's production logs. "
        "Analyze these errors, deduplicate against existing bug issues, and "
        "create new bug issues for genuinely new problems.\n\n---\n"
    ]

    for idx, error in enumerate(errors, 1):
        section = f"\n## Error {idx}\n"
        section += f"**Timestamp**: {error.timestamp}\n"
        section += f"**Module**: {error.module}\n"
        section += f"**Level**: {error.level}\n"
        section += f"**Message**: {error.message}\n"
        if error.traceback:
            section += f"\n**Traceback**:\n```\n{error.traceback}\n```\n"
        section += "\n---\n"
        parts.append(section)

    return "".join(parts)


class MonitorAgent(Agent):
    """Agent that monitors penny's production logs and files bug issues.

    Overrides has_work() and run() to read log files instead of GitHub
    issues. Uses byte offset tracking to only process new log content.
    """

    def __init__(
        self,
        log_path: str | Path | None = None,
        name: str = "monitor",
        interval_seconds: int = 300,
        working_dir: Path | None = None,
        timeout_seconds: int = 600,
        model: str | None = None,
        allowed_tools: list[str] | None = None,
        github_app: GitHubApp | None = None,
        github_api: GitHubAPI | None = None,
        trusted_users: set[str] | None = None,
    ) -> None:
        from penny_team.base import PROJECT_ROOT

        super().__init__(
            name=name,
            interval_seconds=interval_seconds,
            working_dir=working_dir or PROJECT_ROOT,
            timeout_seconds=timeout_seconds,
            model=model,
            allowed_tools=allowed_tools,
            github_app=github_app,
            github_api=github_api,
            trusted_users=trusted_users,
        )
        if log_path is not None:
            self.log_path = Path(log_path)
        else:
            self.log_path = PROJECT_ROOT / "data" / "penny.log"

    def _load_offset(self) -> int:
        """Load saved byte offset from state file."""
        state = self._load_state()
        return int(state.get(MONITOR_STATE_OFFSET, "0"))

    def _save_offset(self, offset: int) -> None:
        """Persist byte offset to state file."""
        self._save_state({MONITOR_STATE_OFFSET: str(offset)})

    def has_work(self) -> bool:
        """Check if the log file has new content since the last read.

        Returns True if the log file has grown beyond the saved offset,
        if log rotation is detected, or on first run. Returns False if
        the file doesn't exist, is empty, or hasn't changed.
        """
        try:
            if not self.log_path.exists():
                logger.info(f"[{self.name}] Log file not found: {self.log_path}")
                return False

            file_size = self.log_path.stat().st_size
            if file_size == 0:
                return False

            saved_offset = self._load_offset()

            if file_size < saved_offset:
                logger.info(f"[{self.name}] Log rotation detected, will read from start")
                return True

            if file_size > saved_offset:
                return True

            logger.info(f"[{self.name}] No new log content since last run")
            return False

        except OSError as e:
            logger.warning(f"[{self.name}] Error checking log file: {e}")
            return True  # Fail-open

    def _read_new_log_content(self) -> tuple[str, int]:
        """Read new log content since the last saved offset.

        On first run, reads the last MONITOR_FIRST_RUN_MAX_BYTES.
        On subsequent runs, reads from the saved offset to EOF.
        If file is smaller than saved offset (rotation), resets to 0.

        Returns (content, new_offset) tuple.
        """
        saved_offset = self._load_offset()
        file_size = self.log_path.stat().st_size

        if file_size < saved_offset:
            saved_offset = 0

        if saved_offset == 0 and file_size > MONITOR_FIRST_RUN_MAX_BYTES:
            saved_offset = file_size - MONITOR_FIRST_RUN_MAX_BYTES

        with open(self.log_path) as f:
            f.seek(saved_offset)
            content = f.read()

        new_offset = self.log_path.stat().st_size
        return content, new_offset

    def run(self) -> AgentRun:
        """Read new log content, extract errors, and run Claude to file bug issues."""
        logger.info(f"[{self.name}] Starting cycle #{self.run_count + 1}")
        start = datetime.now()

        try:
            log_content, new_offset = self._read_new_log_content()
        except OSError as e:
            duration = (datetime.now() - start).total_seconds()
            self.last_run = datetime.now()
            self.run_count += 1
            logger.error(f"[{self.name}] Failed to read log file: {e}")
            return AgentRun(
                agent_name=self.name,
                success=False,
                output=f"Failed to read log: {e}",
                duration=duration,
                timestamp=start,
            )

        errors = extract_errors(log_content)

        if not errors:
            self._save_offset(new_offset)
            duration = (datetime.now() - start).total_seconds()
            self.last_run = datetime.now()
            self.run_count += 1
            logger.info(f"[{self.name}] No errors found in new log content")
            return AgentRun(
                agent_name=self.name,
                success=True,
                output="No errors in logs",
                duration=duration,
                timestamp=start,
            )

        logger.info(f"[{self.name}] Found {len(errors)} error(s) in logs")

        prompt = self.prompt_path.read_text()
        error_section = format_errors_for_prompt(errors)

        if len(error_section) > MONITOR_MAX_ERROR_CONTEXT:
            error_section = error_section[:MONITOR_MAX_ERROR_CONTEXT] + "\n\n... (truncated)\n"

        prompt += error_section

        success, result_text = self._execute_claude(prompt)

        # Save offset after execution so errors aren't re-processed,
        # regardless of Claude success (avoid infinite retry loops)
        self._save_offset(new_offset)

        duration = (datetime.now() - start).total_seconds()
        self.last_run = datetime.now()
        self.run_count += 1

        level = logging.INFO if success else logging.ERROR
        status = "OK" if success else "FAILED"
        logger.log(level, f"[{self.name}] Cycle #{self.run_count} {status} in {duration:.1f}s")

        return AgentRun(
            agent_name=self.name,
            success=success,
            output=result_text,
            duration=duration,
            timestamp=start,
        )

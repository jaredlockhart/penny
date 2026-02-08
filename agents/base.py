"""Base agent class for Penny's autonomous agent system.

Each agent wraps Claude CLI, running with a specific prompt on a schedule.
The orchestrator manages agent lifecycles and scheduling.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from github_app import GitHubApp

CLAUDE_CLI = os.getenv("CLAUDE_CLI", "claude")
GH_CLI = os.getenv("GH_CLI", "gh")
# Labels where external state (CI checks) can change without updating issue timestamps
LABELS_WITH_EXTERNAL_STATE = {"in-review"}
AGENTS_DIR = Path(__file__).parent
PROJECT_ROOT = AGENTS_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "agents"
PROMPT_FILENAME = "CLAUDE.md"

# Stream-json event types from Claude CLI
EVENT_ASSISTANT = "assistant"
EVENT_RESULT = "result"

# Stream-json content block types
BLOCK_TEXT = "text"
BLOCK_TOOL_USE = "tool_use"

# GitHub API JSON field names
GH_FIELD_NUMBER = "number"
GH_FIELD_UPDATED_AT = "updatedAt"

logger = logging.getLogger(__name__)


@dataclass
class AgentRun:
    agent_name: str
    success: bool
    output: str
    duration: float
    timestamp: datetime


class Agent:
    def __init__(
        self,
        name: str,
        interval_seconds: int = 3600,
        working_dir: Path = PROJECT_ROOT,
        timeout_seconds: int = 600,
        model: str | None = None,
        allowed_tools: list[str] | None = None,
        required_labels: list[str] | None = None,
        github_app: GitHubApp | None = None,
        trusted_users: set[str] | None = None,
    ):
        self.name = name
        self.prompt_path = AGENTS_DIR / name / PROMPT_FILENAME
        self.interval_seconds = interval_seconds
        self.working_dir = working_dir
        self.timeout_seconds = timeout_seconds
        self.model = model
        self.allowed_tools = allowed_tools
        self.required_labels = required_labels
        self.github_app = github_app
        self.trusted_users = trusted_users
        self.last_run: datetime | None = None
        self.run_count = 0
        self._process: subprocess.Popen | None = None

    def is_due(self) -> bool:
        if self.last_run is None:
            return True
        elapsed = (datetime.now() - self.last_run).total_seconds()
        return elapsed >= self.interval_seconds

    def _get_env(self) -> dict[str, str] | None:
        """Build subprocess env with bot identity if GitHub App is configured."""
        if self.github_app is None:
            return None
        env = os.environ.copy()
        env.update(self.github_app.get_env())
        return env

    @property
    def _state_path(self) -> Path:
        return DATA_DIR / f"{self.name}.state.json"

    def _load_state(self) -> dict[str, str]:
        """Load saved issue timestamps from disk."""
        try:
            return json.loads(self._state_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_state(self, timestamps: dict[str, str]) -> None:
        """Persist issue timestamps to disk."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps(timestamps))

    def _fetch_issue_timestamps(self) -> dict[str, str]:
        """Fetch updatedAt timestamps for all issues matching required labels.

        Returns a dict mapping issue number (str) to updatedAt timestamp.
        Raises RuntimeError if any gh call fails (caller should fail open).
        """
        timestamps: dict[str, str] = {}
        for label in self.required_labels or []:
            result = subprocess.run(
                [GH_CLI, "issue", "list", "--label", label, "--json", f"{GH_FIELD_NUMBER},{GH_FIELD_UPDATED_AT}", "--limit", "20"],
                capture_output=True, text=True, timeout=15, env=self._get_env(),
            )
            if result.returncode != 0:
                raise RuntimeError(f"gh issue list failed for label '{label}'")
            for issue in json.loads(result.stdout):
                timestamps[str(issue[GH_FIELD_NUMBER])] = issue[GH_FIELD_UPDATED_AT]
        return timestamps

    def has_work(self) -> bool:
        """Check if there are new or updated GitHub issues since the last run.

        Returns True if no labels are configured (always run), if any issue
        has been created/updated/removed since the last saved state, or if
        gh fails (fail-open).

        For labels with external state (e.g. in-review where CI checks can
        change without updating issue timestamps), always returns True when
        matching issues exist.
        """
        if not self.required_labels:
            return True

        try:
            current = self._fetch_issue_timestamps()
        except (subprocess.TimeoutExpired, OSError, RuntimeError):
            return True

        if not current:
            return False

        # CI check status changes don't update issue timestamps, so skip
        # the timestamp optimization for labels with external state
        has_external_state = any(
            label in LABELS_WITH_EXTERNAL_STATE
            for label in self.required_labels
        )
        if has_external_state:
            return True

        saved = self._load_state()
        if current == saved:
            logger.info(f"[{self.name}] No issue changes since last run, skipping")
            return False

        return True

    def _build_command(self, prompt: str) -> list[str]:
        cmd = [
            CLAUDE_CLI,
            "-p",
            prompt,
            "--dangerously-skip-permissions",
            "--verbose",
            "--output-format", "stream-json",
        ]
        if self.model:
            cmd.extend(["--model", self.model])
        if self.allowed_tools:
            cmd.extend(["--allowedTools", *self.allowed_tools])
        return cmd

    def run(self) -> AgentRun:
        logger.info(f"[{self.name}] Starting cycle #{self.run_count + 1}")
        start = datetime.now()

        prompt = self.prompt_path.read_text()

        # Pre-fetch, filter, and pick one actionable issue
        if self.required_labels:
            from issue_filter import fetch_issues_for_labels, format_issues_for_prompt, pick_actionable_issue

            bot_login = self.github_app.bot_name if self.github_app else None
            all_issues = fetch_issues_for_labels(self.required_labels, trusted_users=self.trusted_users, env=self._get_env())

            # Enrich in-review issues with CI and merge conflict status (no-op if none match)
            from pr_checks import enrich_issues_with_pr_status
            enrich_issues_with_pr_status(all_issues, env=self._get_env())

            issue = pick_actionable_issue(all_issues, bot_login)

            if issue is None:
                duration = (datetime.now() - start).total_seconds()
                self.last_run = datetime.now()
                self.run_count += 1
                logger.info(f"[{self.name}] No actionable issues (bot has last comment on all), skipping")
                try:
                    self._save_state(self._fetch_issue_timestamps())
                except (subprocess.TimeoutExpired, OSError, RuntimeError) as e:
                    logger.warning(f"[{self.name}] Failed to save issue state: {e}")
                return AgentRun(
                    agent_name=self.name,
                    success=True,
                    output="No actionable issues",
                    duration=duration,
                    timestamp=start,
                )

            prompt += format_issues_for_prompt([issue])

        cmd = self._build_command(prompt)

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(self.working_dir),
                env=self._get_env(),
            )
            self._process = process

            result_text = ""
            assert process.stdout is not None
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    self._log_event(event)
                    if event.get("type") == EVENT_RESULT:
                        result_text = event.get("result", "")
                except json.JSONDecodeError:
                    logger.info(f"[{self.name}] {line}")

            process.wait(timeout=self.timeout_seconds)
            self._process = None

            duration = (datetime.now() - start).total_seconds()
            self.last_run = datetime.now()
            self.run_count += 1

            success = process.returncode == 0
            level = logging.INFO if success else logging.ERROR
            logger.log(level, f"[{self.name}] Cycle #{self.run_count} {'OK' if success else 'FAILED'} in {duration:.1f}s")

            # Don't save state here — there may be more actionable issues.
            # State is only saved when pick_actionable_issue() returns None
            # (all issues handled), so has_work() keeps returning True until
            # the queue is fully burned down.

            return AgentRun(
                agent_name=self.name,
                success=success,
                output=result_text,
                duration=duration,
                timestamp=start,
            )

        except subprocess.TimeoutExpired:
            process.kill()
            self._process = None
            duration = (datetime.now() - start).total_seconds()
            self.last_run = datetime.now()
            self.run_count += 1
            logger.error(f"[{self.name}] Timed out after {duration:.1f}s")
            return AgentRun(
                agent_name=self.name,
                success=False,
                output="Process timed out",
                duration=duration,
                timestamp=start,
            )

    def _log_event(self, event: dict) -> None:
        """Log a stream-json event in a human-readable way."""
        event_type = event.get("type", "")

        if event_type == EVENT_ASSISTANT:
            # Assistant text output
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == BLOCK_TEXT:
                    for text_line in block[BLOCK_TEXT].split("\n"):
                        logger.info(f"[{self.name}] {text_line}")
                elif block.get("type") == BLOCK_TOOL_USE:
                    tool_name = block.get("name", "?")
                    logger.info(f"[{self.name}] [tool] {tool_name}")

        elif event_type == EVENT_RESULT:
            for text_line in event.get("result", "").split("\n"):
                logger.info(f"[{self.name}] {text_line}")

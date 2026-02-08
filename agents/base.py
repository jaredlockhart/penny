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
PROJECT_ROOT = Path(__file__).parent.parent

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
        prompt_path: Path,
        interval_seconds: int = 3600,
        working_dir: Path = PROJECT_ROOT,
        timeout_seconds: int = 600,
        model: str | None = None,
        allowed_tools: list[str] | None = None,
        required_labels: list[str] | None = None,
        max_issues: int | None = None,
        github_app: GitHubApp | None = None,
        trusted_users: set[str] | None = None,
    ):
        self.name = name
        self.prompt_path = prompt_path
        self.interval_seconds = interval_seconds
        self.working_dir = working_dir
        self.timeout_seconds = timeout_seconds
        self.model = model
        self.allowed_tools = allowed_tools
        self.required_labels = required_labels
        self.max_issues = max_issues
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

    def has_work(self) -> bool:
        """Check if there are GitHub issues matching any required label.

        Returns True if no labels are configured (always run) or if any
        label has at least one open issue.
        """
        if not self.required_labels:
            return True

        for label in self.required_labels:
            try:
                result = subprocess.run(
                    [GH_CLI, "issue", "list", "--label", label, "--json", "number", "--limit", "1"],
                    capture_output=True, text=True, timeout=15, env=self._get_env(),
                )
                if result.returncode == 0 and result.stdout.strip() not in ("", "[]"):
                    return True
            except (subprocess.TimeoutExpired, OSError):
                # If gh fails, run the agent anyway to be safe
                return True

        return False

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

        # Pre-fetch (and optionally filter) issue content to prevent prompt injection
        if self.required_labels:
            from issue_filter import fetch_issues_for_labels, format_issues_for_prompt

            issues = fetch_issues_for_labels(self.required_labels, trusted_users=self.trusted_users, env=self._get_env())
            if self.max_issues is not None:
                issues = issues[:self.max_issues]
            prompt += format_issues_for_prompt(issues)

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
                    if event.get("type") == "result":
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

        if event_type == "assistant":
            # Assistant text output
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "text":
                    for text_line in block["text"].split("\n"):
                        logger.info(f"[{self.name}] {text_line}")
                elif block.get("type") == "tool_use":
                    tool_name = block.get("name", "?")
                    logger.info(f"[{self.name}] [tool] {tool_name}")

        elif event_type == "result":
            for text_line in event.get("result", "").split("\n"):
                logger.info(f"[{self.name}] {text_line}")

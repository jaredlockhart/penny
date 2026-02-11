"""Base agent class for Penny's autonomous agent system.

Each agent wraps Claude CLI, running with a specific prompt on a schedule.
The orchestrator manages agent lifecycles and scheduling.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from github_api.api import GitHubAPI
from pydantic import BaseModel

from penny_team.constants import (
    APP_PREFIX,
    BLOCK_TEXT,
    BLOCK_TOOL_USE,
    CI_STATUS_FAILING,
    CLAUDE_CLI,
    EVENT_ASSISTANT,
    EVENT_RESULT,
    LABELS_WITH_EXTERNAL_STATE,
    MAX_CI_FIX_ATTEMPTS,
    PROMPT_FILENAME,
)

AGENTS_DIR = Path(__file__).parent
PROJECT_ROOT = AGENTS_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "penny-team"
LOG_DIR = PROJECT_ROOT / "data" / "logs"

logger = logging.getLogger(__name__)


class AgentState(BaseModel):
    """Persisted state for an agent, stored as JSON on disk.

    timestamps: issue updatedAt values for fast has_work() change detection.
    processed: per-agent record of when each issue was last processed,
               keyed by issue number (str). Used to distinguish "this agent
               handled it" from "another agent using the same bot identity
               handled it".
    ci_fix_attempts: number of times the agent has tried to fix CI for each
                     issue (keyed by issue number str). Reset when CI passes
                     or a human leaves review feedback.
    """

    timestamps: dict[str, str] = {}
    processed: dict[str, str] = {}
    ci_fix_attempts: dict[str, int] = {}


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
        github_app=None,  # GitHub App instance, kept for backward compat
        github_api: GitHubAPI | None = None,
        trusted_users: set[str] | None = None,
        post_output_as_comment: bool = False,
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
        self.github_api = github_api
        self.trusted_users = trusted_users
        self.post_output_as_comment = post_output_as_comment
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
    def _bot_logins(self) -> set[str] | None:
        """All login forms for the bot.

        GitHub uses different formats in different API responses:
          "slug"      — e.g. "penny-team"
          "slug[bot]" — e.g. "penny-team[bot]"
          "app/slug"  — e.g. "app/penny-team" (issue/comment author)
        """
        if self.github_app is None:
            return None
        slug = self.github_app._fetch_slug()
        return {slug, self.github_app.bot_name, f"{APP_PREFIX}{slug}"}

    @property
    def _state_path(self) -> Path:
        return DATA_DIR / f"{self.name}.state.json"

    def _load_full_state(self) -> AgentState:
        """Load full agent state from disk.

        Handles backward compat with old flat format (plain timestamp dict).
        """
        try:
            data = json.loads(self._state_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return AgentState()
        if "timestamps" not in data:
            # Old format: flat dict of {number: updatedAt}
            return AgentState(timestamps=data)
        return AgentState.model_validate(data)

    def _save_full_state(self, state: AgentState) -> None:
        """Persist full agent state to disk."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(state.model_dump_json())

    def _load_state(self) -> dict[str, str]:
        """Load saved issue timestamps from disk."""
        return self._load_full_state().timestamps

    def _save_state(self, timestamps: dict[str, str]) -> None:
        """Persist issue timestamps to disk."""
        state = self._load_full_state()
        state.timestamps = timestamps
        self._save_full_state(state)

    def _load_processed(self) -> dict[str, str]:
        """Load per-agent processed issue timestamps."""
        return self._load_full_state().processed

    def _mark_processed(self, issue_number: int) -> None:
        """Record that this agent processed an issue."""
        state = self._load_full_state()
        state.processed[str(issue_number)] = datetime.now().isoformat()
        self._save_full_state(state)

    def _post_comment(self, issue_number: int, body: str) -> bool:
        """Post a comment on a GitHub issue via the GitHub API.

        Returns True if the comment was posted successfully.
        """
        if self.github_api is None:
            logger.warning(f"[{self.name}] No GitHub API configured, cannot post comment")
            return False
        try:
            self.github_api.comment_issue(issue_number, body)
            logger.info(f"[{self.name}] Posted comment on issue #{issue_number}")
            return True
        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"[{self.name}] Failed to post comment on issue #{issue_number}: {e}")
            return False

    def _fetch_issue_timestamps(self) -> dict[str, str]:
        """Fetch updatedAt timestamps for all issues matching required labels.

        Returns a dict mapping issue number (str) to updatedAt timestamp.
        Raises RuntimeError if no API is configured or a query fails.
        """
        if self.github_api is None:
            raise RuntimeError("No GitHub API configured")
        timestamps: dict[str, str] = {}
        for label in self.required_labels or []:
            for item in self.github_api.list_issues(label):
                timestamps[str(item.number)] = item.updated_at
        return timestamps

    def has_work(self) -> bool:
        """Check if there are new or updated GitHub issues since the last run.

        Returns True if no labels are configured (always run), if any issue
        has been created/updated/removed since the last saved state, or if
        gh fails (fail-open).

        For labels with external state (e.g. in-review where CI checks can
        change without updating issue timestamps), performs a full
        actionability check including CI status, merge conflicts, and
        review feedback.
        """
        if not self.required_labels:
            return True

        try:
            current = self._fetch_issue_timestamps()
        except (OSError, RuntimeError):
            return True

        if not current:
            return False

        saved = self._load_state()
        if current != saved:
            return True

        # Timestamps unchanged. For labels with external state (CI, reviews),
        # check if any issue actually needs attention.
        has_external_state = any(
            label in LABELS_WITH_EXTERNAL_STATE for label in self.required_labels
        )
        if has_external_state:
            return self._check_actionable_issues()

        logger.info(f"[{self.name}] No issue changes since last run, skipping")
        return False

    def _check_actionable_issues(self) -> bool:
        """Check if any issues actually need agent attention.

        Performs the full issue fetch, PR status enrichment, and
        actionability check. Used by has_work() when timestamp
        comparison alone can't determine if work is needed.
        """
        from penny_team.utils.issue_filter import fetch_issues_for_labels, pick_actionable_issue
        from penny_team.utils.pr_checks import enrich_issues_with_pr_status

        if not self.required_labels:
            return True

        try:
            issues = fetch_issues_for_labels(
                self.required_labels,
                trusted_users=self.trusted_users,
                api=self.github_api,
            )
            processed = self._load_processed()
            enrich_issues_with_pr_status(
                issues, api=self.github_api, bot_logins=self._bot_logins, processed_at=processed
            )
            issue = pick_actionable_issue(issues, self._bot_logins, processed)
        except Exception:
            # Fail open — if anything goes wrong, let the agent run
            return True

        if issue is None:
            logger.info(f"[{self.name}] No actionable issues, skipping")
            return False
        return True

    def _build_command(self, prompt: str) -> list[str]:
        cmd = [
            CLAUDE_CLI,
            "-p",
            prompt,
            "--verbose",
            "--output-format",
            "stream-json",
            "--system-prompt",
            "",
        ]
        if self.allowed_tools is None:
            # Full tool access (worker, monitor)
            cmd.append("--dangerously-skip-permissions")
        elif self.allowed_tools:
            # Restricted tool set
            cmd.append("--dangerously-skip-permissions")
            cmd.extend(["--allowedTools", *self.allowed_tools])
        # else: allowed_tools == [] → no tools, no --dangerously-skip-permissions
        if self.model:
            cmd.extend(["--model", self.model])
        return cmd

    def _execute_claude(self, prompt: str) -> tuple[bool, str]:
        """Execute Claude CLI with the given prompt, streaming output.

        Returns (success, result_text) tuple. Handles subprocess creation,
        stream-json event parsing, timeout, and cleanup.
        """
        prompt_path = LOG_DIR / f"{self.name}.prompt.md"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt)

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

            return process.returncode == 0, result_text

        except subprocess.TimeoutExpired:
            process.kill()
            self._process = None
            return False, "Process timed out"

    def run(self) -> AgentRun:
        logger.info(f"[{self.name}] Starting cycle #{self.run_count + 1}")
        start = datetime.now()

        prompt = self.prompt_path.read_text()
        selected_issue = None

        # Pre-fetch, filter, and pick one actionable issue
        if self.required_labels:
            from penny_team.utils.issue_filter import (
                fetch_issues_for_labels,
                format_issues_for_prompt,
                pick_actionable_issue,
            )

            all_issues = fetch_issues_for_labels(
                self.required_labels, trusted_users=self.trusted_users, api=self.github_api
            )

            # Enrich in-review issues with CI and merge conflict status (no-op if none match)
            from penny_team.utils.pr_checks import enrich_issues_with_pr_status

            processed = self._load_processed()
            enrich_issues_with_pr_status(
                all_issues, api=self.github_api, bot_logins=self._bot_logins, processed_at=processed
            )

            issue = pick_actionable_issue(all_issues, self._bot_logins, processed)

            if issue is None:
                duration = (datetime.now() - start).total_seconds()
                self.last_run = datetime.now()
                self.run_count += 1
                logger.info(
                    f"[{self.name}] No actionable issues (bot has last comment on all), skipping"
                )
                try:
                    self._save_state(self._fetch_issue_timestamps())
                except (OSError, RuntimeError) as e:
                    logger.warning(f"[{self.name}] Failed to save issue state: {e}")
                return AgentRun(
                    agent_name=self.name,
                    success=True,
                    output="No actionable issues",
                    duration=duration,
                    timestamp=start,
                )

            selected_issue = issue

            # CI fix attempt cap: if we've tried MAX_CI_FIX_ATTEMPTS times
            # and CI is still failing (with no new review feedback), pause
            # and ask for human help instead of burning more tokens.
            issue_key = str(issue.number)
            state = self._load_full_state()
            attempts = state.ci_fix_attempts.get(issue_key, 0)

            if (
                issue.ci_status == CI_STATUS_FAILING
                and not issue.has_review_feedback
                and attempts >= MAX_CI_FIX_ATTEMPTS
            ):
                duration = (datetime.now() - start).total_seconds()
                self.last_run = datetime.now()
                self.run_count += 1
                msg = (
                    f"*[Worker Agent]*\n\n"
                    f"I've attempted to fix CI {MAX_CI_FIX_ATTEMPTS} times without success. "
                    f"Pausing automated attempts — a human needs to take a look at the "
                    f"failing checks and provide guidance."
                )
                self._post_comment(issue.number, msg)
                self._mark_processed(issue.number)
                logger.warning(
                    f"[{self.name}] CI fix attempt limit ({MAX_CI_FIX_ATTEMPTS}) reached "
                    f"for issue #{issue.number}, pausing"
                )
                return AgentRun(
                    agent_name=self.name,
                    success=True,
                    output=f"CI fix attempt limit reached for issue #{issue.number}",
                    duration=duration,
                    timestamp=start,
                )

            # Reset CI fix counter when CI passes or human has provided feedback
            if (
                issue.ci_status != CI_STATUS_FAILING or issue.has_review_feedback
            ) and issue_key in state.ci_fix_attempts:
                del state.ci_fix_attempts[issue_key]
                self._save_full_state(state)

            prompt += format_issues_for_prompt([issue])

        success, result_text = self._execute_claude(prompt)

        if success and selected_issue is not None:
            # If post_output_as_comment is enabled and result is empty, don't
            # mark as processed — empty output means the agent failed to produce
            # a response, not that it successfully handled the issue.
            if self.post_output_as_comment and not result_text:
                success = False

            if (
                self.post_output_as_comment
                and result_text
                and not self._post_comment(selected_issue.number, result_text)
            ):
                success = False

            if success:
                self._mark_processed(selected_issue.number)

                # Increment CI fix attempt counter if this run was for failing CI
                if (
                    selected_issue.ci_status == CI_STATUS_FAILING
                    and not selected_issue.has_review_feedback
                ):
                    s = self._load_full_state()
                    key = str(selected_issue.number)
                    s.ci_fix_attempts[key] = s.ci_fix_attempts.get(key, 0) + 1
                    self._save_full_state(s)

        duration = (datetime.now() - start).total_seconds()
        self.last_run = datetime.now()
        self.run_count += 1

        level = logging.INFO if success else logging.ERROR
        status = "OK" if success else "FAILED"
        logger.log(level, f"[{self.name}] Cycle #{self.run_count} {status} in {duration:.1f}s")

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

"""Quality agent: evaluates Penny's responses, files bugs for low-quality output.

Reads message pairs from Penny's database, evaluates response quality
using Ollama, and files GitHub issues for bad responses. Privacy is
enforced by Python-level substring validation before any issue is filed.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from github_api.api import GitHubAPI

from penny_team.base import Agent, AgentRun
from penny_team.constants import (
    OLLAMA_CHAT_ENDPOINT,
    OLLAMA_DEFAULT_URL,
    PENNY_DB_RELATIVE_PATH,
    QUALITY_LABELS,
    QUALITY_MAX_ISSUES_PER_CYCLE,
    QUALITY_MAX_LOOKBACK_HOURS,
    QUALITY_PRIVACY_MIN_LENGTH,
    QUALITY_STATE_TIMESTAMP,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data models
# =============================================================================


@dataclass
class MessagePair:
    """A user message and Penny's response, with optional prompt log data."""

    outgoing_id: int
    timestamp: str
    user_message: str
    response: str
    sender: str
    thinking: str | None = None
    prompt_messages: str | None = None
    model_response: str | None = None


@dataclass
class QualityIssue:
    """A quality problem detected in a message pair."""

    category: str
    reason: str


# =============================================================================
# Prompt templates
# =============================================================================

EVALUATION_SYSTEM_PROMPT = """\
You are a quality reviewer for an AI chat assistant called Penny.
Evaluate the message pair below. Determine if the response is low-quality.

Quality issues to flag:
- Exposed function/tool calls (e.g., <function=search> syntax leaked to user)
- Error messages or stack traces leaked to user
- Empty or extremely short responses (< 10 words for a substantive question)
- Response doesn't address the user's question at all
- Timeout/truncation indicators
- Garbled or malformed output

If the response has a quality issue, return a JSON object with:
- "is_bad": true
- "category": one of "exposed_function_call", "leaked_error", \
"empty_response", "off_topic", "truncated", "malformed"
- "reason": detailed explanation of the problem — describe the nature of the \
failure, what the user likely expected, and what went wrong, so that someone \
unfamiliar with the original conversation could write a meaningful bug report

If the response is acceptable, return {"is_bad": false}.\
"""

BUG_DESCRIPTION_SYSTEM_PROMPT = """\
Write a GitHub bug report for a quality issue in Penny (an AI chat assistant).

CRITICAL PRIVACY RULE: Do NOT include the actual user message or Penny's \
actual response. Instead, describe the problem and use a CONTRIVED example \
that illustrates the same issue but with completely different content.

Return a JSON object with "title" and "body" fields.
The title should start with "bug: " and be under 60 characters.
The body should be markdown with these sections:
- Start with "*[Quality Agent]*" on its own line
- "## Quality Issue" describing the problem category
- "## Example (Contrived)" with a made-up similar example
- "## Suggested Fix" with where to investigate\
"""


# =============================================================================
# Privacy validation
# =============================================================================


def validate_privacy(original_messages: list[str], issue_body: str) -> bool:
    """Assert no original content appears in the issue body.

    Checks that none of the original user messages or Penny responses
    appear as substrings in the proposed issue body. Short messages
    (< QUALITY_PRIVACY_MIN_LENGTH chars) are skipped to avoid false
    positives on common phrases like "yes", "thanks", etc.

    Returns True if the issue body is safe to file.
    """
    body_lower = issue_body.lower()
    for msg in original_messages:
        if len(msg) < QUALITY_PRIVACY_MIN_LENGTH:
            continue
        if msg.lower() in body_lower:
            return False
    return True


# =============================================================================
# QualityAgent
# =============================================================================


class QualityAgent(Agent):
    """Agent that evaluates Penny's response quality and files bug issues.

    Overrides has_work() and run() to read from Penny's database instead
    of GitHub issues. Uses Ollama for quality evaluation (not Claude CLI).
    Files issues via GitHubAPI with privacy validation.
    """

    def __init__(
        self,
        ollama_url: str = OLLAMA_DEFAULT_URL,
        ollama_model: str = "",
        db_path: str | Path | None = None,
        name: str = "quality",
        interval_seconds: int = 3600,
        working_dir: Path | None = None,
        timeout_seconds: int = 600,
        github_app=None,
        github_api: GitHubAPI | None = None,
        trusted_users: set[str] | None = None,
    ) -> None:
        from penny_team.base import PROJECT_ROOT

        super().__init__(
            name=name,
            interval_seconds=interval_seconds,
            working_dir=working_dir or PROJECT_ROOT,
            timeout_seconds=timeout_seconds,
            github_app=github_app,
            github_api=github_api,
            trusted_users=trusted_users,
        )
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_model = ollama_model
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            self.db_path = PROJECT_ROOT / PENNY_DB_RELATIVE_PATH

    # --- State management ---

    def _load_last_timestamp(self) -> str:
        """Load the last processed message timestamp from state file.

        On first run (no saved state), defaults to QUALITY_MAX_LOOKBACK_HOURS
        ago to avoid re-evaluating the entire message history.
        """
        state = self._load_state()
        saved = state.get(QUALITY_STATE_TIMESTAMP, "")
        if saved:
            return saved
        cutoff = datetime.now(UTC) - timedelta(hours=QUALITY_MAX_LOOKBACK_HOURS)
        return cutoff.isoformat()

    def _save_last_timestamp(self, timestamp: str) -> None:
        """Persist the last processed message timestamp."""
        self._save_state({QUALITY_STATE_TIMESTAMP: timestamp})

    # --- has_work() override ---

    def has_work(self) -> bool:
        """Check if Penny's database has new messages since last run."""
        if not self.db_path.exists():
            logger.info(f"[{self.name}] Database not found: {self.db_path}")
            return False

        last_ts = self._load_last_timestamp()

        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            try:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM messagelog "
                    "WHERE direction = 'outgoing' AND timestamp > ?",
                    (last_ts,),
                )
                count = cursor.fetchone()[0]
            finally:
                conn.close()
        except (sqlite3.Error, OSError) as e:
            logger.warning(f"[{self.name}] Error checking database: {e}")
            return True  # Fail-open

        if count == 0:
            logger.info(f"[{self.name}] No new messages since last run")
            return False
        return True

    # --- Database reading ---

    def _read_message_pairs(self) -> list[MessagePair]:
        """Read message pairs from Penny's database since last run.

        Joins outgoing messages with their parent incoming messages.
        Also attempts to find the closest PromptLog entry for each pair.
        """
        last_ts = self._load_last_timestamp()

        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT o.id, o.timestamp, o.content AS response, "
                "       i.content AS user_message, i.sender "
                "FROM messagelog o "
                "JOIN messagelog i ON o.parent_id = i.id "
                "WHERE o.direction = 'outgoing' "
                "  AND i.direction = 'incoming' "
                "  AND o.timestamp > ? "
                "ORDER BY o.timestamp ASC",
                (last_ts,),
            ).fetchall()

            pairs = []
            for row in rows:
                pair = MessagePair(
                    outgoing_id=row["id"],
                    timestamp=row["timestamp"],
                    user_message=row["user_message"],
                    response=row["response"],
                    sender=row["sender"],
                )
                # Try to find the closest PromptLog entry by timestamp
                prompt_row = conn.execute(
                    "SELECT messages, response, thinking FROM promptlog "
                    "WHERE timestamp <= ? "
                    "ORDER BY timestamp DESC LIMIT 1",
                    (row["timestamp"],),
                ).fetchone()
                if prompt_row:
                    pair.thinking = prompt_row["thinking"]
                    pair.prompt_messages = prompt_row["messages"]
                    pair.model_response = prompt_row["response"]
                pairs.append(pair)
        finally:
            conn.close()

        return pairs

    # --- Ollama API ---

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> dict | None:
        """Call Ollama chat API and return parsed JSON response.

        Returns None on failure (network error, parse error, etc.).
        """
        url = f"{self.ollama_url}{OLLAMA_CHAT_ENDPOINT}"
        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
        }

        try:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(url, data=data, method="POST")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                result = json.loads(resp.read())
            content = result.get("message", {}).get("content", "")
            return json.loads(content)
        except (OSError, json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"[{self.name}] Ollama call failed: {e}")
            return None

    # --- Evaluation ---

    def _evaluate_pair(self, pair: MessagePair) -> QualityIssue | None:
        """Evaluate a single message pair via Ollama.

        Returns a QualityIssue if the response is bad, or None if acceptable.
        """
        user_prompt = f"**User**: {pair.user_message}\n**Penny**: {pair.response}\n"
        if pair.thinking:
            user_prompt += f"**Thinking**: {pair.thinking}\n"

        result = self._call_ollama(EVALUATION_SYSTEM_PROMPT, user_prompt)
        if result is None:
            return None

        if not result.get("is_bad"):
            return None

        try:
            return QualityIssue(
                category=str(result["category"]),
                reason=str(result["reason"]),
            )
        except (KeyError, TypeError) as e:
            logger.warning(f"[{self.name}] Skipping malformed evaluation result: {e}")
            return None

    # --- Bug description generation ---

    def _generate_bug_description(
        self, issue: QualityIssue, pair: MessagePair
    ) -> tuple[str, str] | None:
        """Generate a privacy-safe bug description via Ollama.

        Returns (title, body) tuple, or None on failure.
        """
        user_prompt = (
            f"Category: {issue.category}\n"
            f"Reason: {issue.reason}\n\n"
            f"The problematic response was about {len(pair.response)} characters long. "
            f"The user sent a message and received a response exhibiting the issue described above."
        )
        result = self._call_ollama(BUG_DESCRIPTION_SYSTEM_PROMPT, user_prompt)

        if result is None:
            return None

        title = result.get("title", "")
        body = result.get("body", "")
        if not title or not body:
            logger.warning(f"[{self.name}] Empty title or body from Ollama")
            return None

        return title, body

    # --- run() override ---

    def run(self) -> AgentRun:
        """Read recent messages, evaluate quality, file issues for bad responses."""
        logger.info(f"[{self.name}] Starting cycle #{self.run_count + 1}")
        start = datetime.now()

        # Step 1: Read message pairs
        try:
            pairs = self._read_message_pairs()
        except (sqlite3.Error, OSError) as e:
            duration = (datetime.now() - start).total_seconds()
            self.last_run = datetime.now()
            self.run_count += 1
            logger.error(f"[{self.name}] Failed to read database: {e}")
            return AgentRun(
                agent_name=self.name,
                success=False,
                output=f"Failed to read database: {e}",
                duration=duration,
                timestamp=start,
            )

        if not pairs:
            # Save timestamp even when no pairs (e.g., outgoing messages with no parent)
            self._advance_timestamp(pairs)
            duration = (datetime.now() - start).total_seconds()
            self.last_run = datetime.now()
            self.run_count += 1
            logger.info(f"[{self.name}] No message pairs to evaluate")
            return AgentRun(
                agent_name=self.name,
                success=True,
                output="No message pairs to evaluate",
                duration=duration,
                timestamp=start,
            )

        logger.info(f"[{self.name}] Evaluating {len(pairs)} message pair(s)")

        # Step 2: Evaluate each pair individually and file issues
        found = 0
        filed = 0
        total = len(pairs)
        for idx, pair in enumerate(pairs, 1):
            if filed >= QUALITY_MAX_ISSUES_PER_CYCLE:
                break

            logger.info(
                f"[{self.name}] Evaluating pair {idx}/{total} (message #{pair.outgoing_id})"
            )
            issue = self._evaluate_pair(pair)
            if issue is None:
                continue

            found += 1
            logger.info(
                f"[{self.name}] Quality issue in message {pair.outgoing_id}: {issue.category}"
            )

            # Generate privacy-safe bug description
            description = self._generate_bug_description(issue, pair)
            if description is None:
                continue

            title, body = description

            # Validate privacy before filing
            original_content = [pair.user_message, pair.response]
            if not validate_privacy(original_content, body):
                logger.error(
                    f"[{self.name}] Privacy validation failed for message {pair.outgoing_id} "
                    f"— original content detected in issue body, skipping"
                )
                continue

            # File the issue
            if self.github_api is None:
                logger.warning(f"[{self.name}] No GitHub API configured, cannot file issue")
                continue

            try:
                url = self.github_api.create_issue(title, body, QUALITY_LABELS)
                logger.info(f"[{self.name}] Filed issue: {url}")
                filed += 1
            except (OSError, RuntimeError, ValueError) as e:
                logger.error(f"[{self.name}] Failed to file issue: {e}")

        # Step 3: Advance timestamp
        self._advance_timestamp(pairs)

        duration = (datetime.now() - start).total_seconds()
        self.last_run = datetime.now()
        self.run_count += 1

        if found == 0:
            output = "All responses passed quality check"
        else:
            output = f"Filed {filed} issue(s) from {found} quality problem(s)"

        logger.info(f"[{self.name}] Cycle #{self.run_count} OK in {duration:.1f}s")

        return AgentRun(
            agent_name=self.name,
            success=True,
            output=output,
            duration=duration,
            timestamp=start,
        )

    def _advance_timestamp(self, pairs: list[MessagePair]) -> None:
        """Save the timestamp of the latest message pair for next run."""
        if pairs:
            self._save_last_timestamp(pairs[-1].timestamp)

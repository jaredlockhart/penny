"""PR status detection for worker agent PRs.

Fetches PR check statuses, merge conflict status, and review feedback
via gh CLI and enriches FilteredIssue objects with CI failure details,
merge conflict information, and review state. This enables the worker
agent to detect and fix failing checks, rebase conflicting branches,
and address review feedback.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass

from penny_team.base import GH_CLI
from penny_team.utils.issue_filter import CI_STATUS_FAILING, CI_STATUS_PASSING, FilteredIssue

LABEL_IN_REVIEW = "in-review"

# GitHub check conclusions that count as passing
PASSING_CONCLUSIONS = {"SUCCESS", "NEUTRAL", "SKIPPED", ""}

# statusCheckRollup states that mean "still running"
PENDING_STATES = {"PENDING", "QUEUED", "IN_PROGRESS", "EXPECTED"}

# gh CLI JSON fields for PR list
PR_FIELDS = "number,headRefName,statusCheckRollup,mergeable,reviews"

# GitHub review states that indicate feedback needing attention
REVIEW_STATE_CHANGES_REQUESTED = "CHANGES_REQUESTED"

# Max characters of failure log to include in prompt
MAX_LOG_CHARS = 3000

logger = logging.getLogger(__name__)


@dataclass
class FailedCheck:
    """A single failing CI check."""

    name: str
    conclusion: str


MERGE_STATUS_CONFLICTING = "CONFLICTING"


def enrich_issues_with_pr_status(
    issues: list[FilteredIssue],
    env: dict[str, str] | None = None,
) -> None:
    """Enrich in-review issues with CI check and merge conflict status from their PRs.

    Mutates FilteredIssue objects in place, setting ci_status,
    ci_failure_details, merge_conflict, and merge_conflict_branch.
    Fail-open: if gh fails, issues are left unchanged.
    """
    in_review = [i for i in issues if LABEL_IN_REVIEW in i.labels]
    if not in_review:
        return

    try:
        prs = _fetch_open_prs(env)
    except (subprocess.TimeoutExpired, OSError, RuntimeError):
        logger.warning("Failed to fetch PR statuses, skipping CI/merge detection")
        return

    pr_by_issue = _match_prs_to_issues(prs, in_review)

    for issue in in_review:
        pr = pr_by_issue.get(issue.number)
        if pr is None:
            continue

        # Merge conflict detection
        if pr.get("mergeable", "") == MERGE_STATUS_CONFLICTING:
            issue.merge_conflict = True
            issue.merge_conflict_branch = pr["headRefName"]

        # Review feedback detection (latest review per reviewer wins)
        if _has_changes_requested(pr.get("reviews", [])):
            issue.has_review_feedback = True

        # CI check detection
        failed = _extract_failed_checks(pr.get("statusCheckRollup", []))

        if not failed:
            issue.ci_status = CI_STATUS_PASSING
            continue

        issue.ci_status = CI_STATUS_FAILING

        branch = pr["headRefName"]
        log_output = _fetch_failure_log(branch, env)

        check_names = ", ".join(f.name for f in failed)
        details = f"**Failing checks**: {check_names}\n"
        if log_output:
            details += f"\n**Error output** (truncated):\n```\n{log_output}\n```"
        issue.ci_failure_details = details


def _fetch_open_prs(
    env: dict[str, str] | None = None,
) -> list[dict]:
    """Fetch all open PRs with check status data."""
    result = subprocess.run(
        [GH_CLI, "pr", "list", "--state", "open", "--json", PR_FIELDS, "--limit", "20"],
        capture_output=True,
        text=True,
        timeout=15,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh pr list failed: {result.stderr}")
    return json.loads(result.stdout)


def _match_prs_to_issues(
    prs: list[dict],
    issues: list[FilteredIssue],
) -> dict[int, dict]:
    """Match PRs to issues by branch naming convention (issue-N-*)."""
    issue_numbers = {i.number for i in issues}
    result: dict[int, dict] = {}
    for pr in prs:
        branch = pr.get("headRefName", "")
        parts = branch.split("-", 2)
        if len(parts) >= 2 and parts[0] == "issue":
            try:
                num = int(parts[1])
                if num in issue_numbers:
                    result[num] = pr
            except ValueError:
                continue
    return result


def _has_changes_requested(reviews: list[dict]) -> bool:
    """Check if any reviewer's latest review requests changes.

    A reviewer might request changes then later approve. We only care
    about each reviewer's most recent review (last in the list).
    """
    latest_by_reviewer: dict[str, str] = {}
    for review in reviews:
        login = review.get("author", {}).get("login", "")
        state = review.get("state", "")
        if login and state:
            latest_by_reviewer[login] = state
    return any(state == REVIEW_STATE_CHANGES_REQUESTED for state in latest_by_reviewer.values())


def _extract_failed_checks(status_rollup: list[dict]) -> list[FailedCheck]:
    """Extract failing checks from statusCheckRollup data."""
    failed: list[FailedCheck] = []
    for check in status_rollup:
        state = check.get("state", "")
        conclusion = check.get("conclusion", "")
        if state in PENDING_STATES:
            continue
        if conclusion not in PASSING_CONCLUSIONS:
            failed.append(
                FailedCheck(
                    name=check.get("context", check.get("name", "unknown")),
                    conclusion=conclusion,
                )
            )
    return failed


def _fetch_failure_log(
    branch: str,
    env: dict[str, str] | None = None,
) -> str:
    """Fetch truncated log output from the most recent failing run."""
    try:
        result = subprocess.run(
            [
                GH_CLI,
                "run",
                "list",
                "--branch",
                branch,
                "--status",
                "failure",
                "--json",
                "databaseId",
                "--limit",
                "1",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return ""

        runs = json.loads(result.stdout)
        if not runs:
            return ""

        run_id = str(runs[0]["databaseId"])

        result = subprocess.run(
            [GH_CLI, "run", "view", run_id, "--log-failed"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        if result.returncode != 0:
            return ""

        log = result.stdout.strip()
        if len(log) > MAX_LOG_CHARS:
            log = log[-MAX_LOG_CHARS:]
            log = f"... (truncated)\n{log}"
        return log

    except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to fetch failure log for branch {branch}: {e}")
        return ""

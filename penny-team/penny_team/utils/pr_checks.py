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

from pydantic import BaseModel, Field

from penny_team.constants import (
    API_PR_REVIEW_COMMENTS,
    CI_STATUS_FAILING,
    CI_STATUS_PASSING,
    GH_CLI,
    GH_PR_FIELDS,
    MAX_LOG_CHARS,
    MERGE_STATUS_CONFLICTING,
    PASSING_CONCLUSIONS,
    PENDING_STATES,
    REVIEW_STATE_CHANGES_REQUESTED,
    Label,
)
from penny_team.utils.issue_filter import FilteredIssue

logger = logging.getLogger(__name__)


class CommentAuthor(BaseModel):
    """Author of a PR comment or review."""

    login: str = ""


class PRComment(BaseModel):
    """A top-level PR comment (from gh pr list --json comments)."""

    author: CommentAuthor = CommentAuthor()
    body: str = ""
    created_at: str = Field("", alias="createdAt")


class PRReview(BaseModel):
    """A formal PR review (from gh pr list --json reviews)."""

    author: CommentAuthor = CommentAuthor()
    state: str = ""
    submitted_at: str = Field("", alias="submittedAt")


class ReviewCommentUser(BaseModel):
    """Author of an inline PR review comment (REST API format)."""

    login: str = ""


class ReviewComment(BaseModel):
    """An inline review comment on a pull request (from REST API)."""

    user: ReviewCommentUser = ReviewCommentUser()
    body: str = ""
    path: str = ""
    created_at: str = ""


@dataclass
class FailedCheck:
    """A single failing CI check."""

    name: str
    conclusion: str


def enrich_issues_with_pr_status(
    issues: list[FilteredIssue],
    env: dict[str, str] | None = None,
    bot_logins: set[str] | None = None,
    processed_at: dict[str, str] | None = None,
) -> None:
    """Enrich in-review issues with CI check and merge conflict status from their PRs.

    Mutates FilteredIssue objects in place, setting ci_status,
    ci_failure_details, merge_conflict, and merge_conflict_branch.
    When processed_at is provided, only review feedback newer than
    the agent's last processing time is included (prevents re-addressing
    already-handled comments).
    Fail-open: if gh fails, issues are left unchanged.
    """
    in_review = [i for i in issues if Label.IN_REVIEW in i.labels]
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

        # Timestamp of when the agent last processed this issue —
        # comments older than this have already been addressed.
        since = (processed_at or {}).get(str(issue.number))

        # Merge conflict detection
        if pr.get("mergeable", "") == MERGE_STATUS_CONFLICTING:
            issue.merge_conflict = True
            issue.merge_conflict_branch = pr["headRefName"]

        # Review feedback detection: formal reviews, top-level comments, or inline review comments
        # Only include feedback newer than when we last processed this issue.
        review_parts: list[str] = []

        if _has_changes_requested(pr.get("reviews", []), since=since):
            issue.has_review_feedback = True

        human_comments = _collect_human_comments(pr.get("comments", []), bot_logins, since=since)
        if human_comments:
            issue.has_review_feedback = True
            review_parts.append("**PR comments:**\n")
            review_parts.extend(human_comments)

        inline_comments = _collect_human_review_comments(pr["number"], bot_logins, env, since=since)
        if inline_comments:
            issue.has_review_feedback = True
            review_parts.append("**Inline review comments (on specific code lines):**\n")
            review_parts.extend(inline_comments)

        if review_parts:
            issue.review_comments = "\n".join(review_parts)

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
        [GH_CLI, "pr", "list", "--state", "open", "--json", GH_PR_FIELDS, "--limit", "20"],
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


def _has_changes_requested(raw_reviews: list[dict], since: str | None = None) -> bool:
    """Check if any reviewer's latest review requests changes.

    A reviewer might request changes then later approve. We only care
    about each reviewer's most recent review (last in the list).
    When since is set, only reviews submitted after that ISO timestamp
    count (already-addressed reviews are ignored).
    """
    latest_by_reviewer: dict[str, PRReview] = {}
    for raw in raw_reviews:
        review = PRReview.model_validate(raw)
        if review.author.login and review.state:
            latest_by_reviewer[review.author.login] = review
    for review in latest_by_reviewer.values():
        if review.state == REVIEW_STATE_CHANGES_REQUESTED and (
            since is None or not review.submitted_at or review.submitted_at > since
        ):
            return True
    return False


def _collect_human_review_comments(
    pr_number: int,
    bot_logins: set[str] | None,
    env: dict[str, str] | None,
    since: str | None = None,
) -> list[str]:
    """Fetch inline review comments from human (non-bot) users.

    These are comments left on specific lines of code during a review,
    which are not included in gh pr list --json comments or reviews.
    When since is set, only comments created after that ISO timestamp
    are included (filters out already-addressed feedback).
    Fail-open: returns empty list if the API call fails.
    """
    try:
        api_path = API_PR_REVIEW_COMMENTS.format(pr_number=pr_number)
        result = subprocess.run(
            [GH_CLI, "api", api_path],
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
        )
        if result.returncode != 0:
            return []
        raw_comments = json.loads(result.stdout)
        parts: list[str] = []
        for raw in raw_comments:
            comment = ReviewComment.model_validate(raw)
            if since and comment.created_at and comment.created_at <= since:
                continue
            if comment.user.login and (bot_logins is None or comment.user.login not in bot_logins):
                location = f" (`{comment.path}`)" if comment.path else ""
                parts.append(f"**{comment.user.login}**{location}:\n{comment.body}\n")
        return parts
    except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError, ValueError):
        return []


def _collect_human_comments(
    raw_comments: list[dict],
    bot_logins: set[str] | None,
    since: str | None = None,
) -> list[str]:
    """Collect PR comments authored by human (non-bot) users.

    When since is set, only comments created after that ISO timestamp
    are included (filters out already-addressed feedback).
    """
    parts: list[str] = []
    for raw in raw_comments:
        comment = PRComment.model_validate(raw)
        if not comment.author.login:
            continue
        if since and comment.created_at and comment.created_at <= since:
            continue
        if bot_logins is None or comment.author.login not in bot_logins:
            parts.append(f"**{comment.author.login}**:\n{comment.body}\n")
    return parts


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

"""PR status detection for worker agent PRs.

Fetches PR check statuses, merge conflict status, and review feedback
via the GitHub API and enriches FilteredIssue objects with CI failure
details, merge conflict information, and review state. This enables the
worker agent to detect and fix failing checks, rebase conflicting
branches, and address review feedback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from penny_team.constants import (
    CI_STATUS_FAILING,
    CI_STATUS_PASSING,
    MAX_LOG_CHARS,
    MERGE_STATUS_CONFLICTING,
    PASSING_CONCLUSIONS,
    PENDING_STATES,
    REVIEW_STATE_CHANGES_REQUESTED,
    Label,
)
from penny_team.utils.github_api import (
    CheckStatus,
    GitHubAPI,
    PRComment,
    PRReview,
    PullRequest,
)
from penny_team.utils.issue_filter import FilteredIssue

logger = logging.getLogger(__name__)


@dataclass
class FailedCheck:
    """A single failing CI check."""

    name: str
    conclusion: str


def enrich_issues_with_pr_status(
    issues: list[FilteredIssue],
    api: GitHubAPI | None = None,
    bot_logins: set[str] | None = None,
    processed_at: dict[str, str] | None = None,
) -> None:
    """Enrich in-review issues with CI check and merge conflict status from their PRs.

    Mutates FilteredIssue objects in place, setting ci_status,
    ci_failure_details, merge_conflict, and merge_conflict_branch.
    When processed_at is provided, only review feedback newer than
    the agent's last processing time is included (prevents re-addressing
    already-handled comments).
    Fail-open: if the API fails, issues are left unchanged.
    """
    in_review = [i for i in issues if Label.IN_REVIEW in i.labels]
    if not in_review:
        return

    try:
        prs = _fetch_open_prs(api)
    except (OSError, RuntimeError):
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
        if pr.mergeable == MERGE_STATUS_CONFLICTING:
            issue.merge_conflict = True
            issue.merge_conflict_branch = pr.head_ref_name

        # Review feedback detection: formal reviews, top-level comments, or inline review comments
        # Only include feedback newer than when we last processed this issue.
        review_parts: list[str] = []

        if _has_changes_requested(pr.reviews, since=since):
            issue.has_review_feedback = True
            feedback = _collect_review_feedback(pr.reviews, since=since)
            if feedback:
                review_parts.append("**Review feedback:**\n")
                review_parts.extend(feedback)

        human_comments = _collect_human_comments(pr.comments, bot_logins, since=since)
        if human_comments:
            issue.has_review_feedback = True
            review_parts.append("**PR comments:**\n")
            review_parts.extend(human_comments)

        inline_comments = _collect_human_review_comments(pr.number, bot_logins, api, since=since)
        if inline_comments:
            issue.has_review_feedback = True
            review_parts.append("**Inline review comments (on specific code lines):**\n")
            review_parts.extend(inline_comments)

        if review_parts:
            issue.review_comments = "\n".join(review_parts)

        # CI check detection
        failed = _extract_failed_checks(pr.status_check_rollup)

        if not failed:
            issue.ci_status = CI_STATUS_PASSING
            continue

        issue.ci_status = CI_STATUS_FAILING

        branch = pr.head_ref_name
        log_output = _fetch_failure_log(branch, api)

        check_names = ", ".join(f.name for f in failed)
        details = f"**Failing checks**: {check_names}\n"
        if log_output:
            details += f"\n**Error output** (truncated):\n```\n{log_output}\n```"
        issue.ci_failure_details = details


def _fetch_open_prs(
    api: GitHubAPI | None = None,
) -> list[PullRequest]:
    """Fetch all open PRs with check status data."""
    if api is None:
        raise RuntimeError("No GitHub API configured")
    return api.list_open_prs()


def _match_prs_to_issues(
    prs: list[PullRequest],
    issues: list[FilteredIssue],
) -> dict[int, PullRequest]:
    """Match PRs to issues by branch naming convention (issue-N-*)."""
    issue_numbers = {i.number for i in issues}
    result: dict[int, PullRequest] = {}
    for pr in prs:
        branch = pr.head_ref_name
        parts = branch.split("-", 2)
        if len(parts) >= 2 and parts[0] == "issue":
            try:
                num = int(parts[1])
                if num in issue_numbers:
                    result[num] = pr
            except ValueError:
                continue
    return result


def _collect_review_feedback(reviews: list[PRReview], since: str | None = None) -> list[str]:
    """Collect body text from CHANGES_REQUESTED reviews.

    Uses the same latest-per-reviewer logic as _has_changes_requested:
    only the most recent review per reviewer is considered, and the
    since filter excludes already-addressed feedback.
    """
    latest_by_reviewer: dict[str, PRReview] = {}
    for review in reviews:
        if review.author.login and review.state:
            latest_by_reviewer[review.author.login] = review
    parts: list[str] = []
    for review in latest_by_reviewer.values():
        if (
            review.state == REVIEW_STATE_CHANGES_REQUESTED
            and (since is None or not review.submitted_at or review.submitted_at > since)
            and review.body
        ):
            parts.append(f"**{review.author.login}** (changes requested):\n{review.body}\n")
    return parts


def _has_changes_requested(reviews: list[PRReview], since: str | None = None) -> bool:
    """Check if any reviewer's latest review requests changes.

    A reviewer might request changes then later approve. We only care
    about each reviewer's most recent review (last in the list).
    When since is set, only reviews submitted after that ISO timestamp
    count (already-addressed reviews are ignored).
    """
    latest_by_reviewer: dict[str, PRReview] = {}
    for review in reviews:
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
    api: GitHubAPI | None,
    since: str | None = None,
) -> list[str]:
    """Fetch inline review comments from human (non-bot) users.

    These are comments left on specific lines of code during a review,
    which are not included in the PR comments or reviews from GraphQL.
    When since is set, only comments created after that ISO timestamp
    are included (filters out already-addressed feedback).
    Fail-open: returns empty list if the API call fails.
    """
    if api is None:
        return []
    try:
        comments = api.list_pr_review_comments(pr_number)
        parts: list[str] = []
        for comment in comments:
            if since and comment.created_at and comment.created_at <= since:
                continue
            if comment.user.login and (bot_logins is None or comment.user.login not in bot_logins):
                location = f" (`{comment.path}`)" if comment.path else ""
                parts.append(f"**{comment.user.login}**{location}:\n{comment.body}\n")
        return parts
    except (OSError, ValueError, RuntimeError):
        return []


def _collect_human_comments(
    comments: list[PRComment],
    bot_logins: set[str] | None,
    since: str | None = None,
) -> list[str]:
    """Collect PR comments authored by human (non-bot) users.

    When since is set, only comments created after that ISO timestamp
    are included (filters out already-addressed feedback).
    """
    parts: list[str] = []
    for comment in comments:
        if not comment.author.login:
            continue
        if since and comment.created_at and comment.created_at <= since:
            continue
        if bot_logins is None or comment.author.login not in bot_logins:
            parts.append(f"**{comment.author.login}**:\n{comment.body}\n")
    return parts


def _extract_failed_checks(status_rollup: list[CheckStatus]) -> list[FailedCheck]:
    """Extract failing checks from statusCheckRollup data."""
    failed: list[FailedCheck] = []
    for check in status_rollup:
        if check.state in PENDING_STATES:
            continue
        if check.conclusion not in PASSING_CONCLUSIONS:
            failed.append(
                FailedCheck(
                    name=check.name,
                    conclusion=check.conclusion,
                )
            )
    return failed


def _fetch_failure_log(
    branch: str,
    api: GitHubAPI | None = None,
) -> str:
    """Fetch truncated log output from the most recent failing run."""
    if api is None:
        return ""
    try:
        runs = api.list_failed_runs(branch, limit=1)
        if not runs:
            return ""

        log = api.get_failed_job_log(runs[0].id)
        if not log:
            return ""

        if len(log) > MAX_LOG_CHARS:
            log = log[-MAX_LOG_CHARS:]
            log = f"... (truncated)\n{log}"
        return log

    except (OSError, RuntimeError, ValueError) as e:
        logger.warning(f"Failed to fetch failure log for branch {branch}: {e}")
        return ""

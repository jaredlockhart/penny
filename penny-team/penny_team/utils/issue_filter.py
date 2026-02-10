"""GitHub issue fetcher with trust-based content filtering.

Fetches issues via the GitHub GraphQL API and strips content from
authors not listed in CODEOWNERS. This prevents prompt injection
through GitHub issue bodies and comments on public repositories.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from penny_team.constants import (
    CI_STATUS_FAILING,
    Label,
)
from penny_team.utils.github_api import GitHubAPI, IssueDetail

logger = logging.getLogger(__name__)


@dataclass
class FilteredComment:
    """A single comment from a trusted author."""

    author: str
    body: str
    created_at: str


@dataclass
class FilteredIssue:
    """An issue with only trusted content preserved."""

    number: int
    title: str
    body: str
    author: str
    labels: list[str] = field(default_factory=list)
    trusted_comments: list[FilteredComment] = field(default_factory=list)
    author_is_trusted: bool = True
    ci_status: str | None = None
    ci_failure_details: str | None = None
    merge_conflict: bool = False
    merge_conflict_branch: str | None = None
    has_review_feedback: bool = False
    review_comments: str | None = None


def fetch_issues_for_labels(
    labels: list[str],
    trusted_users: set[str] | None = None,
    api: GitHubAPI | None = None,
) -> list[FilteredIssue]:
    """Fetch all open issues matching any label, with untrusted content filtered out.

    Uses OR logic across labels — an issue matching any label is included.
    When trusted_users is None, all content is included unfiltered.
    Skips labels where the API fails; may return partial results.
    """
    if api is None:
        return []

    issues: list[FilteredIssue] = []
    seen_numbers: set[int] = set()

    for label in labels:
        try:
            details = api.list_issues_detailed(label)

            for detail in details:
                if detail.number in seen_numbers:
                    continue

                filtered = _filter_issue(detail, trusted_users)
                issues.append(filtered)
                seen_numbers.add(detail.number)

        except (OSError, RuntimeError, ValueError) as e:
            logger.error(f"Failed to list issues for label '{label}': {e}")

    return issues


def _filter_issue(
    detail: IssueDetail,
    trusted_users: set[str] | None,
) -> FilteredIssue:
    """Apply trust filtering to an issue detail from the API."""
    author_login = detail.author.login
    author_trusted = trusted_users is None or author_login in trusted_users

    # Filter title and body: only include if author is trusted
    title = detail.title if author_trusted else "[Title hidden: untrusted author]"
    body = detail.body if author_trusted else ""
    if not author_trusted:
        logger.warning(
            f"Issue #{detail.number}: title/body filtered "
            f"(author '{author_login}' not in CODEOWNERS)"
        )

    # Filter comments: only include comments from trusted users
    trusted_comments: list[FilteredComment] = []
    for comment in detail.comments:
        comment_author = comment.author.login
        if trusted_users is None or comment_author in trusted_users:
            trusted_comments.append(
                FilteredComment(
                    author=comment_author,
                    body=comment.body,
                    created_at=comment.created_at,
                )
            )
        else:
            logger.info(f"Issue #{detail.number}: comment by '{comment_author}' filtered out")

    labels = [label.name for label in detail.labels]

    return FilteredIssue(
        number=detail.number,
        title=title,
        body=body,
        author=author_login,
        labels=labels,
        trusted_comments=trusted_comments,
        author_is_trusted=author_trusted,
    )


def pick_actionable_issue(
    issues: list[FilteredIssue],
    bot_logins: set[str] | None = None,
    processed_at: dict[str, str] | None = None,
) -> FilteredIssue | None:
    """Pick the first issue that needs agent attention.

    Uses per-agent processed timestamps to determine actionability.
    An issue needs attention if this agent has never processed it, or
    if a human has commented since the agent last processed it.

    processed_at maps issue number (str) to ISO timestamp of when
    this specific agent last processed that issue. This allows agents
    sharing the same bot identity to independently track their work.

    bot_logins is used to distinguish human comments from bot comments
    when checking for new feedback after processing.

    When bot_logins is None (no GitHub App), returns the first issue.
    """
    if not issues:
        return None

    if bot_logins is None:
        return issues[0]

    # Prioritize bugs over non-bugs (external signals like CI/merge/review
    # are handled by early return inside the loop and remain highest priority)
    sorted_issues = sorted(issues, key=lambda i: Label.BUG not in i.labels)

    for issue in sorted_issues:
        # Check if external signals require attention regardless of comments
        if issue.ci_status == CI_STATUS_FAILING:
            return issue
        if issue.merge_conflict:
            return issue
        if issue.has_review_feedback:
            return issue

        issue_key = str(issue.number)
        last_processed = (processed_at or {}).get(issue_key)

        if last_processed is None:
            # Agent has never processed this issue
            if not issue.trusted_comments and Label.IN_REVIEW in issue.labels:
                # in-review with no issue comments — PR already created,
                # and no CI/merge/review issues detected above. Waiting
                # for human review.
                continue
            # New to this agent — needs processing
            return issue

        # Agent has processed this issue before.
        # Check if a human has commented since we last processed it.
        has_new_human_comment = any(
            c.created_at > last_processed and c.author not in bot_logins
            for c in issue.trusted_comments
        )
        if has_new_human_comment:
            return issue

    # All issues handled or waiting for human action — nothing to do
    return None


def format_issues_for_prompt(issues: list[FilteredIssue]) -> str:
    """Format all filtered issues into a prompt section for injection."""
    if not issues:
        return "\n\n# GitHub Issues (Pre-Fetched, Filtered)\n\nNo matching issues found.\n"

    header = (
        "\n\n# GitHub Issues (Pre-Fetched, Filtered)\n\n"
        "The following issue content has been pre-fetched and filtered.\n\n---\n"
    )

    sections = [header]
    for issue in issues:
        sections.append(_format_single_issue(issue))

    return "\n".join(sections)


def _format_single_issue(issue: FilteredIssue) -> str:
    """Format a single filtered issue as markdown."""
    trust_note = "trusted" if issue.author_is_trusted else "UNTRUSTED — body hidden"
    parts = [
        f"\n## Issue #{issue.number}: {issue.title}",
        f"**Author**: {issue.author} ({trust_note})",
        f"**Labels**: {', '.join(issue.labels)}",
    ]

    if issue.body:
        parts.append(f"\n### Body\n\n{issue.body}")
    elif not issue.author_is_trusted:
        parts.append("\n### Body\n\n*[Content hidden: author is not a CODEOWNERS maintainer]*")

    if issue.trusted_comments:
        parts.append("\n### Comments (trusted authors only)\n")
        for comment in issue.trusted_comments:
            parts.append(f"**{comment.author}** ({comment.created_at}):\n{comment.body}\n")
    else:
        parts.append("\n### Comments\n\n*No trusted comments.*")

    if issue.merge_conflict:
        parts.append(
            f"\n### Merge Status: CONFLICTING\n\n"
            f"This PR's branch (`{issue.merge_conflict_branch}`) has merge conflicts "
            f"with `main` and needs to be rebased."
        )

    if issue.review_comments:
        parts.append(f"\n### Review Feedback\n\n{issue.review_comments}")

    if issue.ci_failure_details and not issue.has_review_feedback:
        parts.append(f"\n### CI Status: FAILING\n\n{issue.ci_failure_details}")

    parts.append("\n---")
    return "\n".join(parts)

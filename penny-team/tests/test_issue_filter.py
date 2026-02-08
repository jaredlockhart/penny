"""Tests for issue filtering edge cases and error handling.

Happy-path flows (fetch → filter → pick → format) are covered by
test_agent_flows.py. These tests cover edge cases, error handling,
and security-critical paths that the flow tests don't exercise.
"""

from __future__ import annotations

import json

from penny_team.utils.issue_filter import (
    FilteredComment,
    FilteredIssue,
    fetch_issues_for_labels,
    format_issues_for_prompt,
    pick_actionable_issue,
)

BOT_LOGINS = {"penny-team", "penny-team[bot]"}


def _make_issue(
    number: int = 1,
    title: str = "Test issue",
    body: str = "Body text",
    author: str = "alice",
    labels: list[str] | None = None,
    trusted_comments: list | None = None,
    author_is_trusted: bool = True,
    ci_status: str | None = None,
    ci_failure_details: str | None = None,
    merge_conflict: bool = False,
    merge_conflict_branch: str | None = None,
    has_review_feedback: bool = False,
) -> FilteredIssue:
    return FilteredIssue(
        number=number,
        title=title,
        body=body,
        author=author,
        labels=labels or [],
        trusted_comments=trusted_comments or [],
        author_is_trusted=author_is_trusted,
        ci_status=ci_status,
        ci_failure_details=ci_failure_details,
        merge_conflict=merge_conflict,
        merge_conflict_branch=merge_conflict_branch,
        has_review_feedback=has_review_feedback,
    )


# --- pick_actionable_issue (edge cases only) ---


class TestPickActionableIssue:
    def test_empty_list(self):
        assert pick_actionable_issue([], BOT_LOGINS) is None

    def test_no_bot_logins_returns_first(self):
        issues = [_make_issue(number=1), _make_issue(number=2)]
        result = pick_actionable_issue(issues, bot_logins=None)
        assert result is not None
        assert result.number == 1

    def test_in_review_no_comments_skips(self):
        """in-review issues with no comments are waiting for human review."""
        issue = _make_issue(labels=["in-review"], trusted_comments=[])
        assert pick_actionable_issue([issue], BOT_LOGINS) is None

    def test_in_review_no_comments_but_failing_ci_is_actionable(self):
        """in-review with no comments but failing CI is still actionable.

        Bug fix: external signals (CI failure, merge conflict, review feedback)
        should override the "no comments → skip" logic for in-review issues.
        """
        issue = _make_issue(
            labels=["in-review"],
            trusted_comments=[],
            ci_status="failing",
            ci_failure_details="ruff check failed",
        )
        result = pick_actionable_issue([issue], BOT_LOGINS)
        assert result is not None
        assert result.number == 1

    def test_in_review_no_comments_but_merge_conflict_is_actionable(self):
        """in-review with no comments but merge conflict is still actionable."""
        issue = _make_issue(
            labels=["in-review"],
            trusted_comments=[],
            merge_conflict=True,
            merge_conflict_branch="issue-1-fix",
        )
        result = pick_actionable_issue([issue], BOT_LOGINS)
        assert result is not None

    def test_in_review_no_comments_but_review_feedback_is_actionable(self):
        """in-review with no comments but review feedback is still actionable."""
        issue = _make_issue(
            labels=["in-review"],
            trusted_comments=[],
            has_review_feedback=True,
        )
        result = pick_actionable_issue([issue], BOT_LOGINS)
        assert result is not None

    def test_in_review_bot_last_comment_but_failing_ci_is_actionable(self):
        """in-review where bot has last comment but CI is failing → still actionable.

        Bug fix: bot having the last comment should not prevent the worker
        from waking when there are external signals (failing CI).
        """
        issue = _make_issue(
            labels=["in-review"],
            trusted_comments=[
                FilteredComment(author="penny-team[bot]", body="PR created.", created_at="t1"),
            ],
            ci_status="failing",
            ci_failure_details="test failed",
        )
        result = pick_actionable_issue([issue], BOT_LOGINS)
        assert result is not None


# --- format_issues_for_prompt (edge cases only) ---


class TestFormatIssuesForPrompt:
    def test_empty_issues(self):
        result = format_issues_for_prompt([])
        assert "No matching issues found" in result

    def test_untrusted_author_hides_body(self):
        issue = _make_issue(
            title="[Title hidden: untrusted author]",
            body="",
            author="attacker",
            author_is_trusted=False,
        )
        result = format_issues_for_prompt([issue])
        assert "UNTRUSTED" in result
        assert "Content hidden" in result


# --- fetch_issues_for_labels (error handling and edge cases) ---


class TestFetchIssuesForLabels:
    def test_gh_failure_returns_empty(self, mock_subprocess):
        mock_subprocess.add_response("issue list", returncode=1, stderr="error")

        issues = fetch_issues_for_labels(["requirements"], trusted_users={"alice"})
        assert issues == []

    def test_deduplicates_across_labels(self, mock_subprocess):
        """Same issue number from multiple labels is only fetched once."""
        mock_subprocess.add_response(
            "issue list",
            stdout=json.dumps([{"number": 1}]),
        )
        mock_subprocess.add_response(
            "issue view",
            stdout=json.dumps(
                {
                    "title": "Test",
                    "body": "Body",
                    "author": {"login": "alice"},
                    "comments": [],
                    "labels": [{"name": "in-progress"}, {"name": "in-review"}],
                }
            ),
        )

        issues = fetch_issues_for_labels(["in-progress", "in-review"], trusted_users={"alice"})

        assert len(issues) == 1
        # issue list called twice (once per label), issue view called once (dedup)
        view_calls = [c for c in mock_subprocess.calls if "issue" in c[0] and "view" in c[0]]
        assert len(view_calls) == 1

    def test_no_trusted_users_includes_all(self, mock_subprocess):
        """When trusted_users is None, all content passes through."""
        mock_subprocess.add_response(
            "issue list",
            stdout=json.dumps([{"number": 1}]),
        )
        mock_subprocess.add_response(
            "issue view",
            stdout=json.dumps(
                {
                    "title": "Untrusted Issue",
                    "body": "Body from anyone",
                    "author": {"login": "stranger"},
                    "comments": [
                        {"author": {"login": "stranger"}, "body": "comment", "createdAt": "t1"},
                    ],
                    "labels": [],
                }
            ),
        )

        issues = fetch_issues_for_labels(["requirements"], trusted_users=None)
        assert len(issues) == 1
        assert issues[0].title == "Untrusted Issue"
        assert issues[0].author_is_trusted is True
        assert len(issues[0].trusted_comments) == 1

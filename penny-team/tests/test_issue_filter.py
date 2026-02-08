"""Tests for issue filtering and actionability detection."""

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
    trusted_comments: list[FilteredComment] | None = None,
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


# --- pick_actionable_issue ---


class TestPickActionableIssue:
    def test_empty_list(self):
        assert pick_actionable_issue([], BOT_LOGINS) is None

    def test_no_bot_logins_returns_first(self):
        issues = [_make_issue(number=1), _make_issue(number=2)]
        result = pick_actionable_issue(issues, bot_logins=None)
        assert result is not None
        assert result.number == 1

    def test_no_comments_returns_issue(self):
        issue = _make_issue(trusted_comments=[])
        assert pick_actionable_issue([issue], BOT_LOGINS) == issue

    def test_human_last_comment_returns_issue(self):
        issue = _make_issue(
            trusted_comments=[FilteredComment(author="alice", body="feedback", created_at="t1")]
        )
        assert pick_actionable_issue([issue], BOT_LOGINS) == issue

    def test_bot_last_comment_skips(self):
        issue = _make_issue(
            trusted_comments=[
                FilteredComment(author="alice", body="feedback", created_at="t1"),
                FilteredComment(author="penny-team[bot]", body="done", created_at="t2"),
            ]
        )
        assert pick_actionable_issue([issue], BOT_LOGINS) is None

    def test_in_review_no_comments_skips(self):
        """in-review issues with no comments are waiting for human review."""
        issue = _make_issue(labels=["in-review"], trusted_comments=[])
        assert pick_actionable_issue([issue], BOT_LOGINS) is None

    def test_ci_failing_overrides_bot_comment(self):
        issue = _make_issue(
            ci_status="failing",
            trusted_comments=[
                FilteredComment(author="penny-team[bot]", body="done", created_at="t1")
            ],
        )
        assert pick_actionable_issue([issue], BOT_LOGINS) == issue

    def test_merge_conflict_overrides_bot_comment(self):
        issue = _make_issue(
            merge_conflict=True,
            trusted_comments=[
                FilteredComment(author="penny-team[bot]", body="done", created_at="t1")
            ],
        )
        assert pick_actionable_issue([issue], BOT_LOGINS) == issue

    def test_review_feedback_overrides_bot_comment(self):
        issue = _make_issue(
            has_review_feedback=True,
            trusted_comments=[
                FilteredComment(author="penny-team[bot]", body="done", created_at="t1")
            ],
        )
        assert pick_actionable_issue([issue], BOT_LOGINS) == issue

    def test_all_handled_returns_none(self):
        issues = [
            _make_issue(
                number=1,
                trusted_comments=[
                    FilteredComment(author="penny-team[bot]", body="done", created_at="t1")
                ],
            ),
            _make_issue(
                number=2,
                trusted_comments=[
                    FilteredComment(author="penny-team", body="done", created_at="t1")
                ],
            ),
        ]
        assert pick_actionable_issue(issues, BOT_LOGINS) is None

    def test_skips_handled_returns_actionable(self):
        """First issue handled by bot, second needs attention."""
        issues = [
            _make_issue(
                number=1,
                trusted_comments=[
                    FilteredComment(author="penny-team[bot]", body="done", created_at="t1")
                ],
            ),
            _make_issue(
                number=2,
                trusted_comments=[
                    FilteredComment(author="alice", body="help", created_at="t1")
                ],
            ),
        ]
        result = pick_actionable_issue(issues, BOT_LOGINS)
        assert result is not None
        assert result.number == 2


# --- format_issues_for_prompt ---


class TestFormatIssuesForPrompt:
    def test_empty_issues(self):
        result = format_issues_for_prompt([])
        assert "No matching issues found" in result

    def test_trusted_author_includes_body(self):
        issue = _make_issue(body="Implementation details here")
        result = format_issues_for_prompt([issue])
        assert "Implementation details here" in result
        assert "trusted" in result

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

    def test_trusted_comments_included(self):
        issue = _make_issue(
            trusted_comments=[
                FilteredComment(author="alice", body="Looks good", created_at="2024-01-01")
            ]
        )
        result = format_issues_for_prompt([issue])
        assert "alice" in result
        assert "Looks good" in result

    def test_ci_failure_details_included(self):
        issue = _make_issue(
            ci_status="failing",
            ci_failure_details="**Failing checks**: lint\n\n```\nerror\n```",
        )
        result = format_issues_for_prompt([issue])
        assert "CI Status: FAILING" in result
        assert "Failing checks" in result

    def test_merge_conflict_shown(self):
        issue = _make_issue(
            merge_conflict=True,
            merge_conflict_branch="issue-1-fix",
        )
        result = format_issues_for_prompt([issue])
        assert "CONFLICTING" in result
        assert "issue-1-fix" in result

    def test_no_comments_shows_placeholder(self):
        issue = _make_issue(trusted_comments=[])
        result = format_issues_for_prompt([issue])
        assert "No trusted comments" in result


# --- fetch_issues_for_labels (subprocess mock) ---


class TestFetchIssuesForLabels:
    def test_fetches_and_filters(self, mock_subprocess):
        """Fetches issues by label, fetches detail, filters untrusted comments."""
        mock_subprocess.add_response(
            "issue list",
            stdout=json.dumps([{"number": 42}]),
        )
        mock_subprocess.add_response(
            "issue view",
            stdout=json.dumps(
                {
                    "title": "Test",
                    "body": "Body",
                    "author": {"login": "alice"},
                    "comments": [
                        {"author": {"login": "alice"}, "body": "trusted", "createdAt": "t1"},
                        {"author": {"login": "attacker"}, "body": "injected", "createdAt": "t2"},
                    ],
                    "labels": [{"name": "requirements"}],
                }
            ),
        )

        issues = fetch_issues_for_labels(["requirements"], trusted_users={"alice"})

        assert len(issues) == 1
        assert issues[0].number == 42
        assert issues[0].title == "Test"
        assert len(issues[0].trusted_comments) == 1
        assert issues[0].trusted_comments[0].author == "alice"

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

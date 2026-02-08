"""Tests for PR status detection edge cases.

Happy-path enrichment flows (CI pass/fail, merge conflict, review feedback)
are covered end-to-end by test_agent_flows.py. These tests cover edge cases,
error handling, and nuanced logic that the flow tests don't exercise.
"""

from __future__ import annotations

from penny_team.utils.issue_filter import FilteredIssue
from penny_team.utils.pr_checks import (
    _extract_failed_checks,
    _has_changes_requested,
    _match_prs_to_issues,
    enrich_issues_with_pr_status,
)


def _make_issue(number: int, labels: list[str] | None = None) -> FilteredIssue:
    return FilteredIssue(
        number=number,
        title=f"Issue #{number}",
        body="",
        author="alice",
        labels=labels or [],
    )


def _make_pr(
    number: int,
    branch: str,
    checks: list[dict] | None = None,
    mergeable: str = "MERGEABLE",
    reviews: list[dict] | None = None,
) -> dict:
    return {
        "number": number,
        "headRefName": branch,
        "statusCheckRollup": checks or [],
        "mergeable": mergeable,
        "reviews": reviews or [],
    }


# --- _match_prs_to_issues (edge cases) ---


class TestMatchPRsToIssues:
    def test_no_match_for_wrong_pattern(self):
        prs = [_make_pr(10, "feature-branch")]
        issues = [_make_issue(5)]
        assert _match_prs_to_issues(prs, issues) == {}

    def test_no_match_for_unrelated_issue_number(self):
        prs = [_make_pr(10, "issue-99-fix")]
        issues = [_make_issue(5)]
        assert _match_prs_to_issues(prs, issues) == {}

    def test_multiple_matches(self):
        prs = [
            _make_pr(10, "issue-1-feat"),
            _make_pr(11, "issue-2-fix"),
        ]
        issues = [_make_issue(1), _make_issue(2)]
        result = _match_prs_to_issues(prs, issues)
        assert len(result) == 2
        assert result[1]["number"] == 10
        assert result[2]["number"] == 11

    def test_non_numeric_issue_part_ignored(self):
        prs = [_make_pr(10, "issue-abc-fix")]
        issues = [_make_issue(1)]
        assert _match_prs_to_issues(prs, issues) == {}


# --- _has_changes_requested (edge cases) ---


class TestHasChangesRequested:
    def test_approved(self):
        reviews = [{"author": {"login": "bob"}, "state": "APPROVED"}]
        assert _has_changes_requested(reviews) is False

    def test_later_approval_overrides(self):
        """Latest review per reviewer wins — later approval overrides changes request."""
        reviews = [
            {"author": {"login": "bob"}, "state": "CHANGES_REQUESTED"},
            {"author": {"login": "bob"}, "state": "APPROVED"},
        ]
        assert _has_changes_requested(reviews) is False

    def test_different_reviewers(self):
        """One reviewer approves, another requests changes."""
        reviews = [
            {"author": {"login": "alice"}, "state": "APPROVED"},
            {"author": {"login": "bob"}, "state": "CHANGES_REQUESTED"},
        ]
        assert _has_changes_requested(reviews) is True


# --- _extract_failed_checks (edge cases) ---


class TestExtractFailedChecks:
    def test_pending_checks_skipped(self):
        checks = [
            {"state": "PENDING", "conclusion": "", "name": "deploy"},
            {"state": "IN_PROGRESS", "conclusion": "", "name": "build"},
        ]
        assert _extract_failed_checks(checks) == []

    def test_multiple_failures(self):
        checks = [
            {"state": "COMPLETED", "conclusion": "FAILURE", "name": "lint"},
            {"state": "COMPLETED", "conclusion": "FAILURE", "name": "test"},
        ]
        result = _extract_failed_checks(checks)
        assert len(result) == 2


# --- enrich_issues_with_pr_status (error handling) ---


class TestEnrichIssuesWithPRStatus:
    def test_gh_failure_is_noop(self, mock_subprocess):
        """If gh pr list fails, issues are left unchanged (fail-open)."""
        issue = _make_issue(1, labels=["in-review"])
        mock_subprocess.add_response("pr list", returncode=1, stderr="auth error")

        enrich_issues_with_pr_status([issue])
        assert issue.ci_status is None
        assert issue.merge_conflict is False

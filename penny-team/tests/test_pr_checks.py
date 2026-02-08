"""Tests for PR status detection (CI checks, merge conflicts, reviews)."""

from __future__ import annotations

import json

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


# --- _match_prs_to_issues ---


class TestMatchPRsToIssues:
    def test_matches_by_branch_convention(self):
        prs = [_make_pr(10, "issue-5-add-feature")]
        issues = [_make_issue(5)]
        result = _match_prs_to_issues(prs, issues)
        assert result == {5: prs[0]}

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


# --- _has_changes_requested ---


class TestHasChangesRequested:
    def test_no_reviews(self):
        assert _has_changes_requested([]) is False

    def test_approved(self):
        reviews = [{"author": {"login": "bob"}, "state": "APPROVED"}]
        assert _has_changes_requested(reviews) is False

    def test_changes_requested(self):
        reviews = [{"author": {"login": "bob"}, "state": "CHANGES_REQUESTED"}]
        assert _has_changes_requested(reviews) is True

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


# --- _extract_failed_checks ---


class TestExtractFailedChecks:
    def test_all_passing(self):
        checks = [
            {"state": "COMPLETED", "conclusion": "SUCCESS", "name": "lint"},
            {"state": "COMPLETED", "conclusion": "NEUTRAL", "name": "optional"},
        ]
        assert _extract_failed_checks(checks) == []

    def test_one_failure(self):
        checks = [
            {"state": "COMPLETED", "conclusion": "SUCCESS", "name": "lint"},
            {"state": "COMPLETED", "conclusion": "FAILURE", "name": "test"},
        ]
        result = _extract_failed_checks(checks)
        assert len(result) == 1
        assert result[0].name == "test"
        assert result[0].conclusion == "FAILURE"

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

    def test_uses_context_field_as_name(self):
        """Falls back to 'context' field when 'name' is absent."""
        checks = [{"state": "COMPLETED", "conclusion": "FAILURE", "context": "ci/check"}]
        result = _extract_failed_checks(checks)
        assert result[0].name == "ci/check"


# --- enrich_issues_with_pr_status ---


class TestEnrichIssuesWithPRStatus:
    def test_no_in_review_issues_is_noop(self, mock_subprocess):
        """Issues without in-review label are not enriched."""
        issue = _make_issue(1, labels=["in-progress"])
        enrich_issues_with_pr_status([issue])
        assert issue.ci_status is None

    def test_ci_passing(self, mock_subprocess):
        issue = _make_issue(1, labels=["in-review"])
        mock_subprocess.add_response(
            "pr list",
            stdout=json.dumps(
                [
                    _make_pr(
                        10,
                        "issue-1-fix",
                        checks=[{"state": "COMPLETED", "conclusion": "SUCCESS", "name": "lint"}],
                    )
                ]
            ),
        )

        enrich_issues_with_pr_status([issue])
        assert issue.ci_status == "passing"

    def test_ci_failing_with_log(self, mock_subprocess):
        issue = _make_issue(1, labels=["in-review"])
        mock_subprocess.add_response(
            "pr list",
            stdout=json.dumps(
                [
                    _make_pr(
                        10,
                        "issue-1-fix",
                        checks=[
                            {"state": "COMPLETED", "conclusion": "FAILURE", "name": "test"}
                        ],
                    )
                ]
            ),
        )
        mock_subprocess.add_response(
            "run list",
            stdout=json.dumps([{"databaseId": 999}]),
        )
        mock_subprocess.add_response(
            "run view",
            stdout="Error: test failed\nassert False",
        )

        enrich_issues_with_pr_status([issue])
        assert issue.ci_status == "failing"
        assert "Failing checks" in issue.ci_failure_details
        assert "test" in issue.ci_failure_details

    def test_merge_conflict(self, mock_subprocess):
        issue = _make_issue(1, labels=["in-review"])
        mock_subprocess.add_response(
            "pr list",
            stdout=json.dumps([_make_pr(10, "issue-1-fix", mergeable="CONFLICTING")]),
        )

        enrich_issues_with_pr_status([issue])
        assert issue.merge_conflict is True
        assert issue.merge_conflict_branch == "issue-1-fix"

    def test_review_feedback(self, mock_subprocess):
        issue = _make_issue(1, labels=["in-review"])
        mock_subprocess.add_response(
            "pr list",
            stdout=json.dumps(
                [
                    _make_pr(
                        10,
                        "issue-1-fix",
                        reviews=[{"author": {"login": "bob"}, "state": "CHANGES_REQUESTED"}],
                    )
                ]
            ),
        )

        enrich_issues_with_pr_status([issue])
        assert issue.has_review_feedback is True

    def test_gh_failure_is_noop(self, mock_subprocess):
        """If gh pr list fails, issues are left unchanged (fail-open)."""
        issue = _make_issue(1, labels=["in-review"])
        mock_subprocess.add_response("pr list", returncode=1, stderr="auth error")

        enrich_issues_with_pr_status([issue])
        assert issue.ci_status is None
        assert issue.merge_conflict is False

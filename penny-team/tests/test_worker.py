"""Integration tests for the Worker agent flow, plus PR status edge cases.

The Worker implements features from specs (in-progress), handles
PR feedback including failing CI, merge conflicts, and review comments
(in-review), and fixes bugs with a specialized test-driven workflow (bug).
It's the only agent that transitions labels itself:
in-progress → in-review and bug → in-review after creating a PR.

Flow: specification (Architect) → in-progress (Worker) → in-review (Worker) → closed
      bug → in-review (Worker) → closed
"""

from __future__ import annotations

import json

from penny_team.utils.issue_filter import FilteredIssue
from penny_team.utils.pr_checks import (
    _extract_failed_checks,
    _has_changes_requested,
    _match_prs_to_issues,
    enrich_issues_with_pr_status,
)
from tests.conftest import (
    BOT_LOGIN,
    BOT_LOGINS,
    extract_prompt,
    issue_list_response,
    issue_view_response,
    make_agent,
    make_pr_response,
    result_event,
)


# =============================================================================
# Worker agent flows — integration tests through agent.run()
# =============================================================================


class TestWorkerFlow:
    """Worker agent processes issues labeled 'in-progress', 'in-review', or 'bug'."""

    def test_new_in_progress_issue_implements(
        self, tmp_path, mock_subprocess, capture_popen, monkeypatch
    ):
        """Fresh in-progress issue with spec → Claude implements feature.

        Flow: issue has spec from Architect, no PR yet
        → pick_actionable_issue returns it (no bot comment)
        → prompt assembled with Worker CLAUDE.md + issue + spec.
        """
        agent = make_agent(tmp_path, name="worker", required_labels=["in-progress", "in-review", "bug"])
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        mock_subprocess.add_response("issue list", stdout=issue_list_response(42))
        mock_subprocess.add_response(
            "issue view",
            stdout=issue_view_response(
                number=42,
                labels=["in-progress"],
                comments=[
                    {
                        "author": {"login": BOT_LOGIN},
                        "body": "## Detailed Specification\n\n**Technical Approach**: New ReminderAgent",
                        "createdAt": "2024-01-06T00:00:00Z",
                    },
                    {
                        "author": {"login": "alice"},
                        "body": "Go ahead and implement this!",
                        "createdAt": "2024-01-07T00:00:00Z",
                    },
                ],
            ),
        )
        mock_subprocess.add_response("pr list", stdout="[]")

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        prompt = extract_prompt(calls)
        assert "Worker Agent Prompt" in prompt
        assert "Add reminders feature" in prompt
        assert "Detailed Specification" in prompt

    def test_in_review_failing_ci_triggers_fix(
        self, tmp_path, mock_subprocess, capture_popen, monkeypatch
    ):
        """In-review issue with failing CI → Claude fixes CI.

        Flow: issue is in-review, PR has failing checks
        → enrich_issues_with_pr_status sets ci_status="failing"
        → pick_actionable_issue returns it (CI failure is always actionable)
        → prompt includes CI failure details so Claude can fix them.
        """
        agent = make_agent(tmp_path, name="worker", required_labels=["in-progress", "in-review", "bug"])
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        mock_subprocess.add_response("issue list", stdout=issue_list_response(42))
        mock_subprocess.add_response(
            "issue view",
            stdout=issue_view_response(
                number=42,
                labels=["in-review"],
                comments=[
                    {
                        "author": {"login": BOT_LOGIN},
                        "body": "Created PR #10 for this issue.",
                        "createdAt": "2024-01-08T00:00:00Z",
                    },
                ],
            ),
        )
        mock_subprocess.add_response(
            "pr list",
            stdout=json.dumps([
                make_pr_response(
                    10,
                    "issue-42-add-reminders",
                    checks=[
                        {
                            "name": "check",
                            "state": "COMPLETED",
                            "conclusion": "FAILURE",
                            "context": "CI / check",
                        }
                    ],
                )
            ]),
        )
        mock_subprocess.add_response(
            "run list",
            stdout=json.dumps([{"databaseId": 99}]),
        )
        mock_subprocess.add_response(
            "run view",
            stdout="error: ruff check failed\n  penny_team/base.py:10: E501 line too long",
        )

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        prompt = extract_prompt(calls)
        assert "CI Status: FAILING" in prompt
        assert "ruff check failed" in prompt

    def test_in_review_merge_conflict_triggers_rebase(
        self, tmp_path, mock_subprocess, capture_popen, monkeypatch
    ):
        """In-review issue with merge conflict → Claude rebases.

        Flow: issue is in-review, PR has CONFLICTING mergeable status
        → enrich_issues_with_pr_status sets merge_conflict=True
        → pick_actionable_issue returns it (merge conflict always actionable)
        → prompt includes "Merge Status: CONFLICTING" with branch name.
        """
        agent = make_agent(tmp_path, name="worker", required_labels=["in-progress", "in-review", "bug"])
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        mock_subprocess.add_response("issue list", stdout=issue_list_response(42))
        mock_subprocess.add_response(
            "issue view",
            stdout=issue_view_response(
                number=42,
                labels=["in-review"],
                comments=[
                    {
                        "author": {"login": BOT_LOGIN},
                        "body": "Created PR for this issue.",
                        "createdAt": "2024-01-08T00:00:00Z",
                    },
                ],
            ),
        )
        mock_subprocess.add_response(
            "pr list",
            stdout=json.dumps([
                make_pr_response(10, "issue-42-add-reminders", mergeable="CONFLICTING")
            ]),
        )

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        prompt = extract_prompt(calls)
        assert "Merge Status: CONFLICTING" in prompt
        assert "issue-42-add-reminders" in prompt

    def test_in_review_review_feedback_triggers_response(
        self, tmp_path, mock_subprocess, capture_popen, monkeypatch
    ):
        """In-review issue with CHANGES_REQUESTED review → Claude addresses feedback.

        Flow: issue is in-review, PR reviewer requested changes
        → enrich_issues_with_pr_status sets has_review_feedback=True
        → pick_actionable_issue returns it (review feedback always actionable)
        → prompt includes the issue so Claude can read PR and address feedback.
        """
        agent = make_agent(tmp_path, name="worker", required_labels=["in-progress", "in-review", "bug"])
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        mock_subprocess.add_response("issue list", stdout=issue_list_response(42))
        mock_subprocess.add_response(
            "issue view",
            stdout=issue_view_response(
                number=42,
                labels=["in-review"],
                comments=[
                    {
                        "author": {"login": BOT_LOGIN},
                        "body": "Created PR for this issue.",
                        "createdAt": "2024-01-08T00:00:00Z",
                    },
                ],
            ),
        )
        mock_subprocess.add_response(
            "pr list",
            stdout=json.dumps([
                make_pr_response(
                    10,
                    "issue-42-add-reminders",
                    reviews=[
                        {
                            "author": {"login": "alice"},
                            "state": "CHANGES_REQUESTED",
                        }
                    ],
                )
            ]),
        )

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        prompt = extract_prompt(calls)
        assert "Issue #42" in prompt

    def test_in_review_all_passing_no_feedback_skips(
        self, tmp_path, mock_subprocess, capture_popen, monkeypatch
    ):
        """In-review, CI passing, no conflicts, no feedback, bot last → skip.

        Flow: everything looks good, bot has the last comment, no external
        signals → pick_actionable_issue returns None → agent skips.
        This represents a PR waiting for human review.
        """
        agent = make_agent(tmp_path, name="worker", required_labels=["in-progress", "in-review", "bug"])
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))
        monkeypatch.setattr(
            type(agent), "_state_path", property(lambda self: tmp_path / "worker.state.json")
        )

        mock_subprocess.add_response(
            "issue list",
            stdout=json.dumps([{"number": 42, "updatedAt": "2024-01-08T00:00:00Z"}]),
        )
        mock_subprocess.add_response(
            "issue view",
            stdout=issue_view_response(
                number=42,
                labels=["in-review"],
                comments=[
                    {
                        "author": {"login": BOT_LOGIN},
                        "body": "Created PR #10. Ready for review.",
                        "createdAt": "2024-01-08T00:00:00Z",
                    },
                ],
            ),
        )
        mock_subprocess.add_response(
            "pr list",
            stdout=json.dumps([
                make_pr_response(
                    10,
                    "issue-42-add-reminders",
                    checks=[
                        {
                            "name": "check",
                            "state": "COMPLETED",
                            "conclusion": "SUCCESS",
                            "context": "CI / check",
                        }
                    ],
                )
            ]),
        )

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        assert result.output == "No actionable issues"
        assert len(calls) == 0

    def test_worker_prioritizes_in_review_over_in_progress(
        self, tmp_path, mock_subprocess, capture_popen, monkeypatch
    ):
        """Both in-review (CI failing) and in-progress issues exist → picks in-review.

        Flow: issue #42 is in-review with failing CI, issue #43 is in-progress
        → pick_actionable_issue checks CI-failing first (always actionable)
        → returns #42 → Worker fixes CI, not implementing #43.
        """
        agent = make_agent(tmp_path, name="worker", required_labels=["in-progress", "in-review", "bug"])
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        mock_subprocess.add_response(
            "issue list",
            stdout=json.dumps([{"number": 42}, {"number": 43}]),
        )
        mock_subprocess.add_response(
            "issue view 42",
            stdout=issue_view_response(
                number=42,
                title="Add reminders feature",
                labels=["in-review"],
                comments=[
                    {
                        "author": {"login": BOT_LOGIN},
                        "body": "Created PR for this issue.",
                        "createdAt": "2024-01-08T00:00:00Z",
                    },
                ],
            ),
        )
        mock_subprocess.add_response(
            "issue view 43",
            stdout=issue_view_response(
                number=43,
                title="Add user profiles",
                labels=["in-progress"],
                comments=[],
            ),
        )
        mock_subprocess.add_response(
            "pr list",
            stdout=json.dumps([
                make_pr_response(
                    10,
                    "issue-42-add-reminders",
                    checks=[
                        {
                            "name": "check",
                            "state": "COMPLETED",
                            "conclusion": "FAILURE",
                            "context": "CI / check",
                        }
                    ],
                )
            ]),
        )
        mock_subprocess.add_response(
            "run list",
            stdout=json.dumps([{"databaseId": 99}]),
        )
        mock_subprocess.add_response(
            "run view",
            stdout="error: test_reminders failed",
        )

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        prompt = extract_prompt(calls)
        assert "CI Status: FAILING" in prompt
        assert "Add reminders feature" in prompt


# =============================================================================
# Worker in-review actionability — integration tests through agent.run()
#
# These test edge cases where external signals (CI failure, merge conflict,
# review feedback) should override normal skip logic for in-review issues.
# =============================================================================


class TestWorkerInReviewActionability:
    def test_in_review_no_comments_skips(
        self, tmp_path, mock_subprocess, capture_popen, monkeypatch
    ):
        """in-review issue with no comments → waiting for human review → skip."""
        agent = make_agent(tmp_path, name="worker", required_labels=["in-progress", "in-review", "bug"])
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))
        monkeypatch.setattr(
            type(agent), "_state_path", property(lambda self: tmp_path / "w.state.json")
        )

        mock_subprocess.add_response(
            "issue list",
            stdout=json.dumps([{"number": 42, "updatedAt": "2024-01-01T00:00:00Z"}]),
        )
        mock_subprocess.add_response(
            "issue view",
            stdout=issue_view_response(number=42, labels=["in-review"], comments=[]),
        )
        mock_subprocess.add_response("pr list", stdout="[]")

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        assert result.output == "No actionable issues"
        assert len(calls) == 0

    def test_in_review_no_comments_but_failing_ci_is_actionable(
        self, tmp_path, mock_subprocess, capture_popen, monkeypatch
    ):
        """in-review with no comments but failing CI → still actionable.

        Bug fix: external signals (CI failure, merge conflict, review feedback)
        should override the "no comments → skip" logic for in-review issues.
        """
        agent = make_agent(tmp_path, name="worker", required_labels=["in-progress", "in-review", "bug"])
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        mock_subprocess.add_response("issue list", stdout=issue_list_response(42))
        mock_subprocess.add_response(
            "issue view",
            stdout=issue_view_response(number=42, labels=["in-review"], comments=[]),
        )
        mock_subprocess.add_response(
            "pr list",
            stdout=json.dumps([
                make_pr_response(
                    10,
                    "issue-42-fix",
                    checks=[
                        {
                            "name": "check",
                            "state": "COMPLETED",
                            "conclusion": "FAILURE",
                            "context": "CI / check",
                        }
                    ],
                )
            ]),
        )
        mock_subprocess.add_response("run list", stdout=json.dumps([{"databaseId": 99}]))
        mock_subprocess.add_response("run view", stdout="ruff check failed")

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        assert len(calls) == 1
        prompt = extract_prompt(calls)
        assert "CI Status: FAILING" in prompt

    def test_in_review_no_comments_but_merge_conflict_is_actionable(
        self, tmp_path, mock_subprocess, capture_popen, monkeypatch
    ):
        """in-review with no comments but merge conflict → still actionable."""
        agent = make_agent(tmp_path, name="worker", required_labels=["in-progress", "in-review", "bug"])
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        mock_subprocess.add_response("issue list", stdout=issue_list_response(42))
        mock_subprocess.add_response(
            "issue view",
            stdout=issue_view_response(number=42, labels=["in-review"], comments=[]),
        )
        mock_subprocess.add_response(
            "pr list",
            stdout=json.dumps([
                make_pr_response(10, "issue-42-fix", mergeable="CONFLICTING")
            ]),
        )

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        assert len(calls) == 1
        prompt = extract_prompt(calls)
        assert "Merge Status: CONFLICTING" in prompt

    def test_in_review_no_comments_but_review_feedback_is_actionable(
        self, tmp_path, mock_subprocess, capture_popen, monkeypatch
    ):
        """in-review with no comments but review feedback → still actionable."""
        agent = make_agent(tmp_path, name="worker", required_labels=["in-progress", "in-review", "bug"])
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        mock_subprocess.add_response("issue list", stdout=issue_list_response(42))
        mock_subprocess.add_response(
            "issue view",
            stdout=issue_view_response(number=42, labels=["in-review"], comments=[]),
        )
        mock_subprocess.add_response(
            "pr list",
            stdout=json.dumps([
                make_pr_response(
                    10,
                    "issue-42-fix",
                    reviews=[{"author": {"login": "alice"}, "state": "CHANGES_REQUESTED"}],
                )
            ]),
        )

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        assert len(calls) == 1

    def test_in_review_bot_last_comment_but_failing_ci_is_actionable(
        self, tmp_path, mock_subprocess, capture_popen, monkeypatch
    ):
        """in-review where bot has last comment but CI is failing → still actionable.

        Bug fix: bot having the last comment should not prevent the worker
        from waking when there are external signals (failing CI).
        """
        agent = make_agent(tmp_path, name="worker", required_labels=["in-progress", "in-review", "bug"])
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        mock_subprocess.add_response("issue list", stdout=issue_list_response(42))
        mock_subprocess.add_response(
            "issue view",
            stdout=issue_view_response(
                number=42,
                labels=["in-review"],
                comments=[
                    {
                        "author": {"login": BOT_LOGIN},
                        "body": "PR created.",
                        "createdAt": "2024-01-08T00:00:00Z",
                    },
                ],
            ),
        )
        mock_subprocess.add_response(
            "pr list",
            stdout=json.dumps([
                make_pr_response(
                    10,
                    "issue-42-fix",
                    checks=[
                        {
                            "name": "check",
                            "state": "COMPLETED",
                            "conclusion": "FAILURE",
                            "context": "CI / check",
                        }
                    ],
                )
            ]),
        )
        mock_subprocess.add_response("run list", stdout=json.dumps([{"databaseId": 99}]))
        mock_subprocess.add_response("run view", stdout="test failed")

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        assert len(calls) == 1
        prompt = extract_prompt(calls)
        assert "CI Status: FAILING" in prompt


# =============================================================================
# Worker deduplication across labels — integration test through agent.run()
# =============================================================================


class TestWorkerDeduplicatesAcrossLabels:
    def test_same_issue_from_multiple_labels_fetched_once(
        self, tmp_path, mock_subprocess, capture_popen, monkeypatch
    ):
        """Same issue number from both in-progress and in-review → fetched once."""
        agent = make_agent(tmp_path, name="worker", required_labels=["in-progress", "in-review", "bug"])
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        # Both label searches return issue #42
        mock_subprocess.add_response(
            "issue list",
            stdout=json.dumps([{"number": 42}]),
        )
        mock_subprocess.add_response(
            "issue view",
            stdout=issue_view_response(
                number=42,
                labels=["in-progress", "in-review"],
                comments=[
                    {
                        "author": {"login": "alice"},
                        "body": "Ready to implement",
                        "createdAt": "2024-01-01T00:00:00Z",
                    },
                ],
            ),
        )
        mock_subprocess.add_response("pr list", stdout="[]")

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        agent.run()

        # issue view should only be called once (dedup)
        view_calls = [
            c for c in mock_subprocess.calls
            if "issue" in c[0] and "view" in c[0]
        ]
        assert len(view_calls) == 1


# =============================================================================
# PR matching — unit tests for pure function edge cases
# =============================================================================


def _make_issue(number: int, labels: list[str] | None = None) -> FilteredIssue:
    return FilteredIssue(
        number=number,
        title=f"Issue #{number}",
        body="",
        author="alice",
        labels=labels or [],
    )


class TestWorkerPRMatching:
    def test_no_match_for_wrong_branch_pattern(self):
        prs = [make_pr_response(10, "feature-branch")]
        issues = [_make_issue(5)]
        assert _match_prs_to_issues(prs, issues) == {}

    def test_no_match_for_unrelated_issue_number(self):
        prs = [make_pr_response(10, "issue-99-fix")]
        issues = [_make_issue(5)]
        assert _match_prs_to_issues(prs, issues) == {}

    def test_multiple_prs_multiple_issues(self):
        prs = [
            make_pr_response(10, "issue-1-feat"),
            make_pr_response(11, "issue-2-fix"),
        ]
        issues = [_make_issue(1), _make_issue(2)]
        result = _match_prs_to_issues(prs, issues)
        assert len(result) == 2
        assert result[1]["number"] == 10
        assert result[2]["number"] == 11

    def test_non_numeric_issue_part_ignored(self):
        prs = [make_pr_response(10, "issue-abc-fix")]
        issues = [_make_issue(1)]
        assert _match_prs_to_issues(prs, issues) == {}


# =============================================================================
# Review handling — unit tests for pure function edge cases
# =============================================================================


class TestWorkerReviewHandling:
    def test_approved_review_no_action(self):
        reviews = [{"author": {"login": "bob"}, "state": "APPROVED"}]
        assert _has_changes_requested(reviews) is False

    def test_later_approval_overrides_changes_requested(self):
        """Latest review per reviewer wins — later approval overrides changes request."""
        reviews = [
            {"author": {"login": "bob"}, "state": "CHANGES_REQUESTED"},
            {"author": {"login": "bob"}, "state": "APPROVED"},
        ]
        assert _has_changes_requested(reviews) is False

    def test_different_reviewers_mixed_outcome(self):
        """One reviewer approves, another requests changes."""
        reviews = [
            {"author": {"login": "alice"}, "state": "APPROVED"},
            {"author": {"login": "bob"}, "state": "CHANGES_REQUESTED"},
        ]
        assert _has_changes_requested(reviews) is True


# =============================================================================
# CI check extraction — unit tests for pure function edge cases
# =============================================================================


class TestWorkerCIChecks:
    def test_pending_checks_not_treated_as_failure(self):
        checks = [
            {"state": "PENDING", "conclusion": "", "name": "deploy"},
            {"state": "IN_PROGRESS", "conclusion": "", "name": "build"},
        ]
        assert _extract_failed_checks(checks) == []

    def test_multiple_ci_failures(self):
        checks = [
            {"state": "COMPLETED", "conclusion": "FAILURE", "name": "lint"},
            {"state": "COMPLETED", "conclusion": "FAILURE", "name": "test"},
        ]
        result = _extract_failed_checks(checks)
        assert len(result) == 2


# =============================================================================
# PR enrichment error handling — unit test
# =============================================================================


class TestWorkerPREnrichFailure:
    def test_gh_pr_list_failure_is_noop(self, mock_subprocess):
        """If gh pr list fails, issues are left unchanged (fail-open)."""
        issue = _make_issue(1, labels=["in-review"])
        mock_subprocess.add_response("pr list", returncode=1, stderr="auth error")

        enrich_issues_with_pr_status([issue])
        assert issue.ci_status is None
        assert issue.merge_conflict is False


# =============================================================================
# Bug fix flow — integration tests through agent.run()
#
# Bug issues bypass PM and Architect entirely. Worker picks them up directly
# from the 'bug' label and follows a test-driven bug fix workflow.
# =============================================================================


class TestWorkerBugFixFlow:
    """Worker handles bug issues with a specialized workflow.

    Bugs bypass the PM/Architect pipeline — Worker picks them up directly.
    Bug fixes are prioritized over feature work (in-progress).
    """

    def test_bug_issue_triggers_worker(
        self, tmp_path, mock_subprocess, capture_popen, monkeypatch
    ):
        """Bug issue with no comments → Claude runs bug fix workflow.

        Flow: issue labeled 'bug' with no comments → bypasses PM/Architect
        → pick_actionable_issue returns it → prompt assembled with Worker
        CLAUDE.md + issue data. Worker follows bug fix workflow.
        """
        agent = make_agent(
            tmp_path, name="worker", required_labels=["in-progress", "in-review", "bug"]
        )
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        # Issue list returns #50 for all labels (dedup handles duplicates)
        mock_subprocess.add_response("issue list", stdout=issue_list_response(50))
        mock_subprocess.add_response(
            "issue view",
            stdout=issue_view_response(
                number=50,
                title="Search fails when query is empty",
                body="When a user sends an empty search query, the app crashes.",
                labels=["bug"],
                comments=[],
            ),
        )
        # PR enrichment: no open PRs
        mock_subprocess.add_response("pr list", stdout="[]")

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        prompt = extract_prompt(calls)
        assert "Worker Agent Prompt" in prompt
        assert "Search fails when query is empty" in prompt
        assert "Issue #50" in prompt

    def test_bug_prioritized_over_in_progress(
        self, tmp_path, mock_subprocess, capture_popen, monkeypatch
    ):
        """Both bug and in-progress issues exist → picks bug first.

        Flow: issue #50 is labeled 'bug', issue #43 is labeled 'in-progress'
        → pick_actionable_issue sorts bugs before non-bugs
        → returns #50 → Worker fixes bug, not implementing #43.

        This verifies that bug fixes are prioritized over new feature work.
        """
        agent = make_agent(
            tmp_path, name="worker", required_labels=["in-progress", "in-review", "bug"]
        )
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        # Issue list returns both issues (first match wins; dedup prevents re-fetch)
        mock_subprocess.add_response(
            "issue list",
            stdout=json.dumps([{"number": 43}, {"number": 50}]),
        )
        # Issue views — matched by "issue view 43" vs "issue view 50"
        mock_subprocess.add_response(
            "issue view 43",
            stdout=issue_view_response(
                number=43,
                title="Add user profiles",
                labels=["in-progress"],
                comments=[
                    {
                        "author": {"login": "alice"},
                        "body": "Go ahead and implement this.",
                        "createdAt": "2024-01-07T00:00:00Z",
                    },
                ],
            ),
        )
        mock_subprocess.add_response(
            "issue view 50",
            stdout=issue_view_response(
                number=50,
                title="Search fails when query is empty",
                body="When a user sends an empty search query, the app crashes.",
                labels=["bug"],
                comments=[],
            ),
        )
        # PR enrichment: no open PRs
        mock_subprocess.add_response("pr list", stdout="[]")

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        prompt = extract_prompt(calls)
        # Should be working on #50 (bug), not #43 (feature)
        assert "Search fails when query is empty" in prompt
        assert "Issue #50" in prompt

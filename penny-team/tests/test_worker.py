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

from penny_team.utils.github_api import CheckStatus, CommentAuthor, PRReview, PullRequest, WorkflowRun
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
    make_agent,
    make_check_status,
    make_issue_detail,
    make_issue_list_items,
    make_pull_request,
    result_event,
)


# =============================================================================
# Worker agent flows — integration tests through agent.run()
# =============================================================================


class TestWorkerFlow:
    """Worker agent processes issues labeled 'in-progress', 'in-review', or 'bug'."""

    def test_new_in_progress_issue_implements(
        self, tmp_path, mock_github_api, capture_popen, monkeypatch
    ):
        """Fresh in-progress issue with spec → Claude implements feature.

        Flow: issue has spec from Architect, no PR yet
        → pick_actionable_issue returns it (no bot comment)
        → prompt assembled with Worker CLAUDE.md + issue + spec.
        """
        agent = make_agent(tmp_path, name="worker", required_labels=["in-progress", "in-review", "bug"], github_api=mock_github_api)
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        mock_github_api.set_issues("in-progress", make_issue_list_items((42, "2024-01-01T00:00:00Z")))
        mock_github_api.set_issues("in-review", [])
        mock_github_api.set_issues("bug", [])
        mock_github_api.set_issues_detailed("in-progress", [
            make_issue_detail(
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
        ])
        mock_github_api.set_issues_detailed("in-review", [])
        mock_github_api.set_issues_detailed("bug", [])

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        prompt = extract_prompt(calls)
        assert "Worker Agent Prompt" in prompt
        assert "Add reminders feature" in prompt
        assert "Detailed Specification" in prompt

    def test_in_review_failing_ci_triggers_fix(
        self, tmp_path, mock_github_api, capture_popen, monkeypatch
    ):
        """In-review issue with failing CI → Claude fixes CI.

        Flow: issue is in-review, PR has failing checks
        → enrich_issues_with_pr_status sets ci_status="failing"
        → pick_actionable_issue returns it (CI failure is always actionable)
        → prompt includes CI failure details so Claude can fix them.
        """
        agent = make_agent(tmp_path, name="worker", required_labels=["in-progress", "in-review", "bug"], github_api=mock_github_api)
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        mock_github_api.set_issues("in-progress", [])
        mock_github_api.set_issues("in-review", make_issue_list_items((42, "2024-01-01T00:00:00Z")))
        mock_github_api.set_issues("bug", [])
        mock_github_api.set_issues_detailed("in-progress", [])
        mock_github_api.set_issues_detailed("in-review", [
            make_issue_detail(
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
        ])
        mock_github_api.set_issues_detailed("bug", [])
        mock_github_api.set_prs([
            make_pull_request(
                10,
                "issue-42-add-reminders",
                checks=[make_check_status("check", "COMPLETED", "FAILURE")],
            ),
        ])
        mock_github_api.set_failed_runs("issue-42-add-reminders", [WorkflowRun(id=99)])
        mock_github_api.set_failed_log(99, "error: ruff check failed\n  penny_team/base.py:10: E501 line too long")

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        prompt = extract_prompt(calls)
        assert "CI Status: FAILING" in prompt
        assert "ruff check failed" in prompt

    def test_in_review_merge_conflict_triggers_rebase(
        self, tmp_path, mock_github_api, capture_popen, monkeypatch
    ):
        """In-review issue with merge conflict → Claude rebases.

        Flow: issue is in-review, PR has CONFLICTING mergeable status
        → enrich_issues_with_pr_status sets merge_conflict=True
        → pick_actionable_issue returns it (merge conflict always actionable)
        → prompt includes "Merge Status: CONFLICTING" with branch name.
        """
        agent = make_agent(tmp_path, name="worker", required_labels=["in-progress", "in-review", "bug"], github_api=mock_github_api)
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        mock_github_api.set_issues("in-progress", [])
        mock_github_api.set_issues("in-review", make_issue_list_items((42, "2024-01-01T00:00:00Z")))
        mock_github_api.set_issues("bug", [])
        mock_github_api.set_issues_detailed("in-progress", [])
        mock_github_api.set_issues_detailed("in-review", [
            make_issue_detail(
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
        ])
        mock_github_api.set_issues_detailed("bug", [])
        mock_github_api.set_prs([
            make_pull_request(10, "issue-42-add-reminders", mergeable="CONFLICTING"),
        ])

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        prompt = extract_prompt(calls)
        assert "Merge Status: CONFLICTING" in prompt
        assert "issue-42-add-reminders" in prompt

    def test_in_review_review_feedback_triggers_response(
        self, tmp_path, mock_github_api, capture_popen, monkeypatch
    ):
        """In-review issue with CHANGES_REQUESTED review → Claude addresses feedback.

        Flow: issue is in-review, PR reviewer requested changes
        → enrich_issues_with_pr_status sets has_review_feedback=True
        → pick_actionable_issue returns it (review feedback always actionable)
        → prompt includes the issue so Claude can read PR and address feedback.
        """
        agent = make_agent(tmp_path, name="worker", required_labels=["in-progress", "in-review", "bug"], github_api=mock_github_api)
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        mock_github_api.set_issues("in-progress", [])
        mock_github_api.set_issues("in-review", make_issue_list_items((42, "2024-01-01T00:00:00Z")))
        mock_github_api.set_issues("bug", [])
        mock_github_api.set_issues_detailed("in-progress", [])
        mock_github_api.set_issues_detailed("in-review", [
            make_issue_detail(
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
        ])
        mock_github_api.set_issues_detailed("bug", [])
        mock_github_api.set_prs([
            make_pull_request(
                10,
                "issue-42-add-reminders",
                reviews=[PRReview(
                    author=CommentAuthor(login="alice"),
                    state="CHANGES_REQUESTED",
                    submitted_at="",
                )],
            ),
        ])

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        prompt = extract_prompt(calls)
        assert "Issue #42" in prompt

    def test_in_review_all_passing_no_feedback_skips(
        self, tmp_path, mock_github_api, capture_popen, monkeypatch
    ):
        """In-review, CI passing, no conflicts, no feedback, already processed → skip.

        Flow: everything looks good, agent already processed this issue,
        no new human comments, no external signals
        → pick_actionable_issue returns None → agent skips.
        This represents a PR waiting for human review.
        """
        agent = make_agent(tmp_path, name="worker", required_labels=["in-progress", "in-review", "bug"], github_api=mock_github_api)
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))
        state_path = tmp_path / "worker.state.json"
        monkeypatch.setattr(
            type(agent), "_state_path", property(lambda self: state_path)
        )

        # Pre-populate: worker already processed issue 42
        state_path.write_text(json.dumps({
            "timestamps": {},
            "processed": {"42": "2024-01-09T00:00:00Z"},
        }))

        mock_github_api.set_issues("in-progress", [])
        mock_github_api.set_issues("in-review", make_issue_list_items((42, "2024-01-08T00:00:00Z")))
        mock_github_api.set_issues("bug", [])
        mock_github_api.set_issues_detailed("in-progress", [])
        mock_github_api.set_issues_detailed("in-review", [
            make_issue_detail(
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
        ])
        mock_github_api.set_issues_detailed("bug", [])
        mock_github_api.set_prs([
            make_pull_request(
                10,
                "issue-42-add-reminders",
                checks=[make_check_status("check", "COMPLETED", "SUCCESS")],
            ),
        ])

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        assert result.output == "No actionable issues"
        assert len(calls) == 0

    def test_worker_prioritizes_in_review_over_in_progress(
        self, tmp_path, mock_github_api, capture_popen, monkeypatch
    ):
        """Both in-review (CI failing) and in-progress issues exist → picks in-review.

        Flow: issue #42 is in-review with failing CI, issue #43 is in-progress
        but already processed. CI-failing is always actionable, so #42 is picked
        over the already-handled #43.
        """
        agent = make_agent(tmp_path, name="worker", required_labels=["in-progress", "in-review", "bug"], github_api=mock_github_api)
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))
        state_path = tmp_path / "worker.state.json"
        monkeypatch.setattr(
            type(agent), "_state_path", property(lambda self: state_path)
        )

        # Pre-populate: worker already processed issue 43
        state_path.write_text(json.dumps({
            "timestamps": {},
            "processed": {"43": "2024-01-08T00:00:00Z"},
        }))

        mock_github_api.set_issues("in-progress", make_issue_list_items((43, "2024-01-01T00:00:00Z")))
        mock_github_api.set_issues("in-review", make_issue_list_items((42, "2024-01-01T00:00:00Z")))
        mock_github_api.set_issues("bug", [])
        mock_github_api.set_issues_detailed("in-progress", [
            make_issue_detail(
                number=43,
                title="Add user profiles",
                labels=["in-progress"],
                comments=[
                    {
                        "author": {"login": BOT_LOGIN},
                        "body": "Implementation started.",
                        "createdAt": "2024-01-07T00:00:00Z",
                    },
                ],
            ),
        ])
        mock_github_api.set_issues_detailed("in-review", [
            make_issue_detail(
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
        ])
        mock_github_api.set_issues_detailed("bug", [])
        mock_github_api.set_prs([
            make_pull_request(
                10,
                "issue-42-add-reminders",
                checks=[make_check_status("check", "COMPLETED", "FAILURE")],
            ),
        ])
        mock_github_api.set_failed_runs("issue-42-add-reminders", [WorkflowRun(id=99)])
        mock_github_api.set_failed_log(99, "error: test_reminders failed")

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
        self, tmp_path, mock_github_api, capture_popen, monkeypatch
    ):
        """in-review issue with no comments → waiting for human review → skip."""
        agent = make_agent(tmp_path, name="worker", required_labels=["in-progress", "in-review", "bug"], github_api=mock_github_api)
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))
        monkeypatch.setattr(
            type(agent), "_state_path", property(lambda self: tmp_path / "w.state.json")
        )

        mock_github_api.set_issues("in-progress", [])
        mock_github_api.set_issues("in-review", make_issue_list_items((42, "2024-01-01T00:00:00Z")))
        mock_github_api.set_issues("bug", [])
        mock_github_api.set_issues_detailed("in-progress", [])
        mock_github_api.set_issues_detailed("in-review", [
            make_issue_detail(number=42, labels=["in-review"], comments=[]),
        ])
        mock_github_api.set_issues_detailed("bug", [])

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        assert result.output == "No actionable issues"
        assert len(calls) == 0

    def test_in_review_no_comments_but_failing_ci_is_actionable(
        self, tmp_path, mock_github_api, capture_popen, monkeypatch
    ):
        """in-review with no comments but failing CI → still actionable.

        Bug fix: external signals (CI failure, merge conflict, review feedback)
        should override the "no comments → skip" logic for in-review issues.
        """
        agent = make_agent(tmp_path, name="worker", required_labels=["in-progress", "in-review", "bug"], github_api=mock_github_api)
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        mock_github_api.set_issues("in-progress", [])
        mock_github_api.set_issues("in-review", make_issue_list_items((42, "2024-01-01T00:00:00Z")))
        mock_github_api.set_issues("bug", [])
        mock_github_api.set_issues_detailed("in-progress", [])
        mock_github_api.set_issues_detailed("in-review", [
            make_issue_detail(number=42, labels=["in-review"], comments=[]),
        ])
        mock_github_api.set_issues_detailed("bug", [])
        mock_github_api.set_prs([
            make_pull_request(
                10,
                "issue-42-fix",
                checks=[make_check_status("check", "COMPLETED", "FAILURE")],
            ),
        ])
        mock_github_api.set_failed_runs("issue-42-fix", [WorkflowRun(id=99)])
        mock_github_api.set_failed_log(99, "ruff check failed")

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        assert len(calls) == 1
        prompt = extract_prompt(calls)
        assert "CI Status: FAILING" in prompt

    def test_in_review_no_comments_but_merge_conflict_is_actionable(
        self, tmp_path, mock_github_api, capture_popen, monkeypatch
    ):
        """in-review with no comments but merge conflict → still actionable."""
        agent = make_agent(tmp_path, name="worker", required_labels=["in-progress", "in-review", "bug"], github_api=mock_github_api)
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        mock_github_api.set_issues("in-progress", [])
        mock_github_api.set_issues("in-review", make_issue_list_items((42, "2024-01-01T00:00:00Z")))
        mock_github_api.set_issues("bug", [])
        mock_github_api.set_issues_detailed("in-progress", [])
        mock_github_api.set_issues_detailed("in-review", [
            make_issue_detail(number=42, labels=["in-review"], comments=[]),
        ])
        mock_github_api.set_issues_detailed("bug", [])
        mock_github_api.set_prs([
            make_pull_request(10, "issue-42-fix", mergeable="CONFLICTING"),
        ])

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        assert len(calls) == 1
        prompt = extract_prompt(calls)
        assert "Merge Status: CONFLICTING" in prompt

    def test_in_review_no_comments_but_review_feedback_is_actionable(
        self, tmp_path, mock_github_api, capture_popen, monkeypatch
    ):
        """in-review with no comments but review feedback → still actionable."""
        agent = make_agent(tmp_path, name="worker", required_labels=["in-progress", "in-review", "bug"], github_api=mock_github_api)
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        mock_github_api.set_issues("in-progress", [])
        mock_github_api.set_issues("in-review", make_issue_list_items((42, "2024-01-01T00:00:00Z")))
        mock_github_api.set_issues("bug", [])
        mock_github_api.set_issues_detailed("in-progress", [])
        mock_github_api.set_issues_detailed("in-review", [
            make_issue_detail(number=42, labels=["in-review"], comments=[]),
        ])
        mock_github_api.set_issues_detailed("bug", [])
        mock_github_api.set_prs([
            make_pull_request(
                10,
                "issue-42-fix",
                reviews=[PRReview(
                    author=CommentAuthor(login="alice"),
                    state="CHANGES_REQUESTED",
                    submitted_at="",
                )],
            ),
        ])

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        assert len(calls) == 1

    def test_in_review_bot_last_comment_but_failing_ci_is_actionable(
        self, tmp_path, mock_github_api, capture_popen, monkeypatch
    ):
        """in-review where bot has last comment but CI is failing → still actionable.

        Bug fix: bot having the last comment should not prevent the worker
        from waking when there are external signals (failing CI).
        """
        agent = make_agent(tmp_path, name="worker", required_labels=["in-progress", "in-review", "bug"], github_api=mock_github_api)
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        mock_github_api.set_issues("in-progress", [])
        mock_github_api.set_issues("in-review", make_issue_list_items((42, "2024-01-01T00:00:00Z")))
        mock_github_api.set_issues("bug", [])
        mock_github_api.set_issues_detailed("in-progress", [])
        mock_github_api.set_issues_detailed("in-review", [
            make_issue_detail(
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
        ])
        mock_github_api.set_issues_detailed("bug", [])
        mock_github_api.set_prs([
            make_pull_request(
                10,
                "issue-42-fix",
                checks=[make_check_status("check", "COMPLETED", "FAILURE")],
            ),
        ])
        mock_github_api.set_failed_runs("issue-42-fix", [WorkflowRun(id=99)])
        mock_github_api.set_failed_log(99, "test failed")

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
        self, tmp_path, mock_github_api, capture_popen, monkeypatch
    ):
        """Same issue number from both in-progress and in-review → fetched once."""
        agent = make_agent(tmp_path, name="worker", required_labels=["in-progress", "in-review", "bug"], github_api=mock_github_api)
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        # Both label searches return issue #42
        mock_github_api.set_issues("in-progress", make_issue_list_items((42, "2024-01-01T00:00:00Z")))
        mock_github_api.set_issues("in-review", make_issue_list_items((42, "2024-01-01T00:00:00Z")))
        mock_github_api.set_issues("bug", [])
        mock_github_api.set_issues_detailed("in-progress", [
            make_issue_detail(
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
        ])
        mock_github_api.set_issues_detailed("in-review", [
            make_issue_detail(
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
        ])
        mock_github_api.set_issues_detailed("bug", [])

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        agent.run()

        # Issue should only appear once in the prompt (dedup in fetch_issues_for_labels)
        assert len(calls) == 1


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
        prs = [make_pull_request(10, "feature-branch")]
        issues = [_make_issue(5)]
        assert _match_prs_to_issues(prs, issues) == {}

    def test_no_match_for_unrelated_issue_number(self):
        prs = [make_pull_request(10, "issue-99-fix")]
        issues = [_make_issue(5)]
        assert _match_prs_to_issues(prs, issues) == {}

    def test_multiple_prs_multiple_issues(self):
        prs = [
            make_pull_request(10, "issue-1-feat"),
            make_pull_request(11, "issue-2-fix"),
        ]
        issues = [_make_issue(1), _make_issue(2)]
        result = _match_prs_to_issues(prs, issues)
        assert len(result) == 2
        assert result[1].number == 10
        assert result[2].number == 11

    def test_non_numeric_issue_part_ignored(self):
        prs = [make_pull_request(10, "issue-abc-fix")]
        issues = [_make_issue(1)]
        assert _match_prs_to_issues(prs, issues) == {}


# =============================================================================
# Review handling — unit tests for pure function edge cases
# =============================================================================


class TestWorkerReviewHandling:
    def test_approved_review_no_action(self):
        reviews = [PRReview(author=CommentAuthor(login="bob"), state="APPROVED", submitted_at="")]
        assert _has_changes_requested(reviews) is False

    def test_later_approval_overrides_changes_requested(self):
        """Latest review per reviewer wins — later approval overrides changes request."""
        reviews = [
            PRReview(author=CommentAuthor(login="bob"), state="CHANGES_REQUESTED", submitted_at=""),
            PRReview(author=CommentAuthor(login="bob"), state="APPROVED", submitted_at=""),
        ]
        assert _has_changes_requested(reviews) is False

    def test_different_reviewers_mixed_outcome(self):
        """One reviewer approves, another requests changes."""
        reviews = [
            PRReview(author=CommentAuthor(login="alice"), state="APPROVED", submitted_at=""),
            PRReview(author=CommentAuthor(login="bob"), state="CHANGES_REQUESTED", submitted_at=""),
        ]
        assert _has_changes_requested(reviews) is True


# =============================================================================
# CI check extraction — unit tests for pure function edge cases
# =============================================================================


class TestWorkerCIChecks:
    def test_pending_checks_not_treated_as_failure(self):
        checks = [
            CheckStatus(name="deploy", state="PENDING", conclusion=""),
            CheckStatus(name="build", state="IN_PROGRESS", conclusion=""),
        ]
        assert _extract_failed_checks(checks) == []

    def test_multiple_ci_failures(self):
        checks = [
            CheckStatus(name="lint", state="COMPLETED", conclusion="FAILURE"),
            CheckStatus(name="test", state="COMPLETED", conclusion="FAILURE"),
        ]
        result = _extract_failed_checks(checks)
        assert len(result) == 2


# =============================================================================
# PR enrichment error handling — unit test
# =============================================================================


class TestWorkerPREnrichFailure:
    def test_api_pr_list_failure_is_noop(self, mock_github_api):
        """If API pr list fails, issues are left unchanged (fail-open)."""
        issue = _make_issue(1, labels=["in-review"])
        mock_github_api._list_prs_fail = True

        enrich_issues_with_pr_status([issue], api=mock_github_api)
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
        self, tmp_path, mock_github_api, capture_popen, monkeypatch
    ):
        """Bug issue with no comments → Claude runs bug fix workflow.

        Flow: issue labeled 'bug' with no comments → bypasses PM/Architect
        → pick_actionable_issue returns it → prompt assembled with Worker
        CLAUDE.md + issue data. Worker follows bug fix workflow.
        """
        agent = make_agent(
            tmp_path, name="worker", required_labels=["in-progress", "in-review", "bug"],
            github_api=mock_github_api,
        )
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        mock_github_api.set_issues("in-progress", [])
        mock_github_api.set_issues("in-review", [])
        mock_github_api.set_issues("bug", make_issue_list_items((50, "2024-01-01T00:00:00Z")))
        mock_github_api.set_issues_detailed("in-progress", [])
        mock_github_api.set_issues_detailed("in-review", [])
        mock_github_api.set_issues_detailed("bug", [
            make_issue_detail(
                number=50,
                title="Search fails when query is empty",
                body="When a user sends an empty search query, the app crashes.",
                labels=["bug"],
                comments=[],
            ),
        ])

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        prompt = extract_prompt(calls)
        assert "Worker Agent Prompt" in prompt
        assert "Search fails when query is empty" in prompt
        assert "Issue #50" in prompt

    def test_bug_prioritized_over_in_progress(
        self, tmp_path, mock_github_api, capture_popen, monkeypatch
    ):
        """Both bug and in-progress issues exist → picks bug first.

        Flow: issue #50 is labeled 'bug', issue #43 is labeled 'in-progress'
        → pick_actionable_issue sorts bugs before non-bugs
        → returns #50 → Worker fixes bug, not implementing #43.

        This verifies that bug fixes are prioritized over new feature work.
        """
        agent = make_agent(
            tmp_path, name="worker", required_labels=["in-progress", "in-review", "bug"],
            github_api=mock_github_api,
        )
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        mock_github_api.set_issues("in-progress", make_issue_list_items((43, "2024-01-01T00:00:00Z")))
        mock_github_api.set_issues("in-review", [])
        mock_github_api.set_issues("bug", make_issue_list_items((50, "2024-01-01T00:00:00Z")))
        mock_github_api.set_issues_detailed("in-progress", [
            make_issue_detail(
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
        ])
        mock_github_api.set_issues_detailed("in-review", [])
        mock_github_api.set_issues_detailed("bug", [
            make_issue_detail(
                number=50,
                title="Search fails when query is empty",
                body="When a user sends an empty search query, the app crashes.",
                labels=["bug"],
                comments=[],
            ),
        ])

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        prompt = extract_prompt(calls)
        # Should be working on #50 (bug), not #43 (feature)
        assert "Search fails when query is empty" in prompt
        assert "Issue #50" in prompt

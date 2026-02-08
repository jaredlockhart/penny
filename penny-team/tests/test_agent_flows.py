"""Integration tests tracing the expected flow of each agent type.

These tests exercise agent.run() end-to-end with realistic mock data,
verifying prompt assembly, issue filtering, PR status enrichment, and
Claude CLI invocation. They serve as both integration tests and
executable documentation of each agent's expected behavior.

Flow summary:
  backlog → requirements (PM) → specification (Architect) → in-progress (Worker) → in-review (Worker) → closed
"""

from __future__ import annotations

import json
from pathlib import Path

from penny_team.base import Agent

# --- Shared constants ---

BOT_SLUG = "penny-team"
BOT_LOGIN = "penny-team[bot]"
BOT_LOGINS = {BOT_SLUG, BOT_LOGIN}
TRUSTED_USERS = {"alice", "bob", BOT_SLUG, BOT_LOGIN}

# Realistic issue data templates
ISSUE_42_BASE = {
    "title": "Add reminders feature",
    "body": "Users should be able to set reminders via natural language.",
    "author": {"login": "alice"},
    "labels": [],
    "comments": [],
}


def _make_flow_agent(
    tmp_path: Path,
    name: str,
    required_labels: list[str],
) -> Agent:
    """Create an agent for flow testing with a real-ish prompt file."""
    agent_dir = tmp_path / "penny_team" / name
    agent_dir.mkdir(parents=True)

    # Use a realistic prompt stub that we can verify in the assembled command
    prompt_marker = f"# {name.title().replace('-', ' ')} Agent Prompt"
    (agent_dir / "CLAUDE.md").write_text(f"{prompt_marker}\n\nYou are the {name} agent.\n")

    agent = Agent(
        name=name,
        interval_seconds=300,
        working_dir=tmp_path,
        timeout_seconds=600,
        required_labels=required_labels,
        trusted_users=TRUSTED_USERS,
    )
    agent.prompt_path = agent_dir / "CLAUDE.md"
    return agent


def _extract_prompt(calls: list[tuple[tuple, dict]]) -> str:
    """Extract the prompt string from captured Popen calls."""
    assert calls, "Expected Popen to be called, but it was not"
    cmd = calls[0][0][0]  # First call, positional args, first arg (command list)
    p_index = cmd.index("-p")
    return cmd[p_index + 1]


def _result_event(text: str = "Task completed") -> str:
    """Create a stream-json result event line."""
    return json.dumps({"type": "result", "result": text})


def _issue_list_response(*numbers: int) -> str:
    """Create a gh issue list JSON response."""
    return json.dumps([{"number": n} for n in numbers])


def _issue_view_response(
    number: int = 42,
    title: str = "Add reminders feature",
    body: str = "Users should be able to set reminders via natural language.",
    author: str = "alice",
    labels: list[str] | None = None,
    comments: list[dict] | None = None,
) -> str:
    """Create a gh issue view JSON response."""
    return json.dumps({
        "title": title,
        "body": body,
        "author": {"login": author},
        "labels": [{"name": l} for l in (labels or ["requirements"])],
        "comments": comments or [],
    })


# =============================================================================
# Product Manager Flow
#
# Trigger: issues with label "requirements"
# Expected behavior:
#   - New issue with no comments → invoke Claude to post requirements
#   - Issue with bot requirements + user feedback → invoke Claude to refine
#   - Issue with bot as last commenter, no feedback → skip (no actionable issues)
# =============================================================================


class TestProductManagerFlow:
    """Product Manager agent processes issues labeled 'requirements'.

    The PM reads rough ideas, posts structured requirements, and refines
    them based on user feedback. Once the user is satisfied, they move
    the issue to 'specification' and the Architect takes over.
    """

    def test_new_issue_posts_requirements(
        self, tmp_path, mock_subprocess, capture_popen, monkeypatch
    ):
        """A fresh requirements issue with no comments triggers Claude CLI.

        Flow: issue has no comments → pick_actionable_issue returns it
        → prompt assembled with PM CLAUDE.md + issue data → Claude invoked.
        """
        agent = _make_flow_agent(tmp_path, "product-manager", ["requirements"])
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        # gh issue list returns issue #42
        mock_subprocess.add_response("issue list", stdout=_issue_list_response(42))
        # gh issue view returns the issue with no comments
        mock_subprocess.add_response(
            "issue view",
            stdout=_issue_view_response(
                number=42, labels=["requirements"], comments=[]
            ),
        )

        calls = capture_popen(stdout_lines=[_result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        assert result.output == "Task completed"

        # Verify the prompt contains both the agent prompt and issue data
        prompt = _extract_prompt(calls)
        assert "Product Manager Agent Prompt" in prompt
        assert "Add reminders feature" in prompt
        assert "Issue #42" in prompt
        assert "Pre-Fetched, Filtered" in prompt

    def test_user_feedback_triggers_refinement(
        self, tmp_path, mock_subprocess, capture_popen, monkeypatch
    ):
        """Issue with bot requirements comment + user feedback triggers Claude CLI.

        Flow: bot posted requirements, then user commented with feedback
        → last comment is from user (not bot) → pick_actionable_issue returns it
        → prompt includes full comment history so Claude can refine.
        """
        agent = _make_flow_agent(tmp_path, "product-manager", ["requirements"])
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        mock_subprocess.add_response("issue list", stdout=_issue_list_response(42))
        mock_subprocess.add_response(
            "issue view",
            stdout=_issue_view_response(
                number=42,
                labels=["requirements"],
                comments=[
                    {
                        "author": {"login": BOT_LOGIN},
                        "body": "## Requirements (Draft)\n\n**In Scope**: reminders",
                        "createdAt": "2024-01-01T00:00:00Z",
                    },
                    {
                        "author": {"login": "alice"},
                        "body": "Can we also support cancelling reminders?",
                        "createdAt": "2024-01-02T00:00:00Z",
                    },
                ],
            ),
        )

        calls = capture_popen(stdout_lines=[_result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        prompt = _extract_prompt(calls)
        # Prompt should contain both the requirements and user feedback
        assert "Requirements (Draft)" in prompt
        assert "cancelling reminders" in prompt

    def test_bot_last_comment_skips(
        self, tmp_path, mock_subprocess, capture_popen, monkeypatch
    ):
        """Issue where bot has the last comment → no actionable issues → skip.

        Flow: bot posted requirements, no user feedback since
        → pick_actionable_issue returns None → agent returns early
        → Claude CLI is NOT invoked.
        """
        agent = _make_flow_agent(tmp_path, "product-manager", ["requirements"])
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))
        monkeypatch.setattr(
            type(agent), "_state_path", property(lambda self: tmp_path / "pm.state.json")
        )

        mock_subprocess.add_response(
            "issue list",
            stdout=json.dumps([{"number": 42, "updatedAt": "2024-01-01T00:00:00Z"}]),
        )
        mock_subprocess.add_response(
            "issue view",
            stdout=_issue_view_response(
                number=42,
                labels=["requirements"],
                comments=[
                    {
                        "author": {"login": BOT_LOGIN},
                        "body": "## Requirements (Draft)\n\nPosted requirements.",
                        "createdAt": "2024-01-01T00:00:00Z",
                    },
                ],
            ),
        )

        calls = capture_popen(stdout_lines=[_result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        assert result.output == "No actionable issues"
        # Claude CLI should NOT have been called
        assert len(calls) == 0

    def test_skip_saves_state_success_does_not(
        self, tmp_path, mock_subprocess, capture_popen, monkeypatch
    ):
        """State is saved when agent skips (no actionable issues) but NOT after
        a successful Claude CLI run.

        Bug fix: state is only saved when pick_actionable_issue() returns None
        (all issues handled). This ensures has_work() keeps returning True until
        the entire queue is burned down across multiple cycles.
        """
        agent = _make_flow_agent(tmp_path, "product-manager", ["requirements"])
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))
        state_path = tmp_path / "pm.state.json"
        monkeypatch.setattr(
            type(agent), "_state_path", property(lambda self: state_path)
        )

        # --- Run 1: actionable issue → Claude runs, state NOT saved ---
        mock_subprocess.add_response("issue list", stdout=_issue_list_response(42))
        mock_subprocess.add_response(
            "issue view",
            stdout=_issue_view_response(
                number=42, labels=["requirements"], comments=[]
            ),
        )
        calls = capture_popen(stdout_lines=[_result_event()], returncode=0)

        agent.run()

        assert len(calls) == 1  # Claude CLI was called
        assert not state_path.exists()  # State NOT saved after success


# =============================================================================
# Architect Flow
#
# Trigger: issues with label "specification"
# Expected behavior:
#   - Issue with approved requirements but no spec → invoke Claude to write spec
#   - Issue with spec + user feedback → invoke Claude to revise
#   - Issue with spec, bot last comment → skip
# =============================================================================


class TestArchitectFlow:
    """Architect agent processes issues labeled 'specification'.

    The Architect reads PM-approved requirements and writes detailed
    implementation specifications. Once the user approves the spec,
    they move the issue to 'in-progress' and the Worker takes over.
    """

    def test_new_spec_issue_writes_spec(
        self, tmp_path, mock_subprocess, capture_popen, monkeypatch
    ):
        """Issue with approved requirements but no spec → Claude writes spec.

        Flow: issue has requirements comment from PM but no "Detailed Specification"
        → pick_actionable_issue returns it (PM comment is trusted, but PM is not
        the bot — user feedback exists) → prompt assembled with issue + requirements.
        """
        agent = _make_flow_agent(tmp_path, "architect", ["specification"])
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        mock_subprocess.add_response("issue list", stdout=_issue_list_response(42))
        mock_subprocess.add_response(
            "issue view",
            stdout=_issue_view_response(
                number=42,
                title="Add reminders feature",
                labels=["specification"],
                comments=[
                    {
                        "author": {"login": "alice"},
                        "body": "Requirements look good, moving to spec.",
                        "createdAt": "2024-01-03T00:00:00Z",
                    },
                ],
            ),
        )

        calls = capture_popen(stdout_lines=[_result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        prompt = _extract_prompt(calls)
        assert "Architect Agent Prompt" in prompt
        assert "Add reminders feature" in prompt
        assert "Issue #42" in prompt

    def test_user_feedback_on_spec_triggers_revision(
        self, tmp_path, mock_subprocess, capture_popen, monkeypatch
    ):
        """Spec exists + user posted feedback → Claude revises spec.

        Flow: bot posted spec, user commented with questions
        → last comment from user → actionable → prompt includes spec + feedback.
        """
        agent = _make_flow_agent(tmp_path, "architect", ["specification"])
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        mock_subprocess.add_response("issue list", stdout=_issue_list_response(42))
        mock_subprocess.add_response(
            "issue view",
            stdout=_issue_view_response(
                number=42,
                labels=["specification"],
                comments=[
                    {
                        "author": {"login": BOT_LOGIN},
                        "body": "## Detailed Specification\n\n**Technical Approach**: ...",
                        "createdAt": "2024-01-04T00:00:00Z",
                    },
                    {
                        "author": {"login": "bob"},
                        "body": "Can we use dateparser for time parsing?",
                        "createdAt": "2024-01-05T00:00:00Z",
                    },
                ],
            ),
        )

        calls = capture_popen(stdout_lines=[_result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        prompt = _extract_prompt(calls)
        assert "Detailed Specification" in prompt
        assert "dateparser" in prompt

    def test_bot_last_comment_on_spec_skips(
        self, tmp_path, mock_subprocess, capture_popen, monkeypatch
    ):
        """Bot posted spec, no user feedback → skip.

        Flow: bot has last comment → pick_actionable_issue returns None
        → agent returns "No actionable issues" → Claude CLI not invoked.
        """
        agent = _make_flow_agent(tmp_path, "architect", ["specification"])
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))
        monkeypatch.setattr(
            type(agent), "_state_path", property(lambda self: tmp_path / "arch.state.json")
        )

        mock_subprocess.add_response(
            "issue list",
            stdout=json.dumps([{"number": 42, "updatedAt": "2024-01-04T00:00:00Z"}]),
        )
        mock_subprocess.add_response(
            "issue view",
            stdout=_issue_view_response(
                number=42,
                labels=["specification"],
                comments=[
                    {
                        "author": {"login": BOT_LOGIN},
                        "body": "## Detailed Specification\n\nPosted spec.",
                        "createdAt": "2024-01-04T00:00:00Z",
                    },
                ],
            ),
        )

        calls = capture_popen(stdout_lines=[_result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        assert result.output == "No actionable issues"
        assert len(calls) == 0


# =============================================================================
# Worker Flow
#
# Trigger: issues with labels "in-progress" or "in-review"
# Expected behavior:
#   - in-progress issue with spec → invoke Claude to implement
#   - in-review with failing CI → invoke Claude with CI failure details
#   - in-review with merge conflict → invoke Claude with conflict info
#   - in-review with review feedback → invoke Claude to address feedback
#   - in-review, all passing, bot last comment → skip
#   - Both in-review (CI fail) + in-progress → picks in-review (priority)
# =============================================================================


class TestWorkerFlow:
    """Worker agent processes issues labeled 'in-progress' or 'in-review'.

    The Worker implements features from specs (in-progress) and handles
    PR feedback including failing CI, merge conflicts, and review comments
    (in-review). It's the only agent that transitions labels itself:
    in-progress → in-review after creating a PR.
    """

    def test_new_in_progress_issue_implements(
        self, tmp_path, mock_subprocess, capture_popen, monkeypatch
    ):
        """Fresh in-progress issue with spec → Claude implements feature.

        Flow: issue has spec from Architect, no PR yet
        → pick_actionable_issue returns it (no bot comment)
        → prompt assembled with Worker CLAUDE.md + issue + spec.
        """
        agent = _make_flow_agent(tmp_path, "worker", ["in-progress", "in-review"])
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        # Issue list: in-progress returns #42, in-review returns nothing
        mock_subprocess.add_response("issue list", stdout=_issue_list_response(42))
        mock_subprocess.add_response(
            "issue view",
            stdout=_issue_view_response(
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
        # PR enrichment: no open PRs
        mock_subprocess.add_response("pr list", stdout="[]")

        calls = capture_popen(stdout_lines=[_result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        prompt = _extract_prompt(calls)
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
        agent = _make_flow_agent(tmp_path, "worker", ["in-progress", "in-review"])
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        # Issue list for both labels
        mock_subprocess.add_response("issue list", stdout=_issue_list_response(42))
        mock_subprocess.add_response(
            "issue view",
            stdout=_issue_view_response(
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
        # PR list with failing CI check
        mock_subprocess.add_response(
            "pr list",
            stdout=json.dumps([
                {
                    "number": 10,
                    "headRefName": "issue-42-add-reminders",
                    "mergeable": "MERGEABLE",
                    "statusCheckRollup": [
                        {
                            "name": "check",
                            "state": "COMPLETED",
                            "conclusion": "FAILURE",
                            "context": "CI / check",
                        }
                    ],
                    "reviews": [],
                }
            ]),
        )
        # Failure log fetch
        mock_subprocess.add_response(
            "run list",
            stdout=json.dumps([{"databaseId": 99}]),
        )
        mock_subprocess.add_response(
            "run view",
            stdout="error: ruff check failed\n  penny_team/base.py:10: E501 line too long",
        )

        calls = capture_popen(stdout_lines=[_result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        prompt = _extract_prompt(calls)
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
        agent = _make_flow_agent(tmp_path, "worker", ["in-progress", "in-review"])
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        mock_subprocess.add_response("issue list", stdout=_issue_list_response(42))
        mock_subprocess.add_response(
            "issue view",
            stdout=_issue_view_response(
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
        # PR with merge conflict
        mock_subprocess.add_response(
            "pr list",
            stdout=json.dumps([
                {
                    "number": 10,
                    "headRefName": "issue-42-add-reminders",
                    "mergeable": "CONFLICTING",
                    "statusCheckRollup": [],
                    "reviews": [],
                }
            ]),
        )

        calls = capture_popen(stdout_lines=[_result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        prompt = _extract_prompt(calls)
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
        agent = _make_flow_agent(tmp_path, "worker", ["in-progress", "in-review"])
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        mock_subprocess.add_response("issue list", stdout=_issue_list_response(42))
        mock_subprocess.add_response(
            "issue view",
            stdout=_issue_view_response(
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
        # PR with changes requested review
        mock_subprocess.add_response(
            "pr list",
            stdout=json.dumps([
                {
                    "number": 10,
                    "headRefName": "issue-42-add-reminders",
                    "mergeable": "MERGEABLE",
                    "statusCheckRollup": [],
                    "reviews": [
                        {
                            "author": {"login": "alice"},
                            "state": "CHANGES_REQUESTED",
                        }
                    ],
                }
            ]),
        )

        calls = capture_popen(stdout_lines=[_result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        prompt = _extract_prompt(calls)
        assert "Issue #42" in prompt

    def test_in_review_all_passing_no_feedback_skips(
        self, tmp_path, mock_subprocess, capture_popen, monkeypatch
    ):
        """In-review, CI passing, no conflicts, no feedback, bot last → skip.

        Flow: everything looks good, bot has the last comment, no external
        signals → pick_actionable_issue returns None → agent skips.
        This represents a PR waiting for human review.
        """
        agent = _make_flow_agent(tmp_path, "worker", ["in-progress", "in-review"])
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
            stdout=_issue_view_response(
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
        # PR with all passing checks, no conflicts, no review feedback
        mock_subprocess.add_response(
            "pr list",
            stdout=json.dumps([
                {
                    "number": 10,
                    "headRefName": "issue-42-add-reminders",
                    "mergeable": "MERGEABLE",
                    "statusCheckRollup": [
                        {
                            "name": "check",
                            "state": "COMPLETED",
                            "conclusion": "SUCCESS",
                            "context": "CI / check",
                        }
                    ],
                    "reviews": [],
                }
            ]),
        )

        calls = capture_popen(stdout_lines=[_result_event()], returncode=0)

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

        This verifies that in-review work (especially broken CI) takes priority
        over starting new implementation work.
        """
        agent = _make_flow_agent(tmp_path, "worker", ["in-progress", "in-review"])
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        # Issue list returns both issues (they share the same "issue list" pattern)
        mock_subprocess.add_response(
            "issue list",
            stdout=json.dumps([{"number": 42}, {"number": 43}]),
        )
        # Issue view for #42 (in-review with bot comment)
        mock_subprocess.add_response(
            "issue view 42",
            stdout=_issue_view_response(
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
        # Issue view for #43 (in-progress, no comments)
        mock_subprocess.add_response(
            "issue view 43",
            stdout=_issue_view_response(
                number=43,
                title="Add user profiles",
                labels=["in-progress"],
                comments=[],
            ),
        )
        # PR list with failing CI on #42's branch
        mock_subprocess.add_response(
            "pr list",
            stdout=json.dumps([
                {
                    "number": 10,
                    "headRefName": "issue-42-add-reminders",
                    "mergeable": "MERGEABLE",
                    "statusCheckRollup": [
                        {
                            "name": "check",
                            "state": "COMPLETED",
                            "conclusion": "FAILURE",
                            "context": "CI / check",
                        }
                    ],
                    "reviews": [],
                }
            ]),
        )
        # Failure log for the CI failure
        mock_subprocess.add_response(
            "run list",
            stdout=json.dumps([{"databaseId": 99}]),
        )
        mock_subprocess.add_response(
            "run view",
            stdout="error: test_reminders failed",
        )

        calls = capture_popen(stdout_lines=[_result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        prompt = _extract_prompt(calls)
        # Should be working on #42 (in-review with CI failure), not #43
        assert "CI Status: FAILING" in prompt
        assert "Add reminders feature" in prompt

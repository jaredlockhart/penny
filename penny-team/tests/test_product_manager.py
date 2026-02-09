"""Integration tests for the Product Manager agent flow.

The PM reads rough ideas from issues labeled 'requirements', posts
structured requirements, and refines them based on user feedback.
Once the user is satisfied, they move the issue to 'specification'
and the Architect takes over.

Flow: backlog → requirements (PM) → specification (Architect)
"""

from __future__ import annotations

import json

from tests.conftest import (
    BOT_LOGIN,
    BOT_LOGINS,
    extract_prompt,
    issue_list_response,
    issue_view_response,
    make_agent,
    result_event,
)


class TestProductManagerFlow:
    def test_new_issue_posts_requirements(
        self, tmp_path, mock_subprocess, capture_popen, monkeypatch
    ):
        """A fresh requirements issue with no comments triggers Claude CLI.

        Flow: issue has no comments → pick_actionable_issue returns it
        → prompt assembled with PM CLAUDE.md + issue data → Claude invoked.
        """
        agent = make_agent(tmp_path, name="product-manager", required_labels=["requirements"])
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        mock_subprocess.add_response("issue list", stdout=issue_list_response(42))
        mock_subprocess.add_response(
            "issue view",
            stdout=issue_view_response(
                number=42, labels=["requirements"], comments=[]
            ),
        )

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        assert result.output == "Task completed"

        prompt = extract_prompt(calls)
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
        agent = make_agent(tmp_path, name="product-manager", required_labels=["requirements"])
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        mock_subprocess.add_response("issue list", stdout=issue_list_response(42))
        mock_subprocess.add_response(
            "issue view",
            stdout=issue_view_response(
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

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        prompt = extract_prompt(calls)
        assert "Requirements (Draft)" in prompt
        assert "cancelling reminders" in prompt

    def test_already_processed_no_new_feedback_skips(
        self, tmp_path, mock_subprocess, capture_popen, monkeypatch
    ):
        """Issue already processed by this agent with no new human feedback → skip.

        Flow: agent previously processed issue 42, bot has last comment,
        no human comments since → pick_actionable_issue returns None
        → agent returns early → Claude CLI is NOT invoked.
        """
        agent = make_agent(tmp_path, name="product-manager", required_labels=["requirements"])
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))
        state_path = tmp_path / "pm.state.json"
        monkeypatch.setattr(
            type(agent), "_state_path", property(lambda self: state_path)
        )

        # Pre-populate: agent already processed issue 42
        state_path.write_text(json.dumps({
            "timestamps": {},
            "processed": {"42": "2024-01-02T00:00:00Z"},
        }))

        mock_subprocess.add_response(
            "issue list",
            stdout=json.dumps([{"number": 42, "updatedAt": "2024-01-01T00:00:00Z"}]),
        )
        mock_subprocess.add_response(
            "issue view",
            stdout=issue_view_response(
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

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        assert result.output == "No actionable issues"
        assert len(calls) == 0

    def test_skip_saves_timestamps_success_saves_only_processed(
        self, tmp_path, mock_subprocess, capture_popen, monkeypatch
    ):
        """Timestamp state is saved when agent skips but NOT after success.
        Processed state IS saved after success (to track per-agent processing).

        Timestamps are only saved when pick_actionable_issue() returns None
        (all issues handled). This ensures has_work() keeps returning True
        until the queue is burned down. But processed state is saved after
        each successful run so the agent knows which issues it has handled.
        """
        agent = make_agent(tmp_path, name="product-manager", required_labels=["requirements"])
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))
        state_path = tmp_path / "pm.state.json"
        monkeypatch.setattr(
            type(agent), "_state_path", property(lambda self: state_path)
        )

        # --- Run 1: actionable issue → Claude runs ---
        mock_subprocess.add_response("issue list", stdout=issue_list_response(42))
        mock_subprocess.add_response(
            "issue view",
            stdout=issue_view_response(
                number=42, labels=["requirements"], comments=[]
            ),
        )
        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        agent.run()

        assert len(calls) == 1  # Claude CLI was called
        # Processed state IS saved, but timestamps are NOT
        state = json.loads(state_path.read_text())
        assert "42" in state["processed"]
        assert state["timestamps"] == {}

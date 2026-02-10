"""Integration tests for the Architect agent flow.

The Architect reads PM-approved requirements from issues labeled
'specification' and writes detailed implementation specs. Once the
user approves the spec, they move the issue to 'in-progress' and
the Worker takes over.

Flow: requirements (PM) → specification (Architect) → in-progress (Worker)
"""

from __future__ import annotations

import json

from tests.conftest import (
    BOT_LOGIN,
    BOT_LOGINS,
    extract_prompt,
    make_agent,
    make_issue_detail,
    make_issue_list_items,
    result_event,
)


class TestArchitectFlow:
    def test_new_spec_issue_writes_spec(
        self, tmp_path, mock_github_api, capture_popen, monkeypatch
    ):
        """Issue with approved requirements but no spec → Claude writes spec.

        Flow: issue has requirements comment from PM but no "Detailed Specification"
        → pick_actionable_issue returns it (PM comment is trusted, but PM is not
        the bot — user feedback exists) → prompt assembled with issue + requirements.
        """
        agent = make_agent(tmp_path, name="architect", required_labels=["specification"], github_api=mock_github_api)
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        mock_github_api.set_issues("specification", make_issue_list_items((42, "2024-01-01T00:00:00Z")))
        mock_github_api.set_issues_detailed("specification", [
            make_issue_detail(
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
        ])

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        prompt = extract_prompt(calls)
        assert "Architect Agent Prompt" in prompt
        assert "Add reminders feature" in prompt
        assert "Issue #42" in prompt

    def test_user_feedback_on_spec_triggers_revision(
        self, tmp_path, mock_github_api, capture_popen, monkeypatch
    ):
        """Spec exists + user posted feedback → Claude revises spec.

        Flow: bot posted spec, user commented with questions
        → last comment from user → actionable → prompt includes spec + feedback.
        """
        agent = make_agent(tmp_path, name="architect", required_labels=["specification"], github_api=mock_github_api)
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))

        mock_github_api.set_issues("specification", make_issue_list_items((42, "2024-01-01T00:00:00Z")))
        mock_github_api.set_issues_detailed("specification", [
            make_issue_detail(
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
        ])

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        prompt = extract_prompt(calls)
        assert "Detailed Specification" in prompt
        assert "dateparser" in prompt

    def test_already_processed_no_feedback_skips(
        self, tmp_path, mock_github_api, capture_popen, monkeypatch
    ):
        """Architect already processed issue, no new human feedback → skip.

        Flow: architect previously processed issue 42, bot has last comment,
        no human comments since → pick_actionable_issue returns None
        → agent returns "No actionable issues" → Claude CLI not invoked.
        """
        agent = make_agent(tmp_path, name="architect", required_labels=["specification"], github_api=mock_github_api)
        monkeypatch.setattr(type(agent), "_bot_logins", property(lambda self: BOT_LOGINS))
        state_path = tmp_path / "arch.state.json"
        monkeypatch.setattr(
            type(agent), "_state_path", property(lambda self: state_path)
        )

        # Pre-populate: architect already processed issue 42
        state_path.write_text(json.dumps({
            "timestamps": {},
            "processed": {"42": "2024-01-05T00:00:00Z"},
        }))

        mock_github_api.set_issues("specification", make_issue_list_items((42, "2024-01-04T00:00:00Z")))
        mock_github_api.set_issues_detailed("specification", [
            make_issue_detail(
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
        ])

        calls = capture_popen(stdout_lines=[result_event()], returncode=0)

        result = agent.run()

        assert result.success is True
        assert result.output == "No actionable issues"
        assert len(calls) == 0

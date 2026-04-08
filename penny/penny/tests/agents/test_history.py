"""Integration tests for HistoryAgent: daily/weekly summarization and preference extraction."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from penny.constants import PennyConstants
from penny.database.models import MessageLog, PromptLog
from penny.tests.conftest import TEST_SENDER


def _insert_message(penny, sender, content, direction, timestamp, **kwargs):
    """Insert a message with a specific timestamp (bypasses log_message's auto-now)."""
    with penny.db.get_session() as session:
        msg = MessageLog(
            direction=direction,
            sender=sender,
            content=content,
            timestamp=timestamp,
            **kwargs,
        )
        session.add(msg)
        session.commit()
        session.refresh(msg)
        return msg.id


# ── Summarization ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_summarize_today_creates_history_entry(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """HistoryAgent summarizes today's messages and stores a history entry."""
    config = make_config(history_interval=99999.0)

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        return mock_llm._make_text_response(request, "- Discussed quantum physics")

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            TEST_SENDER,
            "tell me about quantum physics",
        )

        await penny.history_agent.execute()

        entries = penny.db.history.get_recent(
            TEST_SENDER, PennyConstants.HistoryDuration.DAILY, limit=10
        )
        assert len(entries) >= 1
        assert "quantum physics" in entries[0].topics

        # Full system prompt structure assertion
        system_text = [
            m.get("content", "") for m in requests_seen[0]["messages"] if m.get("role") == "system"
        ][0]
        lines = system_text.split("\n")
        assert lines[0].startswith("Current date and time: ")
        rest = "\n".join(lines[1:])
        expected = """\

## Instructions
Summarize what the user said as a short bullet list of topics. \
Each bullet should be 5-10 words. \
Keep the user's exact wording for names, brands, and descriptors \
— do not paraphrase or correct unfamiliar words. \
Omit greetings, small talk, and meta-conversation. \
Return ONLY the bullet list, one topic per line, prefixed with "- "."""
        assert rest == expected, f"System prompt mismatch:\n{rest!r}\n\nvs expected:\n{expected!r}"


@pytest.mark.asyncio
async def test_summarize_today_skips_when_already_rolled_up(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """HistoryAgent skips summarization when history is already up-to-date."""
    config = make_config(history_interval=99999.0)

    summarize_calls: list[dict] = []

    def handler(request, count):
        messages = request.get("messages", [])
        system_msgs = [m for m in messages if m.get("role") == "system"]
        system_text = " ".join(m.get("content", "") for m in system_msgs)
        # Track only summarization calls (system prompt contains "bullet list")
        if "bullet list" in system_text.lower():
            summarize_calls.append(request)
        return mock_llm._make_text_response(request, "- Topics discussed")

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            TEST_SENDER,
            "hello penny",
        )

        # First run: should summarize
        await penny.history_agent.execute()
        assert len(summarize_calls) == 1

        # Second run (no new messages): should skip summarization
        await penny.history_agent.execute()
        assert len(summarize_calls) == 1

        # Outgoing message (Penny's response/notification) should NOT
        # trigger re-summarization — only incoming user messages count.
        # Regression: outgoing timestamps were checked by _already_rolled_up,
        # causing the same input to be re-summarized on every cycle.
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            penny.config.signal_number,
            "here's something interesting I found",
            recipient=TEST_SENDER,
        )
        await penny.history_agent.execute()
        assert len(summarize_calls) == 1  # Still no additional calls


@pytest.mark.asyncio
async def test_backfill_summarizes_past_days(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """HistoryAgent backfills past days even when today already has an entry.

    Regression: after a history reset, _summarize_today creates today's entry
    first.  _resolve_start_date used to return today's period_end, which is
    *after* midnight — so the backfill loop (cursor < midnight_today) found
    zero days.
    """
    config = make_config(history_interval=99999.0)

    def handler(request, count):
        return mock_llm._make_text_response(request, "- Historical topics")

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Seed a message from 2 days ago using direct insert
        two_days_ago = datetime.now(UTC).replace(
            hour=12, minute=0, second=0, microsecond=0, tzinfo=None
        ) - timedelta(days=2)
        _insert_message(
            penny,
            TEST_SENDER,
            "old message from two days ago",
            PennyConstants.MessageDirection.INCOMING,
            two_days_ago,
        )

        # Also add a today message so _summarize_today creates today's entry
        # before backfill runs — this is the scenario that triggers the bug
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            TEST_SENDER,
            "message from today",
        )

        await penny.history_agent.execute()

        entries = penny.db.history.get_recent(
            TEST_SENDER, PennyConstants.HistoryDuration.DAILY, limit=10
        )
        # Should have today's entry AND the backfilled past day
        past_entries = [e for e in entries if e.period_start.date() != datetime.now(UTC).date()]
        assert len(past_entries) >= 1, "Backfill must create entries for past days"


@pytest.mark.asyncio
async def test_backfill_skips_empty_days_without_consuming_budget(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Days with no messages must not consume max_days budget.

    Regression: _find_unsummarized_days returned empty days (no history entry,
    no messages) which counted toward max_days, preventing the scanner from
    reaching days that actually had messages to summarize.
    """
    config = make_config(history_interval=99999.0, history_max_days_per_run=1)

    def handler(request, count):
        return mock_llm._make_text_response(request, "- Historical topics")

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Day with messages: 4 days ago
        four_days_ago = datetime.now(UTC).replace(
            hour=12, minute=0, second=0, microsecond=0, tzinfo=None
        ) - timedelta(days=4)
        _insert_message(
            penny,
            TEST_SENDER,
            "message from four days ago",
            PennyConstants.MessageDirection.INCOMING,
            four_days_ago,
        )
        # Days 3 and 2 ago have NO messages — these are the empty gaps
        # Today message so _summarize_today runs first
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            TEST_SENDER,
            "message from today",
        )

        # With max_days=1 and old bug, the first empty day would consume the
        # budget and the 4-days-ago message would never get backfilled
        await penny.history_agent.execute()

        entries = penny.db.history.get_recent(
            TEST_SENDER, PennyConstants.HistoryDuration.DAILY, limit=10
        )
        past_entries = [e for e in entries if e.period_start.date() != datetime.now(UTC).date()]
        assert len(past_entries) >= 1, (
            "Backfill must reach past days even when empty days exist in between"
        )


@pytest.mark.asyncio
async def test_summarize_uses_only_user_messages(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """HistoryAgent summarizes only user messages, not Penny's responses."""
    config = make_config(history_interval=99999.0)

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        return mock_llm._make_text_response(request, "- Topics")

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            TEST_SENDER,
            "hello there",
        )
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            penny.config.signal_number,
            "hey back!",
            parent_id=1,
            recipient=TEST_SENDER,
        )

        await penny.history_agent.execute()

        assert len(requests_seen) >= 1
        first_msgs = requests_seen[0]["messages"]
        user_msgs = [m for m in first_msgs if m.get("role") == "user"]
        prompt_text = " ".join(m.get("content", "") for m in user_msgs)
        assert "hello there" in prompt_text
        assert "hey back" not in prompt_text


# ── Preference extraction ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_preference_extraction_stores_preferences(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """HistoryAgent extracts and stores user preferences from conversation."""
    config = make_config(history_interval=99999.0)

    def handler(request, count):
        messages = request.get("messages", [])
        user_msgs = [m for m in messages if m.get("role") == "user"]
        prompt_text = " ".join(m.get("content", "") for m in user_msgs)

        # Summarization call (has "User:" formatting)
        if "User:" in prompt_text:
            return mock_llm._make_text_response(request, "- Discussed coffee preferences")

        # Preference identification (pass 1) — check for identification keywords
        if "identify" in prompt_text.lower() or "new preference" in prompt_text.lower():
            result = json.dumps({"new": ["Single-origin coffee beans"], "existing": []})
            return mock_llm._make_text_response(request, result)

        # Preference valence classification (pass 2)
        if "classify" in prompt_text.lower() or "valence" in prompt_text.lower():
            result = json.dumps(
                {"preferences": [{"content": "Single-origin coffee beans", "valence": "positive"}]}
            )
            return mock_llm._make_text_response(request, result)

        return mock_llm._make_text_response(request, "- Topics")

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            TEST_SENDER,
            "I really love single-origin coffee beans",
        )

        await penny.history_agent.execute()

        prefs = penny.db.preferences.get_for_user(TEST_SENDER)
        if prefs:
            assert any("coffee" in p.content.lower() for p in prefs)
            coffee_prefs = [p for p in prefs if "coffee" in p.content.lower()]
            for p in coffee_prefs:
                assert p.source == "extracted"
                assert p.mention_count == 1


@pytest.mark.asyncio
async def test_existing_preference_mention_increments_count(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """When LLM identifies a known preference was discussed, mention_count goes up."""
    config = make_config(history_interval=99999.0)

    def handler(request, count):
        messages = request.get("messages", [])
        user_msgs = [m for m in messages if m.get("role") == "user"]
        prompt_text = " ".join(m.get("content", "") for m in user_msgs)

        # Identification: LLM recognizes known pref, no new topics
        if "identify" in prompt_text.lower() or "sorting" in prompt_text.lower():
            result = json.dumps({"new": [], "existing": ["dark roast coffee"]})
            return mock_llm._make_text_response(request, result)

        if "User:" in prompt_text:
            return mock_llm._make_text_response(request, "- Discussed coffee")

        return mock_llm._make_text_response(request, "- Topics")

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Seed an existing preference
        existing = penny.db.preferences.add(
            user=TEST_SENDER,
            content="dark roast coffee",
            valence="positive",
        )
        assert existing is not None
        assert existing.mention_count == 1

        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            TEST_SENDER,
            "I love dark roast coffee so much",
        )

        await penny.history_agent.execute()

        # The existing preference should have its mention count incremented
        updated = penny.db.preferences.get_by_id(existing.id)
        assert updated is not None
        assert updated.mention_count == 2

        # No duplicate preference should be created
        all_prefs = penny.db.preferences.get_for_user(TEST_SENDER)
        coffee_prefs = [p for p in all_prefs if "coffee" in p.content.lower()]
        assert len(coffee_prefs) == 1


@pytest.mark.asyncio
async def test_preference_extraction_marks_messages_processed(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Messages are marked processed after preference extraction, preventing re-bumps."""
    config = make_config(history_interval=99999.0)

    extract_call_count = 0

    def handler(request, count):
        nonlocal extract_call_count
        messages = request.get("messages", [])
        user_msgs = [m for m in messages if m.get("role") == "user"]
        prompt_text = " ".join(m.get("content", "") for m in user_msgs)

        if "identify" in prompt_text.lower() or "sorting" in prompt_text.lower():
            extract_call_count += 1
            result = json.dumps({"new": ["hiking trails"], "existing": []})
            return mock_llm._make_text_response(request, result)

        if "classify" in prompt_text.lower() or "valence" in prompt_text.lower():
            result = json.dumps(
                {"preferences": [{"content": "hiking trails", "valence": "positive"}]}
            )
            return mock_llm._make_text_response(request, result)

        return mock_llm._make_text_response(request, "- Topics")

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            TEST_SENDER,
            "I love hiking trails in the mountains",
        )

        # First extraction: should process the message
        await penny.history_agent.execute()
        first_count = extract_call_count

        # Second extraction: message is now processed, should NOT re-extract
        extract_call_count = 0
        await penny.history_agent.execute()

        # Identification should not be called again (no unprocessed messages)
        assert extract_call_count == 0, (
            f"Expected 0 identification calls on second run, got {extract_call_count}"
        )

        # Only one preference should exist (not duplicated)
        prefs = penny.db.preferences.get_for_user(TEST_SENDER)
        hiking_prefs = [p for p in prefs if "hiking" in p.content.lower()]
        assert len(hiking_prefs) == 1
        assert first_count >= 1


@pytest.mark.asyncio
async def test_failed_extraction_does_not_mark_processed(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """When preference extraction fails, messages stay unprocessed for retry."""
    config = make_config(history_interval=99999.0)

    call_count = 0

    def handler(request, count):
        nonlocal call_count
        messages = request.get("messages", [])
        user_msgs = [m for m in messages if m.get("role") == "user"]
        prompt_text = " ".join(m.get("content", "") for m in user_msgs)

        # Identification calls: fail on first run, succeed on second
        if "identify" in prompt_text.lower() or "sorting" in prompt_text.lower():
            call_count += 1
            if call_count == 1:
                return mock_llm._make_text_response(request, "INVALID JSON")
            result = json.dumps({"new": ["espresso drinks"], "existing": []})
            return mock_llm._make_text_response(request, result)

        if "classify" in prompt_text.lower() or "valence" in prompt_text.lower():
            result = json.dumps(
                {"preferences": [{"content": "espresso drinks", "valence": "positive"}]}
            )
            return mock_llm._make_text_response(request, result)

        return mock_llm._make_text_response(request, "- Topics")

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            TEST_SENDER,
            "I really enjoy espresso drinks",
        )

        # First run: identification returns invalid JSON, extraction fails
        await penny.history_agent.execute()

        # Messages should still be unprocessed
        unprocessed = penny.db.messages.get_unprocessed(TEST_SENDER, limit=100)
        assert len(unprocessed) >= 1

        # No preference should be created
        prefs = penny.db.preferences.get_for_user(TEST_SENDER)
        espresso_prefs = [p for p in prefs if "espresso" in p.content.lower()]
        assert len(espresso_prefs) == 0

        # Second run: identification succeeds, messages get processed
        await penny.history_agent.execute()

        unprocessed = penny.db.messages.get_unprocessed(TEST_SENDER, limit=100)
        assert len(unprocessed) == 0

        prefs = penny.db.preferences.get_for_user(TEST_SENDER)
        espresso_prefs = [p for p in prefs if "espresso" in p.content.lower()]
        assert len(espresso_prefs) == 1


# ── Reaction handling ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reactions_to_regular_messages_create_no_preferences(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Reactions to regular Penny messages are marked processed with no preference created."""
    config = make_config(history_interval=99999.0)
    mock_llm.set_response_handler(
        lambda req, count: mock_llm._make_text_response(req, "- No topics")
    )

    async with running_penny(config) as penny:
        msg_id = penny.db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            TEST_SENDER,
            "You should try hiking near Boulder!",
        )
        _insert_message(
            penny,
            TEST_SENDER,
            "\U0001f44d",
            PennyConstants.MessageDirection.INCOMING,
            datetime.now(UTC).replace(tzinfo=None),
            is_reaction=True,
            parent_id=msg_id,
        )

        await penny.history_agent.execute()

        assert penny.db.preferences.get_for_user(TEST_SENDER) == []
        assert penny.db.messages.get_user_reactions(TEST_SENDER, limit=100) == []


@pytest.mark.asyncio
async def test_reaction_without_parent_is_marked_processed(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Reactions without a parent message are still marked processed."""
    config = make_config(history_interval=99999.0)
    mock_llm.set_response_handler(
        lambda req, count: mock_llm._make_text_response(req, "- No topics")
    )

    async with running_penny(config) as penny:
        _insert_message(
            penny,
            TEST_SENDER,
            "\U0001f44d",
            PennyConstants.MessageDirection.INCOMING,
            datetime.now(UTC).replace(tzinfo=None),
            is_reaction=True,
        )

        await penny.history_agent.execute()

        assert penny.db.preferences.get_for_user(TEST_SENDER) == []
        assert penny.db.messages.get_user_reactions(TEST_SENDER, limit=100) == []


# ── Helpers ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_thought_reaction_sets_valence_not_preference(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Reactions to thought notification messages set valence on the thought, not preferences."""
    config = make_config(history_interval=99999.0)

    def handler(request, count):
        messages = request.get("messages", [])
        user_msgs = [m for m in messages if m.get("role") == "user"]
        prompt_text = " ".join(m.get("content", "") for m in user_msgs)
        if "User:" in prompt_text:
            return mock_llm._make_text_response(request, "- No topics")
        return mock_llm._make_text_response(request, "- Topics")

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Store a thought, then log a notification message linked to it
        thought = penny.db.thoughts.add(TEST_SENDER, "Interesting content about guitar amps")
        assert thought is not None
        notif_id = penny.db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            TEST_SENDER,
            thought.content[:200],
            recipient=TEST_SENDER,
            thought_id=thought.id,
        )
        # React to the notification (thumbs up)
        _insert_message(
            penny,
            TEST_SENDER,
            "\U0001f44d",
            PennyConstants.MessageDirection.INCOMING,
            datetime.now(UTC).replace(tzinfo=None),
            is_reaction=True,
            parent_id=notif_id,
        )

        await penny.history_agent.execute()

        # Valence should be stored on the thought
        updated = penny.db.thoughts.get_by_id(thought.id)
        assert updated is not None
        assert updated.valence == 1, f"Expected valence=1, got {updated.valence}"

        # No preference should have been created from this reaction
        prefs = penny.db.preferences.get_for_user(TEST_SENDER)
        assert len(prefs) == 0, f"Expected no preferences, got {prefs}"

        # Reaction should be marked processed
        reactions = penny.db.messages.get_user_reactions(TEST_SENDER, limit=100)
        assert len(reactions) == 0


def test_reaction_emoji_classification():
    """HistoryAgent classifies reaction emojis as 1, -1, or None."""
    from penny.agents.history import HistoryAgent

    classify = HistoryAgent._emoji_to_int_valence
    assert classify("\u2764\ufe0f") == 1
    assert classify("\U0001f44d") == 1
    assert classify("\U0001f44e") == -1
    assert classify("\U0001f937") is None


@pytest.mark.asyncio
async def test_known_preferences_context(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """_build_known_preferences_context formats existing preferences for dedup."""
    config = make_config(history_interval=99999.0)

    async with running_penny(config) as penny:
        from penny.database.models import Preference

        existing = [
            Preference(
                user=TEST_SENDER,
                content="Jazz music",
                valence="positive",
            ),
            Preference(
                user=TEST_SENDER,
                content="Country music",
                valence="negative",
            ),
        ]
        result = penny.history_agent._build_known_preferences_context(existing)
        assert "Jazz music" in result
        assert "Country music" in result
        assert "positive" in result
        assert "negative" in result

        assert penny.history_agent._build_known_preferences_context([]) == ""


@pytest.mark.asyncio
async def test_no_messages_produces_no_history(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """HistoryAgent does nothing when there are no messages to summarize."""
    config = make_config(history_interval=99999.0)

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        return mock_llm._make_text_response(request, "- Topics")

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        await penny.history_agent.execute()

        entries = penny.db.history.get_recent(
            TEST_SENDER, PennyConstants.HistoryDuration.DAILY, limit=10
        )
        assert len(entries) == 0


# ── Weekly rollup ─────────────────────────────────────────────────────────


def _seed_daily_entries(penny, user, monday):
    """Seed daily history entries for a full Mon-Sun week starting at monday."""
    for i in range(7):
        day = monday + timedelta(days=i)
        penny.db.history.add(
            user=user,
            period_start=day,
            period_end=day + timedelta(days=1),
            duration=PennyConstants.HistoryDuration.DAILY,
            topics=f"- Topic from {day.strftime('%b %-d')}",
        )


@pytest.mark.asyncio
async def test_weekly_rollup_creates_entry_from_daily_entries(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Weekly rollup summarizes a completed week's daily entries into a weekly entry.

    Seeds both a completed past week AND current-week entries to ensure
    the scan starts from the earliest entry, not the most recent.
    """
    config = make_config(history_interval=99999.0)

    def handler(request, count):
        return mock_llm._make_text_response(request, "- Weekly themes discussed")

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
        two_weeks_ago_monday = today - timedelta(days=today.weekday() + 14)
        _seed_daily_entries(penny, TEST_SENDER, two_weeks_ago_monday)

        # Also seed a current-week entry — the scan must still find the past week
        penny.db.history.add(
            user=TEST_SENDER,
            period_start=today,
            period_end=today + timedelta(hours=12),
            duration=PennyConstants.HistoryDuration.DAILY,
            topics="- Current week topic",
        )

        system_prompt = await penny.history_agent._build_system_prompt(TEST_SENDER)
        await penny.history_agent._rollup_completed_weeks(TEST_SENDER, system_prompt, "test-run-id")

        weekly_entries = penny.db.history.get_recent(
            TEST_SENDER, PennyConstants.HistoryDuration.WEEKLY, limit=10
        )
        assert len(weekly_entries) == 1
        assert weekly_entries[0].period_start == two_weeks_ago_monday
        assert "Weekly themes" in weekly_entries[0].topics


@pytest.mark.asyncio
async def test_weekly_rollup_skips_incomplete_week(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Weekly rollup does not create an entry for the current (incomplete) week."""
    config = make_config(history_interval=99999.0)

    def handler(request, count):
        return mock_llm._make_text_response(request, "- Should not appear")

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Seed daily entries for the current week only
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
        this_monday = today - timedelta(days=today.weekday())
        for i in range(today.weekday() + 1):
            day = this_monday + timedelta(days=i)
            penny.db.history.add(
                user=TEST_SENDER,
                period_start=day,
                period_end=day + timedelta(days=1),
                duration=PennyConstants.HistoryDuration.DAILY,
                topics=f"- Topic from today's week day {i}",
            )

        system_prompt = await penny.history_agent._build_system_prompt(TEST_SENDER)
        await penny.history_agent._rollup_completed_weeks(TEST_SENDER, system_prompt, "test-run-id")

        weekly_entries = penny.db.history.get_recent(
            TEST_SENDER, PennyConstants.HistoryDuration.WEEKLY, limit=10
        )
        assert len(weekly_entries) == 0


@pytest.mark.asyncio
async def test_history_context_includes_weekly_entries(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """_history_section includes both weekly and daily entries."""
    config = make_config(history_interval=99999.0)

    async with running_penny(config) as penny:
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)

        # Seed a weekly entry from 2 weeks ago
        two_weeks_ago_monday = today - timedelta(days=today.weekday() + 14)
        penny.db.history.add(
            user=TEST_SENDER,
            period_start=two_weeks_ago_monday,
            period_end=two_weeks_ago_monday + timedelta(days=7),
            duration=PennyConstants.HistoryDuration.WEEKLY,
            topics="- Discussed AI developments\n- Talked about cooking",
        )

        # Seed a daily entry for today
        penny.db.history.add(
            user=TEST_SENDER,
            period_start=today,
            period_end=today + timedelta(hours=12),
            duration=PennyConstants.HistoryDuration.DAILY,
            topics="- Morning coffee chat",
        )

        context = penny.chat_agent._history_section(TEST_SENDER)
        assert context is not None
        assert "Week of" in context
        assert "AI developments" in context
        assert "Morning coffee chat" in context
        # Weekly entries should appear before daily
        week_pos = context.index("Week of")
        daily_pos = context.index("Morning coffee chat")
        assert week_pos < daily_pos


# ── Knowledge extraction ────────────────────────────────────────────────


def _insert_prompt_with_browse(penny, url, title, page_content):
    """Insert a prompt log with a browse tool result."""
    browse_header = PennyConstants.BROWSE_PAGE_HEADER
    tool_content = f"{browse_header}{url}\nTitle: {title}\nURL: {url}\n\n{page_content}"
    messages = json.dumps(
        [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": None, "tool_calls": []},
            {"role": "tool", "tool_call_id": "call_1", "content": tool_content},
        ]
    )
    with penny.db.get_session() as session:
        prompt = PromptLog(
            model="test",
            messages=messages,
            response=json.dumps({"choices": []}),
            agent_name="chat",
            prompt_type="user_message",
        )
        session.add(prompt)
        session.commit()
        session.refresh(prompt)
        return prompt.id


@pytest.mark.asyncio
async def test_extract_knowledge_from_browse_results(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Knowledge extraction creates an entry from browse tool results."""
    config = make_config(history_interval=99999.0)

    def handler(request, count):
        return mock_llm._make_text_response(
            request, "The Eggnog is a tube overdrive pedal by TubeSteader."
        )

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        _insert_prompt_with_browse(
            penny,
            "https://tubesteader.com/products/eggnog",
            "TubeSteader Eggnog",
            "The Eggnog uses a 12AX7 tube driven at 250 VDC.",
        )

        await penny.history_agent._extract_knowledge()

        entry = penny.db.knowledge.get_by_url("https://tubesteader.com/products/eggnog")
        assert entry is not None
        assert entry.title == "TubeSteader Eggnog"
        assert "tube overdrive" in entry.summary


@pytest.mark.asyncio
async def test_extract_knowledge_upserts_existing_url(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Re-browsing the same URL aggregates into the existing knowledge entry."""
    config = make_config(history_interval=99999.0)

    call_count = 0

    def handler(request, count):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return mock_llm._make_text_response(request, "First summary of the page.")
        return mock_llm._make_text_response(request, "Updated summary with new info.")

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        _insert_prompt_with_browse(
            penny,
            "https://example.com/page",
            "Example Page",
            "Original content.",
        )
        await penny.history_agent._extract_knowledge()

        entry = penny.db.knowledge.get_by_url("https://example.com/page")
        assert entry is not None
        assert entry.summary == "First summary of the page."

        # Insert another prompt with the same URL
        penny.chat_agent.clear_conversation_embedding_cache()
        _insert_prompt_with_browse(
            penny,
            "https://example.com/page",
            "Example Page",
            "Updated content with more details.",
        )
        await penny.history_agent._extract_knowledge()

        entry = penny.db.knowledge.get_by_url("https://example.com/page")
        assert entry is not None
        assert entry.summary == "Updated summary with new info."


@pytest.mark.asyncio
async def test_extract_knowledge_respects_watermark(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Knowledge extraction only processes prompts after the watermark."""
    config = make_config(history_interval=99999.0)

    summaries_generated = []

    def handler(request, count):
        summary = f"Summary {len(summaries_generated) + 1}"
        summaries_generated.append(summary)
        return mock_llm._make_text_response(request, summary)

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        _insert_prompt_with_browse(penny, "https://a.com", "Page A", "Content A")
        _insert_prompt_with_browse(penny, "https://b.com", "Page B", "Content B")

        # First extraction processes both
        await penny.history_agent._extract_knowledge()
        assert penny.db.knowledge.get_by_url("https://a.com") is not None
        assert penny.db.knowledge.get_by_url("https://b.com") is not None

        summaries_generated.clear()

        # Add a third prompt
        _insert_prompt_with_browse(penny, "https://c.com", "Page C", "Content C")
        await penny.history_agent._extract_knowledge()

        # Only the new prompt should be processed
        assert len(summaries_generated) == 1
        assert penny.db.knowledge.get_by_url("https://c.com") is not None

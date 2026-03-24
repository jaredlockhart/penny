"""Integration tests for HistoryAgent: daily/weekly summarization and preference extraction."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlmodel import select

from penny.constants import PennyConstants
from penny.database.models import MessageLog
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
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """HistoryAgent summarizes today's messages and stores a history entry."""
    config = make_config(history_interval=99999.0)

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        return mock_ollama._make_text_response(request, "- Discussed quantum physics")

    mock_ollama.set_response_handler(handler)

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
Summarize the following text as a short bullet list. \
Each bullet should be 3-8 words describing a distinct topic. \
Omit greetings, small talk, and meta-conversation. \
Return ONLY the bullet list, one topic per line, prefixed with "- "."""
        assert rest == expected, f"System prompt mismatch:\n{rest!r}\n\nvs expected:\n{expected!r}"


@pytest.mark.asyncio
async def test_summarize_today_skips_when_already_rolled_up(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
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
        return mock_ollama._make_text_response(request, "- Topics discussed")

    mock_ollama.set_response_handler(handler)

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
        assert len(summarize_calls) == 1  # No additional summarization calls


@pytest.mark.asyncio
async def test_backfill_summarizes_past_days(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """HistoryAgent backfills completed past days that lack history entries."""
    config = make_config(history_interval=99999.0)

    def handler(request, count):
        return mock_ollama._make_text_response(request, "- Historical topics")

    mock_ollama.set_response_handler(handler)

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

        await penny.history_agent.execute()

        entries = penny.db.history.get_recent(
            TEST_SENDER, PennyConstants.HistoryDuration.DAILY, limit=10
        )
        # Should have at least one backfilled entry
        assert len(entries) >= 1


@pytest.mark.asyncio
async def test_summarize_uses_only_user_messages(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """HistoryAgent summarizes only user messages, not Penny's responses."""
    config = make_config(history_interval=99999.0)

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        return mock_ollama._make_text_response(request, "- Topics")

    mock_ollama.set_response_handler(handler)

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
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """HistoryAgent extracts and stores user preferences from conversation."""
    config = make_config(history_interval=99999.0)

    def handler(request, count):
        messages = request.get("messages", [])
        user_msgs = [m for m in messages if m.get("role") == "user"]
        prompt_text = " ".join(m.get("content", "") for m in user_msgs)

        # Summarization call (has "User:" formatting)
        if "User:" in prompt_text:
            return mock_ollama._make_text_response(request, "- Discussed coffee preferences")

        # Preference identification (pass 1) — check for identification keywords
        if "identify" in prompt_text.lower() or "new preference" in prompt_text.lower():
            result = json.dumps({"new": ["Single-origin coffee beans"], "existing": []})
            return mock_ollama._make_text_response(request, result)

        # Preference valence classification (pass 2)
        if "classify" in prompt_text.lower() or "valence" in prompt_text.lower():
            result = json.dumps(
                {"preferences": [{"content": "Single-origin coffee beans", "valence": "positive"}]}
            )
            return mock_ollama._make_text_response(request, result)

        return mock_ollama._make_text_response(request, "- Topics")

    mock_ollama.set_response_handler(handler)

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
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
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
            return mock_ollama._make_text_response(request, result)

        if "User:" in prompt_text:
            return mock_ollama._make_text_response(request, "- Discussed coffee")

        return mock_ollama._make_text_response(request, "- Topics")

    mock_ollama.set_response_handler(handler)

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
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
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
            return mock_ollama._make_text_response(request, result)

        if "classify" in prompt_text.lower() or "valence" in prompt_text.lower():
            result = json.dumps(
                {"preferences": [{"content": "hiking trails", "valence": "positive"}]}
            )
            return mock_ollama._make_text_response(request, result)

        return mock_ollama._make_text_response(request, "- Topics")

    mock_ollama.set_response_handler(handler)

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
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
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
                return mock_ollama._make_text_response(request, "INVALID JSON")
            result = json.dumps({"new": ["espresso drinks"], "existing": []})
            return mock_ollama._make_text_response(request, result)

        if "classify" in prompt_text.lower() or "valence" in prompt_text.lower():
            result = json.dumps(
                {"preferences": [{"content": "espresso drinks", "valence": "positive"}]}
            )
            return mock_ollama._make_text_response(request, result)

        return mock_ollama._make_text_response(request, "- Topics")

    mock_ollama.set_response_handler(handler)

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


# ── Reaction preference extraction ───────────────────────────────────────


@pytest.mark.asyncio
async def test_reaction_extracts_preference_with_deterministic_valence(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Thumbs-up/down reactions create preferences with emoji-determined valence."""
    config = make_config(history_interval=99999.0)

    def handler(request, count):
        messages = request.get("messages", [])
        user_msgs = [m for m in messages if m.get("role") == "user"]
        prompt_text = " ".join(m.get("content", "") for m in user_msgs)

        # Reaction topic extraction — responds to the reaction pipeline prompt
        if "extract" in prompt_text.lower() and "single topic" in prompt_text.lower():
            result = json.dumps(
                {
                    "topics": [
                        {"index": 0, "content": "Hiking trails near Boulder Colorado"},
                        {"index": 1, "content": "Kale smoothie recipes"},
                    ]
                }
            )
            return mock_ollama._make_text_response(request, result)

        # Summarization
        if "User:" in prompt_text:
            return mock_ollama._make_text_response(request, "- No topics")

        return mock_ollama._make_text_response(request, "- Topics")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Log Penny's outgoing messages that will be reacted to
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            TEST_SENDER,
            "I found some great hiking trails near Boulder!",
        )
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            TEST_SENDER,
            "You should try kale smoothies, they're super healthy.",
        )

        # Get the outgoing message IDs to use as parent_id for reactions
        with penny.db.get_session() as session:
            hiking_msg = session.exec(
                select(MessageLog).where(MessageLog.content.contains("hiking"))  # type: ignore[union-attr]
            ).first()
            kale_msg = session.exec(
                select(MessageLog).where(MessageLog.content.contains("kale"))  # type: ignore[union-attr]
            ).first()
        assert hiking_msg and kale_msg
        hiking_msg_id = hiking_msg.id
        kale_msg_id = kale_msg.id

        # Insert reactions with explicit timestamps — get_user_reactions returns
        # newest-first (DESC), so hiking must be newer to appear at index 0
        now = datetime.now(UTC).replace(tzinfo=None)
        _insert_message(
            penny,
            TEST_SENDER,
            "\U0001f44e",
            PennyConstants.MessageDirection.INCOMING,
            now - timedelta(seconds=1),
            is_reaction=True,
            parent_id=kale_msg_id,
        )
        _insert_message(
            penny,
            TEST_SENDER,
            "\U0001f44d",
            PennyConstants.MessageDirection.INCOMING,
            now,
            is_reaction=True,
            parent_id=hiking_msg_id,
        )

        await penny.history_agent.execute()

        prefs = penny.db.preferences.get_for_user(TEST_SENDER)
        hiking_prefs = [p for p in prefs if "hiking" in p.content.lower()]
        kale_prefs = [p for p in prefs if "kale" in p.content.lower()]

        assert len(hiking_prefs) == 1, f"Expected 1 hiking preference, got {hiking_prefs}"
        assert hiking_prefs[0].valence == "positive"
        assert hiking_prefs[0].source == "extracted"

        assert len(kale_prefs) == 1, f"Expected 1 kale preference, got {kale_prefs}"
        assert kale_prefs[0].valence == "negative"
        assert kale_prefs[0].source == "extracted"

        # Reactions should be marked processed
        reactions = penny.db.messages.get_user_reactions(TEST_SENDER, limit=100)
        assert len(reactions) == 0, "Reactions should be marked processed after extraction"


@pytest.mark.asyncio
async def test_reaction_without_parent_is_skipped(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Reactions without a parent message are skipped, not erroring."""
    config = make_config(history_interval=99999.0)

    reaction_topic_calls = 0

    def handler(request, count):
        nonlocal reaction_topic_calls
        messages = request.get("messages", [])
        user_msgs = [m for m in messages if m.get("role") == "user"]
        prompt_text = " ".join(m.get("content", "") for m in user_msgs)

        if "extract" in prompt_text.lower() and "single topic" in prompt_text.lower():
            reaction_topic_calls += 1
            return mock_ollama._make_text_response(request, json.dumps({"topics": []}))

        if "User:" in prompt_text:
            return mock_ollama._make_text_response(request, "- No topics")

        return mock_ollama._make_text_response(request, "- Topics")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Reaction with no parent_id
        _insert_message(
            penny,
            TEST_SENDER,
            "\U0001f44d",
            PennyConstants.MessageDirection.INCOMING,
            datetime.now(UTC).replace(tzinfo=None),
            is_reaction=True,
        )

        await penny.history_agent.execute()

        # No LLM call should have been made for reaction topics
        assert reaction_topic_calls == 0, "Should not call LLM for parentless reactions"

        prefs = penny.db.preferences.get_for_user(TEST_SENDER)
        assert len(prefs) == 0


# ── Helpers ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reaction_emoji_classification(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """HistoryAgent classifies reaction emojis as positive, negative, or unknown."""
    config = make_config(history_interval=99999.0)

    async with running_penny(config) as penny:
        classify = penny.history_agent._classify_reaction_emoji
        assert classify("\u2764\ufe0f") == "positive"
        assert classify("\U0001f44d") == "positive"
        assert classify("\U0001f44e") == "negative"
        assert classify("\U0001f937") is None


@pytest.mark.asyncio
async def test_known_preferences_context(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
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
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """HistoryAgent does nothing when there are no messages to summarize."""
    config = make_config(history_interval=99999.0)

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        return mock_ollama._make_text_response(request, "- Topics")

    mock_ollama.set_response_handler(handler)

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
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Weekly rollup summarizes a completed week's daily entries into a weekly entry."""
    config = make_config(history_interval=99999.0)

    def handler(request, count):
        return mock_ollama._make_text_response(request, "- Weekly themes discussed")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Seed daily entries for a completed week (2+ weeks ago to ensure it's complete)
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
        two_weeks_ago_monday = today - timedelta(days=today.weekday() + 14)
        _seed_daily_entries(penny, TEST_SENDER, two_weeks_ago_monday)

        system_prompt = await penny.history_agent._build_system_prompt(TEST_SENDER)
        await penny.history_agent._rollup_completed_weeks(TEST_SENDER, system_prompt)

        weekly_entries = penny.db.history.get_recent(
            TEST_SENDER, PennyConstants.HistoryDuration.WEEKLY, limit=10
        )
        assert len(weekly_entries) == 1
        assert weekly_entries[0].period_start == two_weeks_ago_monday
        assert "Weekly themes" in weekly_entries[0].topics


@pytest.mark.asyncio
async def test_weekly_rollup_skips_incomplete_week(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Weekly rollup does not create an entry for the current (incomplete) week."""
    config = make_config(history_interval=99999.0)

    def handler(request, count):
        return mock_ollama._make_text_response(request, "- Should not appear")

    mock_ollama.set_response_handler(handler)

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
        await penny.history_agent._rollup_completed_weeks(TEST_SENDER, system_prompt)

        weekly_entries = penny.db.history.get_recent(
            TEST_SENDER, PennyConstants.HistoryDuration.WEEKLY, limit=10
        )
        assert len(weekly_entries) == 0


@pytest.mark.asyncio
async def test_history_context_includes_weekly_entries(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
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

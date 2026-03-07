"""Integration tests for HistoryAgent: daily summarization and preference extraction."""

import json
from datetime import UTC, datetime, timedelta

import pytest

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

    def handler(request, count):
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
async def test_summarize_formats_messages_with_direction(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """HistoryAgent formats messages with User/Penny direction labels."""
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
        assert "User:" in prompt_text
        assert "Penny:" in prompt_text


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
            result = json.dumps({"topics": ["Single-origin coffee beans"]})
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


@pytest.mark.asyncio
async def test_preference_extraction_skips_already_extracted_days(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """HistoryAgent skips preference extraction for days already processed."""
    config = make_config(history_interval=99999.0)

    def handler(request, count):
        return mock_ollama._make_text_response(request, "- Topics")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Seed a message from 2 days ago
        two_days_ago = datetime.now(UTC).replace(
            hour=12, minute=0, second=0, microsecond=0, tzinfo=None
        ) - timedelta(days=2)
        day_start = two_days_ago.replace(hour=0, minute=0, second=0, microsecond=0)

        _insert_message(
            penny,
            TEST_SENDER,
            "old message",
            PennyConstants.MessageDirection.INCOMING,
            two_days_ago,
        )

        # Mark preferences as already extracted for that day
        penny.db.preferences.add(
            user=TEST_SENDER,
            content="already extracted",
            valence="positive",
            source_period_start=day_start,
            source_period_end=day_start + timedelta(days=1),
        )

        await penny.history_agent.execute()

        # Verify no duplicate "already extracted" preference
        prefs = penny.db.preferences.get_for_user(TEST_SENDER)
        already_extracted = [p for p in prefs if p.content == "already extracted"]
        assert len(already_extracted) == 1


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
                source_period_start=datetime(2026, 3, 1),
                source_period_end=datetime(2026, 3, 2),
            ),
            Preference(
                user=TEST_SENDER,
                content="Country music",
                valence="negative",
                source_period_start=datetime(2026, 3, 1),
                source_period_end=datetime(2026, 3, 2),
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

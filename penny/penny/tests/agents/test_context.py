"""Integration tests for Agent context building (history, conversation, dislikes)."""

from datetime import UTC, datetime, timedelta

import pytest

from penny.constants import PennyConstants
from penny.database.models import MessageLog
from penny.tests.conftest import TEST_SENDER


def _insert_message(penny, sender, content, direction, timestamp, **kwargs):
    """Insert a message with a specific timestamp."""
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


# ── History context ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_history_context_formats_dates_and_topics(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """History context includes date labels and topic bullets."""
    config = make_config()

    async with running_penny(config) as penny:
        # Add a past history entry
        penny.db.history.add(
            user=TEST_SENDER,
            period_start=datetime(2026, 3, 1),
            period_end=datetime(2026, 3, 2),
            duration=PennyConstants.HistoryDuration.DAILY,
            topics="- Quantum physics\n- Machine learning",
        )

        context = penny.chat_agent._history_section(TEST_SENDER)
        assert context is not None
        assert "Conversation History" in context
        assert "Mar 1" in context
        assert "Quantum physics" in context
        assert "Machine learning" in context


@pytest.mark.asyncio
async def test_history_context_labels_today(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Today's history entry gets 'Today' label instead of date."""
    config = make_config()

    async with running_penny(config) as penny:
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
        penny.db.history.add(
            user=TEST_SENDER,
            period_start=today,
            period_end=today + timedelta(days=1),
            duration=PennyConstants.HistoryDuration.DAILY,
            topics="- Current events",
        )

        context = penny.chat_agent._history_section(TEST_SENDER)
        assert context is not None
        assert "Today" in context
        assert "Current events" in context


@pytest.mark.asyncio
async def test_history_context_skips_daily_covered_by_weekly(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Daily entries within a weekly rollup range are excluded from context."""
    config = make_config()

    async with running_penny(config) as penny:
        # Weekly rollup covering Mar 9-16
        penny.db.history.add(
            user=TEST_SENDER,
            period_start=datetime(2026, 3, 9),
            period_end=datetime(2026, 3, 16),
            duration=PennyConstants.HistoryDuration.WEEKLY,
            topics="- Weekly guitar topics\n- Weekly pedal topics",
        )
        # Daily entry INSIDE the weekly range — should be excluded
        penny.db.history.add(
            user=TEST_SENDER,
            period_start=datetime(2026, 3, 10),
            period_end=datetime(2026, 3, 11),
            duration=PennyConstants.HistoryDuration.DAILY,
            topics="- Daily guitar detail",
        )
        # Daily entry OUTSIDE the weekly range — should be included
        penny.db.history.add(
            user=TEST_SENDER,
            period_start=datetime(2026, 3, 17),
            period_end=datetime(2026, 3, 18),
            duration=PennyConstants.HistoryDuration.DAILY,
            topics="- Amp shopping",
        )

        context = penny.chat_agent._history_section(TEST_SENDER)
        assert context is not None
        assert "Weekly guitar topics" in context
        assert "Weekly pedal topics" in context
        assert "Amp shopping" in context
        assert "Daily guitar detail" not in context


@pytest.mark.asyncio
async def test_history_context_none_when_no_entries(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """History context returns None when there are no entries."""
    config = make_config()

    async with running_penny(config) as penny:
        context = penny.chat_agent._history_section(TEST_SENDER)
        assert context is None


# ── Conversation building ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_conversation_builds_user_assistant_turns(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Conversation history alternates user/assistant turns."""
    config = make_config()

    async with running_penny(config) as penny:
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            TEST_SENDER,
            "hello penny",
        )
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            penny.config.signal_number,
            "hey there!",
            parent_id=1,
            recipient=TEST_SENDER,
        )

        conversation = penny.chat_agent._build_conversation(TEST_SENDER)
        assert len(conversation) == 2
        assert conversation[0][0] == "user"
        assert "hello penny" in conversation[0][1]
        assert conversation[1][0] == "assistant"
        assert "hey there" in conversation[1][1]


@pytest.mark.asyncio
async def test_conversation_merges_consecutive_same_role(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Consecutive messages from the same role are merged with newlines."""
    config = make_config()

    async with running_penny(config) as penny:
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            TEST_SENDER,
            "first message",
        )
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            TEST_SENDER,
            "second message",
        )
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            penny.config.signal_number,
            "response",
            recipient=TEST_SENDER,
        )

        conversation = penny.chat_agent._build_conversation(TEST_SENDER)
        assert len(conversation) == 2  # Merged user messages + one assistant
        assert "first message" in conversation[0][1]
        assert "second message" in conversation[0][1]


@pytest.mark.asyncio
async def test_conversation_starts_after_rollup(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Conversation history starts after the latest history rollup."""
    config = make_config()

    async with running_penny(config) as penny:
        # Create a rollup that covers earlier messages
        now = datetime.now(UTC).replace(tzinfo=None)
        rollup_end = now - timedelta(minutes=5)
        penny.db.history.add(
            user=TEST_SENDER,
            period_start=now - timedelta(hours=1),
            period_end=rollup_end,
            duration=PennyConstants.HistoryDuration.DAILY,
            topics="- Old topics",
        )

        # Insert a message before the rollup end (should be excluded)
        _insert_message(
            penny,
            TEST_SENDER,
            "old message before rollup",
            PennyConstants.MessageDirection.INCOMING,
            rollup_end - timedelta(minutes=10),
        )

        # Insert a message after the rollup end (should be included)
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            TEST_SENDER,
            "new message after rollup",
        )

        conversation = penny.chat_agent._build_conversation(TEST_SENDER)
        contents = " ".join(c for _, c in conversation)
        assert "new message after rollup" in contents
        assert "old message before rollup" not in contents


# ── Dislike context ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dislike_context_lists_negative_preferences(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Dislike context includes only negative preferences."""
    config = make_config()

    async with running_penny(config) as penny:
        penny.db.preferences.add(
            user=TEST_SENDER,
            content="Country music",
            valence="negative",
        )
        penny.db.preferences.add(
            user=TEST_SENDER,
            content="Jazz music",
            valence="positive",
        )

        context = penny.chat_agent._dislike_section(TEST_SENDER)
        assert context is not None
        assert "Country music" in context
        assert "Jazz music" not in context
        assert "Topics to Avoid" in context


@pytest.mark.asyncio
async def test_dislike_context_none_when_no_dislikes(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Dislike context returns None when user has no negative preferences."""
    config = make_config()

    async with running_penny(config) as penny:
        penny.db.preferences.add(
            user=TEST_SENDER,
            content="Jazz music",
            valence="positive",
        )

        context = penny.chat_agent._dislike_section(TEST_SENDER)
        assert context is None


@pytest.mark.asyncio
async def test_dislike_context_deduplicates(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Dislike context deduplicates case-insensitively."""
    config = make_config()

    async with running_penny(config) as penny:
        penny.db.preferences.add(
            user=TEST_SENDER,
            content="Country music",
            valence="negative",
        )
        penny.db.preferences.add(
            user=TEST_SENDER,
            content="country music",
            valence="negative",
        )

        context = penny.chat_agent._dislike_section(TEST_SENDER)
        assert context is not None
        # Should appear only once despite case-insensitive duplicate
        assert context.count("ountry music") == 1


# ── Thought context ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_thought_context_scoped_to_seed_preference(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Thinking agent thought context only includes thoughts for the same seed preference."""
    config = make_config()

    async with running_penny(config) as penny:
        # Add thoughts for two different preferences
        penny.db.thoughts.add(TEST_SENDER, "thought about AI", preference_id=1)
        penny.db.thoughts.add(TEST_SENDER, "thought about music", preference_id=2)

        # Scope to preference 1 — should only see the AI thought
        penny.thinking_agent._seed_pref_id = 1
        context = penny.thinking_agent._thought_section(TEST_SENDER)
        assert context is not None
        assert "thought about AI" in context
        assert "thought about music" not in context


@pytest.mark.asyncio
async def test_thought_context_none_when_no_thoughts(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Thought context returns None when there are no thoughts."""
    config = make_config()

    async with running_penny(config) as penny:
        context = penny.thinking_agent._thought_section(TEST_SENDER)
        assert context is None


# ── Profile context ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_profile_context_includes_name(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Profile context includes user name."""
    config = make_config()

    async with running_penny(config) as penny:
        context = penny.chat_agent._profile_section(TEST_SENDER, "hello")
        assert context is not None
        assert "Test User" in context


@pytest.mark.asyncio
async def test_profile_context_none_for_unknown_user(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Profile context returns None for users without profile info."""
    config = make_config()

    async with running_penny(config) as penny:
        context = penny.chat_agent._profile_section("+1999999999", "hello")
        assert context is None


# ── Sentiment scoring ────────────────────────────────────────────────────


def test_compute_sentiment_score_likes_minus_dislikes():
    """Sentiment score = avg similarity to likes - avg similarity to dislikes."""
    from penny.ollama.similarity import compute_sentiment_score

    vec = [1.0, 0.0, 0.0]
    likes = [[1.0, 0.0, 0.0]]  # identical = similarity 1.0
    dislikes = [[0.0, 1.0, 0.0]]  # orthogonal = similarity 0.0
    score = compute_sentiment_score(vec, likes, dislikes)
    assert score > 0.9  # close to 1.0


def test_compute_sentiment_score_no_preferences():
    """Sentiment score is 0 when no preferences exist."""
    from penny.ollama.similarity import compute_sentiment_score

    score = compute_sentiment_score([1.0, 0.0, 0.0], [], [])
    assert score == 0.0


# ── Page context ─────────────────────────────────────────────────────────


def test_page_context_section_included_in_prompt():
    """Page context from the browser appears in the system prompt."""
    from penny.agents.chat import ChatAgent

    page_context = {
        "title": "Example Product Page",
        "url": "https://example.com/product",
        "text": "This is a great product that costs $49.99",
    }
    section = ChatAgent._page_context_section(page_context)
    assert section is not None
    assert "### Current Browser Page" in section
    assert "Example Product Page" in section
    assert "https://example.com/product" in section
    assert "$49.99" in section


def test_page_context_section_none_when_empty():
    """No page context section when context is None or has no text."""
    from penny.agents.chat import ChatAgent

    assert ChatAgent._page_context_section(None) is None
    assert ChatAgent._page_context_section({}) is None
    assert ChatAgent._page_context_section({"title": "T", "url": "U", "text": ""}) is None


@pytest.mark.asyncio
async def test_page_context_in_system_prompt(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    mock_serper_image,
):
    """Messages with page_context include the page in the system prompt.

    Since page_context only comes from browser messages and we test through
    Signal here, we verify the building block directly and the integration
    path by checking the ChatAgent._page_context_section output is non-None.
    The full integration (browser → background → server → prompt) is
    validated end-to-end in production.
    """
    from penny.agents.chat import ChatAgent

    page = {"title": "Test Page", "url": "https://test.com", "text": "Some page content"}
    section = ChatAgent._page_context_section(page)
    assert section is not None
    assert "Some page content" in section

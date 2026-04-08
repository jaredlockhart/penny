"""Integration tests for Agent context building (conversation, dislikes, knowledge)."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from penny.constants import PennyConstants
from penny.database.models import MessageLog
from penny.llm.embeddings import serialize_embedding
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


# ── Conversation building ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_conversation_builds_user_assistant_turns(
    signal_server, mock_llm, make_config, test_user_info, running_penny
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
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Consecutive messages from the same role are merged with newlines.

    Proactive outgoing messages (no parent_id) are excluded from context —
    only direct replies (parent_id set) are included alongside user messages.
    """
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
        # Direct reply (parent_id set) — included in context
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            penny.config.signal_number,
            "response",
            parent_id=2,
            recipient=TEST_SENDER,
        )
        # Proactive notification (no parent_id) — excluded from context
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            penny.config.signal_number,
            "proactive thought",
            recipient=TEST_SENDER,
        )

        conversation = penny.chat_agent._build_conversation(TEST_SENDER)
        contents = " ".join(c for _, c in conversation)
        assert len(conversation) == 2  # Merged user messages + one assistant reply
        assert "first message" in conversation[0][1]
        assert "second message" in conversation[0][1]
        assert "response" in contents
        assert "proactive thought" not in contents


# ── Dislike context ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dislike_context_lists_negative_preferences(
    signal_server, mock_llm, make_config, test_user_info, running_penny
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
    signal_server, mock_llm, make_config, test_user_info, running_penny
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
    signal_server, mock_llm, make_config, test_user_info, running_penny
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
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Thinking agent thought context only includes thoughts for the same seed preference."""
    config = make_config()

    async with running_penny(config) as penny:
        # Add thoughts for two different preferences
        penny.db.thoughts.add(TEST_SENDER, "thought about AI", preference_id=1, title="AI advances")
        penny.db.thoughts.add(
            TEST_SENDER, "thought about music", preference_id=2, title="Music theory"
        )

        # Scope to preference 1 — should only see the AI thought title
        penny.thinking_agent._seed_pref_id = 1
        context = penny.thinking_agent._thought_section(TEST_SENDER)
        assert context is not None
        assert "AI advances" in context
        assert "Music theory" not in context


@pytest.mark.asyncio
async def test_thought_context_none_when_no_thoughts(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Thought context returns None when there are no thoughts."""
    config = make_config()

    async with running_penny(config) as penny:
        context = penny.thinking_agent._thought_section(TEST_SENDER)
        assert context is None


# ── Profile context ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_profile_context_includes_name(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Profile context includes user name."""
    config = make_config()

    async with running_penny(config) as penny:
        context = penny.chat_agent._profile_section(TEST_SENDER)
        assert context is not None
        assert "Test User" in context


@pytest.mark.asyncio
async def test_profile_context_none_for_unknown_user(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Profile context returns None for users without profile info."""
    config = make_config()

    async with running_penny(config) as penny:
        context = penny.chat_agent._profile_section("+1999999999")
        assert context is None


# ── Sentiment scoring ────────────────────────────────────────────────────


def _make_pref(vec: list[float], valence: str, mention_count: int = 2):
    """Build a minimal preference-like object for sentiment scoring tests."""
    from types import SimpleNamespace

    from penny.llm.embeddings import serialize_embedding

    return SimpleNamespace(
        embedding=serialize_embedding(vec),
        valence=valence,
        mention_count=mention_count,
    )


def test_compute_mention_weighted_sentiment_likes_minus_dislikes():
    """Score = weighted avg similarity to likes - weighted avg similarity to dislikes."""
    from penny.llm.similarity import compute_mention_weighted_sentiment

    vec = [1.0, 0.0, 0.0]
    prefs = [
        _make_pref([1.0, 0.0, 0.0], "positive"),  # identical = similarity 1.0
        _make_pref([0.0, 1.0, 0.0], "negative"),  # orthogonal = similarity 0.0
    ]
    score = compute_mention_weighted_sentiment(vec, prefs, min_mentions=2)
    assert score > 0.9  # close to 1.0


def test_compute_mention_weighted_sentiment_no_preferences():
    """Score is 0 when no qualifying preferences exist."""
    from penny.llm.similarity import compute_mention_weighted_sentiment

    score = compute_mention_weighted_sentiment([1.0, 0.0, 0.0], [], min_mentions=2)
    assert score == 0.0


# ── Page context ─────────────────────────────────────────────────────────


def test_page_context_injected_as_synthetic_tool_call():
    """Page context is injected as a search tool call + result in messages."""
    from penny.agents.chat import ChatAgent
    from penny.channels.base import PageContext

    page_context = PageContext(
        title="Example Product Page",
        url="https://example.com/product",
        text="This is a great product that costs $49.99",
    )
    messages: list[dict] = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "what is this page?"},
    ]
    ChatAgent._inject_page_context(messages, page_context)

    assert len(messages) == 4
    # Assistant tool call uses BrowseTool format (name="browse", URL in queries)
    assert messages[2]["role"] == "assistant"
    assert messages[2]["tool_calls"][0]["function"]["name"] == "browse"
    assert messages[2]["tool_calls"][0]["function"]["arguments"]["queries"] == [
        "https://example.com/product"
    ]
    # Tool result
    assert messages[3]["role"] == "tool"
    assert messages[3]["tool_name"] == "browse"
    assert "$49.99" in messages[3]["content"]
    assert "Example Product Page" in messages[3]["content"]


def test_page_context_not_injected_when_empty():
    """No injection when page context has no text."""
    from penny.agents.chat import ChatAgent
    from penny.channels.base import PageContext

    messages: list[dict] = [{"role": "user", "content": "hi"}]
    ChatAgent._inject_page_context(messages, PageContext(title="T", url="U", text=""))
    assert len(messages) == 1  # unchanged


def test_page_hint_in_system_prompt():
    """System prompt includes a minimal page hint with title and URL."""
    from penny.agents.chat import ChatAgent
    from penny.channels.base import PageContext

    context = PageContext(title="Cool Article", url="https://example.com/article", text="content")
    agent = ChatAgent.__new__(ChatAgent)
    agent._pending_page_context = context
    hint = agent._page_hint_section()
    assert hint is not None
    assert "Cool Article" in hint
    assert "https://example.com/article" in hint
    # Should NOT contain the full text — that's in the tool result
    assert "content" not in hint


# ── Related messages context ────────────────────────────────────────────


# Deterministic test vectors: pedal_vec is similar to query_vec, space_vec is orthogonal
_QUERY_VEC = [1.0, 0.0, 0.0]
_PEDAL_VEC = [0.9, 0.1, 0.0]  # cosine ~0.994
_SPACE_VEC = [0.0, 1.0, 0.0]  # cosine ~0.0 to query vec (orthogonal)


def _seed_past_messages(penny, timestamp_offset_days: int = 3):
    """Insert past messages with embeddings for related message tests.

    Inserts two messages: a pedal message (older) and a space message (newer).
    The pedal message is similar to _QUERY_VEC, the space message is orthogonal.
    """
    past = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=timestamp_offset_days)
    _insert_message(
        penny,
        TEST_SENDER,
        "i ordered a tone king royalist and tubesteader eggnog pedal",
        PennyConstants.MessageDirection.INCOMING,
        past,
        embedding=serialize_embedding(_PEDAL_VEC),
    )
    _insert_message(
        penny,
        TEST_SENDER,
        "tell me about black holes and neutron stars",
        PennyConstants.MessageDirection.INCOMING,
        past + timedelta(hours=1),
        embedding=serialize_embedding(_SPACE_VEC),
    )


@pytest.mark.asyncio
async def test_related_messages_retrieves_similar_past_messages(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Related messages section includes semantically similar past messages."""
    config = make_config(MESSAGE_CONTEXT_LIMIT=0)

    async with running_penny(config) as penny:
        _seed_past_messages(penny)

        mock_client = AsyncMock()
        # Return query vec for each text (just the current content since context limit is 0)
        mock_client.embed = AsyncMock(side_effect=lambda texts: [_QUERY_VEC] * len(texts))
        penny.chat_agent._embedding_model_client = mock_client

        embeddings = await penny.chat_agent._embed_conversation(
            TEST_SENDER, "the pedals finally arrived"
        )
        context = await penny.chat_agent._related_messages_section(TEST_SENDER, embeddings)
        assert context is not None
        assert "Related Past Messages" in context
        # Full message content shown (no truncation)
        assert "i ordered a tone king royalist and tubesteader eggnog pedal" in context
        # Results sorted by date (pedal message is older, should appear first)
        lines = context.strip().split("\n")
        message_lines = [line for line in lines if line.startswith(("Mar", "Apr"))]
        assert len(message_lines) >= 1
        # Pedal message should appear before the space message (chronological order)
        royalist_pos = context.index("royalist")
        if "black holes" in context:
            space_pos = context.index("black holes")
            assert royalist_pos < space_pos


@pytest.mark.asyncio
async def test_related_messages_excludes_current_conversation(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Messages within the current conversation window are excluded."""
    config = make_config()

    async with running_penny(config) as penny:
        # Insert a message in the current conversation window (after rollup/midnight)
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            TEST_SENDER,
            "recent pedal message in current session",
        )
        # Manually set its embedding
        penny.db.messages.update_embedding(1, serialize_embedding(_PEDAL_VEC))

        mock_client = AsyncMock()
        mock_client.embed = AsyncMock(side_effect=lambda texts: [_QUERY_VEC] * len(texts))
        penny.chat_agent._embedding_model_client = mock_client

        embeddings = await penny.chat_agent._embed_conversation(
            TEST_SENDER, "the pedals finally arrived"
        )
        context = await penny.chat_agent._related_messages_section(TEST_SENDER, embeddings)
        # Current session message should be excluded (in conversation window), so no results
        assert context is None


@pytest.mark.asyncio
async def test_related_messages_none_without_embedding_client(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Related messages section returns None when no embedding client is configured."""
    config = make_config()

    async with running_penny(config) as penny:
        _seed_past_messages(penny)

        penny.chat_agent._embedding_model_client = None

        # No embedding client → None embeddings → no related messages
        context = await penny.chat_agent._related_messages_section(TEST_SENDER, None)
        assert context is None


@pytest.mark.asyncio
async def test_related_messages_in_chat_system_prompt(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Related messages section appears in ChatAgent system prompt."""
    config = make_config(MESSAGE_CONTEXT_LIMIT=0)

    async with running_penny(config) as penny:
        _seed_past_messages(penny)

        mock_client = AsyncMock()
        mock_client.embed = AsyncMock(side_effect=lambda texts: [_QUERY_VEC] * len(texts))
        penny.chat_agent._embedding_model_client = mock_client
        penny.chat_agent._pending_page_context = None

        prompt = await penny.chat_agent._build_system_prompt(
            TEST_SENDER, content="the pedals finally arrived"
        )
        assert "Related Past Messages" in prompt
        assert "royalist" in prompt


@pytest.mark.asyncio
async def test_related_messages_date_ordering_with_multiple_dates(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Related messages are sorted chronologically after similarity selection."""
    config = make_config(MESSAGE_CONTEXT_LIMIT=0)

    async with running_penny(config) as penny:
        # Insert 3 messages on different days — all similar to query
        base = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=10)
        similar_vec = [0.95, 0.05, 0.0]
        _insert_message(
            penny,
            TEST_SENDER,
            "third message about pedals",
            PennyConstants.MessageDirection.INCOMING,
            base + timedelta(days=2),
            embedding=serialize_embedding(similar_vec),
        )
        _insert_message(
            penny,
            TEST_SENDER,
            "first message about pedals",
            PennyConstants.MessageDirection.INCOMING,
            base,
            embedding=serialize_embedding(similar_vec),
        )
        _insert_message(
            penny,
            TEST_SENDER,
            "second message about pedals",
            PennyConstants.MessageDirection.INCOMING,
            base + timedelta(days=1),
            embedding=serialize_embedding(similar_vec),
        )

        mock_client = AsyncMock()
        mock_client.embed = AsyncMock(side_effect=lambda texts: [_QUERY_VEC] * len(texts))
        penny.chat_agent._embedding_model_client = mock_client

        embeddings = await penny.chat_agent._embed_conversation(TEST_SENDER, "pedals")
        context = await penny.chat_agent._related_messages_section(TEST_SENDER, embeddings)
        assert context is not None
        # Messages should appear in chronological order regardless of insertion order
        first_pos = context.index("first message")
        second_pos = context.index("second message")
        third_pos = context.index("third message")
        assert first_pos < second_pos < third_pos


# ── Message store embedding methods ──────────────────────────────────────


@pytest.mark.asyncio
async def test_get_incoming_without_embeddings(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """get_incoming_without_embeddings returns incoming messages lacking embeddings."""
    config = make_config()

    async with running_penny(config) as penny:
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            TEST_SENDER,
            "no embedding yet",
        )
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            penny.config.signal_number,
            "outgoing message",
            parent_id=1,
            recipient=TEST_SENDER,
        )
        # Reaction should be excluded
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            TEST_SENDER,
            "reaction",
            is_reaction=True,
        )

        results = penny.db.messages.get_incoming_without_embeddings()
        assert len(results) == 1
        assert results[0].content == "no embedding yet"


@pytest.mark.asyncio
async def test_get_incoming_with_embeddings(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """get_incoming_with_embeddings returns only incoming messages that have embeddings."""
    config = make_config()

    async with running_penny(config) as penny:
        # Incoming with embedding
        msg_id = penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            TEST_SENDER,
            "has embedding",
        )
        penny.db.messages.update_embedding(msg_id, serialize_embedding(_PEDAL_VEC))

        # Incoming without embedding — should be excluded
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            TEST_SENDER,
            "no embedding",
        )

        # Outgoing with embedding — should be excluded
        out_id = penny.db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            penny.config.signal_number,
            "outgoing",
            parent_id=1,
            recipient=TEST_SENDER,
        )
        penny.db.messages.update_embedding(out_id, serialize_embedding(_PEDAL_VEC))

        results = penny.db.messages.get_incoming_with_embeddings(TEST_SENDER)
        assert len(results) == 1
        assert results[0].content == "has embedding"


# ── Incoming message embedding on arrival ────────────────────────────────


@pytest.mark.asyncio
async def test_incoming_message_embedded_on_arrival(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Incoming messages get embedded when an embedding client is available."""
    config = make_config(llm_embedding_model="test-embedding")
    mock_llm.set_default_flow(final_response="got it! 🎸")

    async with running_penny(config) as penny:
        mock_embed_client = AsyncMock()
        mock_embed_client.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        # Set on the concrete signal channel inside the manager
        for channel in penny.channel._channels.values():
            channel._embedding_model_client = mock_embed_client

        await signal_server.push_message(sender=TEST_SENDER, content="test embedding")
        await signal_server.wait_for_message(timeout=10.0)

        # The incoming message should have been embedded
        messages = penny.db.messages.get_incoming_with_embeddings(TEST_SENDER)
        assert len(messages) >= 1
        assert any("test embedding" in m.content for m in messages)


# ── Startup backfill ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_backfill_incoming_message_embeddings(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Startup backfill embeds incoming messages that lack embeddings."""
    config = make_config()

    async with running_penny(config) as penny:
        # Insert incoming messages without embeddings
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            TEST_SENDER,
            "needs embedding",
        )
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            TEST_SENDER,
            "also needs embedding",
        )
        assert len(penny.db.messages.get_incoming_without_embeddings()) == 2

        # Mock the embedding client and run backfill
        mock_embed_client = AsyncMock()
        mock_embed_client.embed = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        penny.embedding_model_client = mock_embed_client

        count = await penny._backfill_incoming_message_embeddings(batch_limit=50)
        assert count == 2
        assert len(penny.db.messages.get_incoming_without_embeddings()) == 0


# ── Knowledge context ───────────────────────────────────────────────────


def _seed_knowledge(penny):
    """Insert knowledge entries with deterministic embeddings for tests."""
    penny.db.knowledge.upsert_by_url(
        url="https://tubesteader.com/products/eggnog",
        title="TubeSteader Eggnog",
        summary="The Eggnog is a single-channel tube overdrive pedal using a 12AX7 tube.",
        embedding=serialize_embedding(_PEDAL_VEC),
        source_prompt_id=1,
    )
    penny.db.knowledge.upsert_by_url(
        url="https://sci.news/astronomy/d9-binary-star",
        title="Binary Star D9",
        summary="A binary star system designated D9 orbits Sagittarius A*.",
        embedding=serialize_embedding(_SPACE_VEC),
        source_prompt_id=2,
    )


@pytest.mark.asyncio
async def test_knowledge_section_returns_matching_entries(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Knowledge section includes entries relevant to the conversation."""
    config = make_config()

    async with running_penny(config) as penny:
        _seed_knowledge(penny)

        mock_client = AsyncMock()
        # Return pedal-like vector for the conversation embedding
        mock_client.embed = AsyncMock(side_effect=lambda texts: [_PEDAL_VEC] * len(texts))
        penny.chat_agent._embedding_model_client = mock_client

        embeddings = await penny.chat_agent._embed_conversation(
            TEST_SENDER, "tell me about the eggnog pedal"
        )
        context = await penny.chat_agent._related_knowledge_section(embeddings)
        assert context is not None
        assert "### Knowledge" in context
        assert "TubeSteader Eggnog" in context
        assert "tubesteader.com" in context
        assert "tube overdrive" in context


@pytest.mark.asyncio
async def test_knowledge_weighted_scoring_favors_conversation_context(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Weighted scoring uses conversation context, not just the last message."""
    config = make_config()

    async with running_penny(config) as penny:
        _seed_knowledge(penny)

        # Seed conversation history: earlier messages about pedals
        past = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=5)
        _insert_message(
            penny,
            TEST_SENDER,
            "i love my tubesteader pedals",
            PennyConstants.MessageDirection.INCOMING,
            past,
        )
        _insert_message(
            penny,
            penny.config.signal_number,
            "nice! which ones?",
            PennyConstants.MessageDirection.OUTGOING,
            past + timedelta(minutes=1),
            recipient=TEST_SENDER,
        )

        _insert_message(
            penny,
            TEST_SENDER,
            "the eggnog sounds amazing",
            PennyConstants.MessageDirection.INCOMING,
            past + timedelta(minutes=2),
        )
        _insert_message(
            penny,
            penny.config.signal_number,
            "yeah the tweed tone is great",
            PennyConstants.MessageDirection.OUTGOING,
            past + timedelta(minutes=3),
            recipient=TEST_SENDER,
        )

        mock_client = AsyncMock()
        # 4 conversation messages + 1 new content = 5 embeddings
        # First 4 are pedal-like, last (new content) is a neutral/vague vector
        # that doesn't strongly match either knowledge entry
        neutral_vec = [0.5, 0.5, 0.0]  # roughly equidistant from pedal and space
        mock_client.embed = AsyncMock(
            return_value=[_PEDAL_VEC, _PEDAL_VEC, _PEDAL_VEC, _PEDAL_VEC, neutral_vec]
        )
        penny.chat_agent._embedding_model_client = mock_client

        # Last message is vague but 4 prior conversation messages are about pedals
        embeddings = await penny.chat_agent._embed_conversation(TEST_SENDER, "yeah they're great")
        context = await penny.chat_agent._related_knowledge_section(embeddings)
        assert context is not None
        # Pedal knowledge should rank higher due to conversation context
        eggnog_pos = context.find("TubeSteader Eggnog")
        star_pos = context.find("Binary Star D9")
        assert eggnog_pos >= 0
        assert eggnog_pos < star_pos


@pytest.mark.asyncio
async def test_knowledge_none_without_embedding_client(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Knowledge section returns None when no embedding client is configured."""
    config = make_config()

    async with running_penny(config) as penny:
        _seed_knowledge(penny)
        penny.chat_agent._embedding_model_client = None

        # No embedding client → _embed_conversation returns None → no knowledge
        embeddings = await penny.chat_agent._embed_conversation(TEST_SENDER, "tell me about pedals")
        assert embeddings is None
        context = await penny.chat_agent._related_knowledge_section(embeddings)
        assert context is None


@pytest.mark.asyncio
async def test_knowledge_in_chat_system_prompt(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Knowledge section appears in ChatAgent system prompt."""
    config = make_config()

    async with running_penny(config) as penny:
        _seed_knowledge(penny)

        mock_client = AsyncMock()
        mock_client.embed = AsyncMock(return_value=[_PEDAL_VEC])
        penny.chat_agent._embedding_model_client = mock_client
        penny.chat_agent._pending_page_context = None

        prompt = await penny.chat_agent._build_system_prompt(
            TEST_SENDER, content="the eggnog pedal"
        )
        assert "### Knowledge" in prompt
        assert "TubeSteader Eggnog" in prompt
        # Knowledge appears before Related Past Messages (or Instructions if no messages)
        knowledge_pos = prompt.find("### Knowledge")
        instructions_pos = prompt.find("## Instructions")
        assert knowledge_pos < instructions_pos

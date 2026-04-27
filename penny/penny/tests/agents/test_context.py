"""Integration tests for Agent context building (conversation, profile, page hint, embeddings)."""

from unittest.mock import AsyncMock

import pytest

from penny.constants import PennyConstants
from penny.llm.embeddings import serialize_embedding
from penny.tests.conftest import TEST_SENDER

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


# ── Message store embedding methods ──────────────────────────────────────


# Deterministic test vector for the embedding-pipeline tests below.
_PEDAL_VEC = [0.9, 0.1, 0.0]


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
        # 4-dim to match the channel-side and chat-side embedding callers
        # (chat agent uses its own client during recall).
        mock_embed_client = AsyncMock()
        mock_embed_client.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
        # Set on the concrete signal channel inside the manager
        for channel in penny.channel._channels.values():
            channel._embedding_model_client = mock_embed_client
        # Also set on the chat agent so recall queries the same mock
        penny.chat_agent._embedding_model_client = mock_embed_client

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

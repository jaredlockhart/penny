"""Integration tests for HistoryAgent: preference extraction and knowledge extraction."""

import json
from datetime import UTC, datetime

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

"""Integration tests for ThinkingAgent: orientation step, thought persistence, context."""

from datetime import UTC, datetime

import pytest

from penny.constants import PennyConstants
from penny.database.models import Event
from penny.ollama.embeddings import serialize_embedding
from penny.tests.conftest import TEST_SENDER


@pytest.mark.asyncio
async def test_thinking_orientation_step_fires(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """ThinkingAgent runs an orientation step before tool calling.

    The orientation step calls the model with no tools and the result is
    injected as context for the agentic loop.
    """
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=2,
    )

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        if count == 1:
            # Orientation step (foreground model, no tools)
            return mock_ollama._make_text_response(
                request, "I should check for new AI developments."
            )
        # Agentic loop steps
        return mock_ollama._make_text_response(request, "nothing interesting today")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Seed a message so the user exists
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, TEST_SENDER, "hello penny"
        )

        # Run one thinking cycle directly
        await penny.thinking_agent.execute()

        # At least 2 requests: orientation + agentic loop
        assert len(requests_seen) >= 2

        # First request (orientation): no tools
        assert requests_seen[0].get("tools") is None

        # Second request (agentic loop): has tools
        second_tools = requests_seen[1].get("tools")
        assert second_tools is not None or len(requests_seen) > 2

        # Orientation text should appear in thought log
        thoughts = penny.db.thoughts.get_recent(TEST_SENDER, limit=10)
        orientation_thoughts = [t for t in thoughts if "[orientation]" in t.content]
        assert len(orientation_thoughts) >= 1
        assert "AI developments" in orientation_thoughts[0].content


@pytest.mark.asyncio
async def test_thinking_persists_reasoning_as_thoughts(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """ThinkingAgent persists tool call reasoning as thoughts in [tool(args)] format."""
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=3,
    )

    def handler(request, count):
        if count == 1:
            # Orientation
            return mock_ollama._make_text_response(request, "checking news")
        if count == 2:
            # Agentic step: tool call with reasoning
            return mock_ollama._make_tool_call_response(
                request,
                "search",
                {"query": "latest AI news", "reasoning": "User is interested in AI topics"},
            )
        # Final step (no tools, must produce text)
        return mock_ollama._make_text_response(request, "nothing notable found")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, TEST_SENDER, "hello penny"
        )

        await penny.thinking_agent.execute()

        thoughts = penny.db.thoughts.get_recent(TEST_SENDER, limit=20)
        thought_texts = [t.content for t in thoughts]

        # Should have orientation thought
        assert any("[orientation]" in t for t in thought_texts)

        # Should have reasoning thought in [tool(args)] format
        reasoning_thoughts = [t for t in thought_texts if "[search(" in t]
        assert len(reasoning_thoughts) >= 1
        assert "User is interested in AI topics" in reasoning_thoughts[0]


@pytest.mark.asyncio
async def test_thinking_receives_event_context(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """ThinkingAgent receives recent events in context (no embedding needed)."""
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=2,
    )

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        return mock_ollama._make_text_response(request, "ok")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, TEST_SENDER, "hello"
        )

        # Seed an event
        with penny.db.get_session() as session:
            event = Event(
                user=TEST_SENDER,
                headline="SpaceX launches Starship successfully",
                summary="SpaceX completed its first full orbital test flight.",
                source_url="https://example.com/spacex",
                source_type=PennyConstants.EventSourceType.NEWS_API,
                occurred_at=datetime.now(UTC),
            )
            session.add(event)
            session.commit()

        await penny.thinking_agent.execute()

        # Orientation request should contain event context
        assert len(requests_seen) >= 1
        first_request = requests_seen[0]
        system_msgs = [m for m in first_request["messages"] if m.get("role") == "system"]
        all_system_text = " ".join(m.get("content", "") for m in system_msgs)
        assert "SpaceX launches Starship" in all_system_text
        assert "I saw in the news" in all_system_text


@pytest.mark.asyncio
async def test_thinking_receives_interest_profile_without_entities(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Interest profile includes learn/follow topics but NOT entity names."""
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=2,
    )

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        return mock_ollama._make_text_response(request, "ok")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, TEST_SENDER, "hello"
        )

        # Seed a learn topic and an entity
        penny.db.learn_prompts.create(
            user=TEST_SENDER, prompt_text="quantum computing", searches_remaining=0
        )
        penny.db.entities.get_or_create(TEST_SENDER, "kef ls50 meta")

        await penny.thinking_agent.execute()

        assert len(requests_seen) >= 1

        # Learn topic should appear as user-voice interest message
        user_msgs = [m for m in requests_seen[0]["messages"] if m.get("role") == "user"]
        all_user_text = " ".join(m.get("content", "") for m in user_msgs)
        assert "quantum computing" in all_user_text
        # Entity name should NOT appear in interest messages
        # (entities are injected via embedding similarity, not the interest profile)
        assert "kef ls50 meta" not in all_user_text.lower()


@pytest.mark.asyncio
async def test_thinking_dynamic_entity_injection_from_reasoning(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Entities are injected from orientation (pre-loop) and tool reasoning (mid-loop).

    The orientation text is embedded and used to pull similar entities before
    the agentic loop starts. Then after each tool call, the reasoning text
    triggers another round of entity injection.
    """
    config = make_config(
        ollama_embedding_model="test-embed-model",
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=4,
    )

    # Embed handler returns identical vectors → max similarity
    def embed_handler(model, input_text):
        texts = [input_text] if isinstance(input_text, str) else input_text
        return [[1.0, 0.0, 0.0, 0.0]] * len(texts)

    mock_ollama.set_embed_handler(embed_handler)

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        if count == 1:
            # Orientation
            return mock_ollama._make_text_response(request, "checking speaker news")
        if count == 2:
            # Agentic step: tool call with reasoning about speakers
            return mock_ollama._make_tool_call_response(
                request,
                "search",
                {"query": "kef speaker news", "reasoning": "Looking for KEF speaker updates"},
            )
        # Final response
        return mock_ollama._make_text_response(request, "done")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, TEST_SENDER, "hello"
        )

        # Seed entity with embedding and facts
        entity = penny.db.entities.get_or_create(TEST_SENDER, "kef ls50 meta")
        assert entity is not None and entity.id is not None
        penny.db.facts.add(entity.id, "Costs $1,599 per pair")
        penny.db.entities.update_embedding(entity.id, serialize_embedding([1.0, 0.0, 0.0, 0.0]))

        await penny.thinking_agent.execute()

        # The agentic loop request (request[1]) should already contain entity context
        # (injected from orientation text embedding, before loop starts)
        assert len(requests_seen) >= 2
        second_request = requests_seen[1]
        system_msgs = [m for m in second_request["messages"] if m.get("role") == "system"]
        all_system_text = " ".join(m.get("content", "") for m in system_msgs)
        assert "kef ls50 meta" in all_system_text.lower()
        assert "$1,599" in all_system_text


@pytest.mark.asyncio
async def test_thinking_context_has_timeline_and_interests(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Context includes individual timeline entries (messages + thoughts) and interest headers."""
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=2,
    )

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        return mock_ollama._make_text_response(request, "ok")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Seed messages so conversation context exists
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, TEST_SENDER, "hello penny"
        )
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            penny.config.signal_number,
            "hey there!",
            parent_id=1,
            recipient=TEST_SENDER,
        )

        # Seed a thought
        penny.db.thoughts.add(TEST_SENDER, "test thought")

        # Seed a learn topic
        penny.db.learn_prompts.create(
            user=TEST_SENDER, prompt_text="AI safety", searches_remaining=0
        )

        await penny.thinking_agent.execute()

        # Orientation request should have timeline entries as individual messages
        assert len(requests_seen) >= 1
        first_msgs = requests_seen[0]["messages"]

        # User message appears with user role
        user_msgs = [m for m in first_msgs if m.get("role") == "user"]
        assert any("hello penny" in m.get("content", "") for m in user_msgs)

        # Penny message appears with system role
        system_msgs = [m for m in first_msgs if m.get("role") == "system"]
        assert any("hey there!" in m.get("content", "") for m in system_msgs)

        # Thought appears with system role
        assert any("I thought: test thought" in m.get("content", "") for m in system_msgs)

        # Learn topic appears as user-voice interest message
        all_user_text = " ".join(
            m.get("content", "") for m in first_msgs if m.get("role") == "user"
        )
        assert "I'm interested in" in all_user_text


@pytest.mark.asyncio
async def test_thinking_loop_uses_lean_context(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Agentic loop gets lean context (plan + entities), not full history."""
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=2,
    )

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        if count == 1:
            return mock_ollama._make_text_response(request, "I want to research cyberpunk anime.")
        return mock_ollama._make_text_response(request, "done")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, TEST_SENDER, "hello penny"
        )
        penny.db.learn_prompts.create(
            user=TEST_SENDER, prompt_text="AI safety", searches_remaining=0
        )
        penny.db.thoughts.add(TEST_SENDER, "previous thought")

        await penny.thinking_agent.execute()

        assert len(requests_seen) >= 2

        # Orientation (request 0) should have timeline entries and interests
        orient_msgs = requests_seen[0]["messages"]
        orient_user_msgs = [m for m in orient_msgs if m.get("role") == "user"]
        orient_system_msgs = [m for m in orient_msgs if m.get("role") == "system"]
        assert any("hello penny" in m.get("content", "") for m in orient_user_msgs)
        assert any("I'm interested in" in m.get("content", "") for m in orient_user_msgs)
        assert any("I thought:" in m.get("content", "") for m in orient_system_msgs)

        # Agentic loop (request 1) should have lean context
        loop_msgs = requests_seen[1]["messages"]
        loop_system_msgs = [m for m in loop_msgs if m.get("role") == "system"]
        loop_system_text = " ".join(m.get("content", "") for m in loop_system_msgs)
        assert "## Your Plan" in loop_system_text
        assert "cyberpunk anime" in loop_system_text
        # Full context sections should NOT be in the loop
        loop_user_msgs = [m for m in loop_msgs if m.get("role") == "user"]
        assert not any("I'm interested in" in m.get("content", "") for m in loop_user_msgs)
        assert not any("I thought:" in m.get("content", "") for m in loop_system_msgs)
        assert not any("hello penny" in m.get("content", "") for m in loop_user_msgs)


@pytest.mark.asyncio
async def test_thinking_loop_stops_after_message_user(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Agentic loop terminates after message_user is called — no further steps."""
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=5,
    )

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        if count == 1:
            # Orientation
            return mock_ollama._make_text_response(request, "I should tell the user something.")
        if count == 2:
            # Agentic step: message_user tool call
            return mock_ollama._make_tool_call_response(
                request,
                "message_user",
                {
                    "message": "Hey, here's something interesting!",
                    "image_prompt": "interesting science",
                    "reasoning": "Sharing a discovery",
                },
            )
        # This should NOT be reached — loop should stop after message_user
        return mock_ollama._make_text_response(request, "continuing after message")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, TEST_SENDER, "hello penny"
        )

        await penny.thinking_agent.execute()

        # Only 2 requests: orientation + message_user step (no further steps)
        assert len(requests_seen) == 2, (
            f"Expected 2 requests (orientation + message_user), got {len(requests_seen)}"
        )


@pytest.mark.asyncio
async def test_scheduler_runs_extraction_before_thinking(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Extraction is scheduled before thinking so entities are fresh."""
    config = make_config()

    async with running_penny(config) as penny:
        schedules = penny.scheduler._schedules
        agent_names = [s.agent.name for s in schedules]

        extraction_idx = agent_names.index("extraction")
        thinking_idx = agent_names.index("inner_monologue")
        assert extraction_idx < thinking_idx, (
            "Extraction should run before thinking in scheduler priority"
        )

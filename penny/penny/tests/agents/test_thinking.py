"""Integration tests for ThinkingAgent: continuous inner monologue loop."""

from datetime import datetime

import pytest

from penny.constants import PennyConstants
from penny.tests.conftest import TEST_SENDER


def _seed_thinking(penny):
    """Seed a message (so user exists), history, and preference (so seed topic exists)."""
    penny.db.messages.log_message(
        PennyConstants.MessageDirection.INCOMING, TEST_SENDER, "hello penny"
    )
    penny.db.history.add(
        user=TEST_SENDER,
        period_start=datetime(2026, 3, 3),
        period_end=datetime(2026, 3, 4),
        duration=PennyConstants.HistoryDuration.DAILY,
        topics="- Quantum gravity experiments\n- Cyberpunk anime releases",
    )
    penny.db.preferences.add(
        user=TEST_SENDER,
        content="Quantum gravity experiments",
        valence="positive",
        source_period_start=datetime(2026, 3, 3),
        source_period_end=datetime(2026, 3, 4),
    )


@pytest.mark.asyncio
async def test_thinking_loop_accumulates_monologue(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """ThinkingAgent accumulates inner monologue text across steps and stores summary."""
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=3,
    )

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        if count <= 3:
            return mock_ollama._make_text_response(request, f"Thinking step {count}...")
        # Summary call
        return mock_ollama._make_text_response(request, "Explored AI topics, found nothing new.")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_thinking(penny)

        await penny.thinking_agent.execute()

        # 3 thinking steps + 1 summary = 4 requests
        assert len(requests_seen) == 4

        # First request should have tools (not final step)
        assert requests_seen[0].get("tools") is not None

        # "continue" messages should be injected between text steps
        second_request_msgs = requests_seen[1]["messages"]
        user_msgs = [m for m in second_request_msgs if m.get("role") == "user"]
        assert any(m.get("content") == "keep exploring" for m in user_msgs)

        # Summary stored as thought
        thoughts = penny.db.thoughts.get_recent(TEST_SENDER, limit=10)
        assert len(thoughts) == 1
        assert "AI topics" in thoughts[0].content


@pytest.mark.asyncio
async def test_thinking_loop_uses_tools(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """ThinkingAgent calls tools during thinking and continues after tool results."""
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=3,
    )

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        if count == 1:
            return mock_ollama._make_tool_call_response(
                request,
                "search",
                {"query": "latest AI news", "reasoning": "Curious about AI developments"},
            )
        if count == 2:
            return mock_ollama._make_text_response(
                request, "Found some interesting AI news from the search."
            )
        if count == 3:
            return mock_ollama._make_text_response(request, "Nothing else to explore right now.")
        # Summary call
        return mock_ollama._make_text_response(
            request, "Searched for AI news, found recent developments."
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_thinking(penny)

        await penny.thinking_agent.execute()

        # 3 thinking steps + 1 summary = 4 requests
        assert len(requests_seen) == 4

        # Step 2 should see tool results in messages (role=tool)
        step2_msgs = requests_seen[1]["messages"]
        tool_msgs = [m for m in step2_msgs if m.get("role") == "tool"]
        assert len(tool_msgs) >= 1

        # Summary stored
        thoughts = penny.db.thoughts.get_recent(TEST_SENDER, limit=10)
        assert len(thoughts) == 1
        assert "AI news" in thoughts[0].content


@pytest.mark.asyncio
async def test_thinking_stores_summary_not_raw_monologue(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Only the summary is stored as a thought, not individual monologue entries."""
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=2,
    )

    def handler(request, count):
        if count == 1:
            return mock_ollama._make_text_response(
                request, "I'm thinking about space exploration..."
            )
        if count == 2:
            return mock_ollama._make_text_response(request, "Also curious about quantum computing.")
        # Summary call
        return mock_ollama._make_text_response(
            request, "Thought about space and quantum computing."
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_thinking(penny)

        await penny.thinking_agent.execute()

        thoughts = penny.db.thoughts.get_recent(TEST_SENDER, limit=10)
        thought_texts = [t.content for t in thoughts]

        # Only the summary should be stored, not raw monologue entries
        assert len(thoughts) == 1
        assert "space and quantum" in thoughts[0].content
        assert not any("I'm thinking about" in t for t in thought_texts)


@pytest.mark.asyncio
async def test_thinking_seed_topic_drives_prompt(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
):
    """Thinking is seeded with a random topic, not a generic 'go'."""
    # Force non-free-thinking path so seed topic is used
    monkeypatch.setattr("penny.agents.thinking.random.random", lambda: 0.99)
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=1,
    )

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        return mock_ollama._make_text_response(request, "ok")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_thinking(penny)

        await penny.thinking_agent.execute()

        assert len(requests_seen) >= 1
        first_msgs = requests_seen[0]["messages"]
        user_msgs = [m for m in first_msgs if m.get("role") == "user"]

        # Should have "Think about ..." not "go"
        first_user = user_msgs[0]["content"]
        assert first_user.startswith("Think about ")
        assert first_user != "go"


@pytest.mark.asyncio
async def test_thinking_context_has_no_raw_conversation(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
):
    """Thinking gets thought summaries but NOT raw conversation turns."""
    # Force non-free-thinking path so context is injected
    monkeypatch.setattr("penny.agents.thinking.random.random", lambda: 0.99)
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=1,
    )

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        return mock_ollama._make_text_response(request, "ok")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_thinking(penny)

        # Seed messages
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

        await penny.thinking_agent.execute()

        assert len(requests_seen) >= 1
        first_msgs = requests_seen[0]["messages"]

        # Thought appears in system prompt as background thinking
        system_msgs = [m for m in first_msgs if m.get("role") == "system"]
        all_system_text = " ".join(m.get("content", "") for m in system_msgs)
        assert "Recent Background Thinking" in all_system_text
        assert "test thought" in all_system_text

        # Raw conversation should NOT appear as user/assistant turns
        user_msgs = [m for m in first_msgs if m.get("role") == "user"]
        assert not any("hello penny" in m.get("content", "") for m in user_msgs)

        assistant_msgs = [m for m in first_msgs if m.get("role") == "assistant"]
        assert not any("hey there!" in m.get("content", "") for m in assistant_msgs)


@pytest.mark.asyncio
async def test_thinking_has_penny_identity(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Thinking IS Penny — PENNY_IDENTITY should be in the system prompt."""
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=1,
    )

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        return mock_ollama._make_text_response(request, "ok")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_thinking(penny)

        await penny.thinking_agent.execute()

        assert len(requests_seen) >= 1
        system_msgs = [m for m in requests_seen[0]["messages"] if m.get("role") == "system"]
        all_system_text = " ".join(m.get("content", "") for m in system_msgs)

        assert "Penny" in all_system_text
        assert "inner monologue" in all_system_text


@pytest.mark.asyncio
async def test_thinking_no_message_user_tool(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """ThinkingAgent has no message_user or recall tools — it only researches."""
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=1,
    )

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        return mock_ollama._make_text_response(request, "ok")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_thinking(penny)

        await penny.thinking_agent.execute()

        # Check tools in the first request — should NOT have message_user
        assert len(requests_seen) >= 1
        tools = requests_seen[0].get("tools") or []
        tool_names = [t["function"]["name"] for t in tools]
        assert "message_user" not in tool_names

        # ThinkingAgent should not have set_channel method
        assert not hasattr(penny.thinking_agent, "set_channel")


@pytest.mark.asyncio
async def test_thinking_browses_news_when_no_seed_topics(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """ThinkingAgent browses news when no seed topics are available."""
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=1,
    )

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        return mock_ollama._make_text_response(request, "ok")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # User exists but no history — no seed topics
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, TEST_SENDER, "hello"
        )

        await penny.thinking_agent.execute()

        # Should still run (browse news fallback) and produce a thought
        assert len(requests_seen) > 0
        thoughts = penny.db.thoughts.get_recent(TEST_SENDER, limit=10)
        assert len(thoughts) == 1


@pytest.mark.asyncio
async def test_scheduler_runs_history_before_thinking(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """History is scheduled before thinking so context is fresh."""
    config = make_config()

    async with running_penny(config) as penny:
        schedules = penny.scheduler._schedules
        agent_names = [s.agent.name for s in schedules]

        history_idx = agent_names.index("history")
        thinking_idx = agent_names.index("inner_monologue")
        assert history_idx < thinking_idx, (
            "History should run before thinking in scheduler priority"
        )

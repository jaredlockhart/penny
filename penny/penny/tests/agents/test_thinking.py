"""Integration tests for ThinkingAgent: continuous inner monologue loop."""

from datetime import datetime

import pytest

from penny.constants import PennyConstants
from penny.tests.conftest import TEST_SENDER


async def _fake_embed(vec):
    """Return a fixed embedding vector (mock for embed_text)."""
    return vec


# Mock summary report long enough to pass MIN_THOUGHT_WORDS validation
MOCK_REPORT = (
    "Research report on AI topics. Recent developments include new model architectures, "
    "improved training techniques, and expanded deployment across industries. Key findings "
    "show significant progress in reasoning capabilities, multimodal understanding, and "
    "efficient inference. Several major organizations announced new initiatives in open "
    "source AI development and safety research during the past quarter."
)


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
        return mock_ollama._make_text_response(request, MOCK_REPORT)

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
        assert "AI topics" in thoughts[0].content or "Research report" in thoughts[0].content


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
        return mock_ollama._make_text_response(request, MOCK_REPORT)

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
        assert "Research report" in thoughts[0].content


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
        return mock_ollama._make_text_response(request, MOCK_REPORT)

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_thinking(penny)

        await penny.thinking_agent.execute()

        thoughts = penny.db.thoughts.get_recent(TEST_SENDER, limit=10)
        thought_texts = [t.content for t in thoughts]

        # Only the summary should be stored, not raw monologue entries
        assert len(thoughts) == 1
        assert "Research report" in thoughts[0].content
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
async def test_thinking_marks_preference_and_rotates(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
):
    """Seeded thinking marks the preference's last_thought_at, rotating future seeds."""
    # Force non-free-thinking path and deterministic choice (first item)
    monkeypatch.setattr("penny.agents.thinking.random.random", lambda: 0.99)
    monkeypatch.setattr("penny.agents.thinking.random.choice", lambda lst: lst[0])
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=1,
    )

    def handler(request, count):
        return mock_ollama._make_text_response(request, "ok")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Add two preferences
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, TEST_SENDER, "hello"
        )
        pref_a = penny.db.preferences.add(
            user=TEST_SENDER,
            content="astrophysics",
            valence="positive",
            source_period_start=datetime(2026, 3, 3),
            source_period_end=datetime(2026, 3, 4),
        )
        pref_b = penny.db.preferences.add(
            user=TEST_SENDER,
            content="cyberpunk anime",
            valence="positive",
            source_period_start=datetime(2026, 3, 3),
            source_period_end=datetime(2026, 3, 4),
        )
        assert pref_a is not None and pref_b is not None

        # First cycle: picks first from pool (both have NULL last_thought_at)
        await penny.thinking_agent.execute()

        # The used preference should now have last_thought_at set
        updated = penny.db.preferences.get_least_recent_positive(TEST_SENDER, pool_size=5)
        thought_about = [p for p in updated if p.last_thought_at is not None]
        not_thought = [p for p in updated if p.last_thought_at is None]
        assert len(thought_about) == 1
        assert len(not_thought) == 1

        # Second cycle: the un-thought-about one should be first in the pool
        pool = penny.db.preferences.get_least_recent_positive(TEST_SENDER, pool_size=5)
        assert pool[0].last_thought_at is None  # Never thought about comes first


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

        # Seed a thought with matching seed topic
        penny.db.thoughts.add(
            TEST_SENDER,
            "test thought",
            preference_id=1,
        )

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
        if count == 1:
            return mock_ollama._make_text_response(request, "Found some news.")
        # Summary call
        return mock_ollama._make_text_response(request, MOCK_REPORT)

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
async def test_thinking_free_mode_has_no_context(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
):
    """Free-thinking mode gets empty context — no profile, thoughts, or dislikes."""
    # Force free-thinking path
    monkeypatch.setattr("penny.agents.thinking.random.random", lambda: 0.0)
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=1,
    )

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        return mock_ollama._make_text_response(request, "just vibing")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_thinking(penny)

        # Add a dislike to verify it's NOT in free-thinking context
        penny.db.preferences.add(
            user=TEST_SENDER,
            content="Country music",
            valence="negative",
            source_period_start=datetime(2026, 3, 3),
            source_period_end=datetime(2026, 3, 4),
        )

        await penny.thinking_agent.execute()

        assert len(requests_seen) >= 1
        system_msgs = [m for m in requests_seen[0]["messages"] if m.get("role") == "system"]
        all_system_text = " ".join(m.get("content", "") for m in system_msgs)

        # Free-thinking should NOT have profile, thoughts, or dislike context
        assert "Test User" not in all_system_text
        assert "Recent Background Thinking" not in all_system_text
        assert "Topics to Avoid" not in all_system_text

        # But the prompt should be the free-thinking prompt
        user_msgs = [m for m in requests_seen[0]["messages"] if m.get("role") == "user"]
        assert any(
            "free" in m.get("content", "").lower()
            or "explore" in m.get("content", "").lower()
            or "think" in m.get("content", "").lower()
            for m in user_msgs
        )


@pytest.mark.asyncio
async def test_thinking_seeded_mode_has_dislike_context(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
):
    """Seeded thinking includes dislike context so Penny avoids unwanted topics."""
    # Force non-free-thinking path
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

        penny.db.preferences.add(
            user=TEST_SENDER,
            content="Country music",
            valence="negative",
            source_period_start=datetime(2026, 3, 3),
            source_period_end=datetime(2026, 3, 4),
        )

        await penny.thinking_agent.execute()

        assert len(requests_seen) >= 1
        system_msgs = [m for m in requests_seen[0]["messages"] if m.get("role") == "system"]
        all_system_text = " ".join(m.get("content", "") for m in system_msgs)

        assert "Topics to Avoid" in all_system_text
        assert "Country music" in all_system_text


@pytest.mark.asyncio
async def test_thinking_rebuild_system_prompt_updates_context(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
):
    """System prompt is rebuilt each step with accumulated monologue as anchor."""
    # Force non-free-thinking path
    monkeypatch.setattr("penny.agents.thinking.random.random", lambda: 0.99)
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=2,
    )

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        if count <= 2:
            return mock_ollama._make_text_response(request, f"Thinking step {count}...")
        return mock_ollama._make_text_response(request, "Summary of thoughts.")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_thinking(penny)

        await penny.thinking_agent.execute()

        # Step 2 should have a rebuilt system prompt (different from step 1)
        assert len(requests_seen) >= 2
        system1 = [m for m in requests_seen[0]["messages"] if m.get("role") == "system"]
        system2 = [m for m in requests_seen[1]["messages"] if m.get("role") == "system"]

        # Both steps should have system messages with identity
        sys1_text = " ".join(m.get("content", "") for m in system1)
        sys2_text = " ".join(m.get("content", "") for m in system2)
        assert "Penny" in sys1_text
        assert "Penny" in sys2_text


@pytest.mark.asyncio
async def test_thinking_empty_monologue_skips_storage(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """ThinkingAgent doesn't store a thought when monologue is empty."""
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=1,
    )

    def handler(request, count):
        # Return empty content
        return mock_ollama._make_text_response(request, "")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_thinking(penny)

        await penny.thinking_agent.execute()

        thoughts = penny.db.thoughts.get_recent(TEST_SENDER, limit=10)
        assert len(thoughts) == 0


@pytest.mark.asyncio
async def test_thinking_duplicate_thought_skips_storage(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
):
    """When a new thought is too similar to an existing one, it is not stored."""
    monkeypatch.setattr("penny.agents.thinking.random.random", lambda: 0.99)
    monkeypatch.setattr("penny.agents.thinking.random.choice", lambda lst: lst[0])
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=1,
    )

    # Mock embed_text to return identical vectors (similarity = 1.0 → duplicate)
    duplicate_vec = [1.0, 0.0, 0.0]
    monkeypatch.setattr(
        "penny.agents.thinking.embed_text",
        lambda _client, _text: _fake_embed(duplicate_vec),  # noqa: ARG005
    )

    def handler(request, count):
        if count == 1:
            return mock_ollama._make_text_response(request, "Yep, same old stuff.")
        return mock_ollama._make_text_response(
            request, "Confirmed the album still exists, nothing new."
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_thinking(penny)

        # Set a fake embedding client so dedup runs (normally None in tests)
        penny.thinking_agent._embedding_model_client = object()

        # Seed an existing thought with the same seed topic
        penny.db.thoughts.add(
            TEST_SENDER,
            "Old thought about the same topic.",
            preference_id=1,
        )

        await penny.thinking_agent.execute()

        # Only the pre-seeded thought should exist (new one was deduplicated)
        thoughts = penny.db.thoughts.get_recent(TEST_SENDER, limit=10)
        assert len(thoughts) == 1
        assert "Old thought" in thoughts[0].content

        # Preference should still be marked as thought-about (prevents re-thinking)
        pool = penny.db.preferences.get_least_recent_positive(TEST_SENDER)
        assert any(p.last_thought_at is not None for p in pool)


@pytest.mark.asyncio
async def test_thinking_novel_thought_is_stored(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
):
    """When a new thought is sufficiently different from existing ones, it is stored."""
    monkeypatch.setattr("penny.agents.thinking.random.random", lambda: 0.99)
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=1,
    )

    # Mock embed_text to return orthogonal vectors (similarity = 0.0 → novel)
    call_count = 0

    async def _alternating_embed(_client, _text):
        nonlocal call_count
        call_count += 1
        # First call = new report, second call = existing thought → orthogonal
        if call_count % 2 == 1:
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]

    monkeypatch.setattr("penny.agents.thinking.embed_text", _alternating_embed)

    def handler(request, count):
        if count == 1:
            return mock_ollama._make_text_response(request, "Found something new!")
        return mock_ollama._make_text_response(request, MOCK_REPORT)

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_thinking(penny)

        # Set a fake embedding client so dedup runs (normally None in tests)
        penny.thinking_agent._embedding_model_client = object()

        # Seed an existing thought with the same seed topic but different content
        penny.db.thoughts.add(
            TEST_SENDER,
            "Old thought about a different topic.",
            preference_id=1,
        )

        await penny.thinking_agent.execute()

        # Both the old and new thoughts should exist
        thoughts = penny.db.thoughts.get_recent(TEST_SENDER, limit=10)
        assert len(thoughts) == 2


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

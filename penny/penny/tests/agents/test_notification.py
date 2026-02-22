"""Integration tests for the NotificationAgent."""

from datetime import UTC, datetime

import pytest

from penny.agents.notification import NotificationAgent
from penny.constants import PennyConstants
from penny.tests.conftest import TEST_SENDER


def _create_notification_agent(penny, config):
    """Create a NotificationAgent wired to penny's DB and channel."""
    agent = NotificationAgent(
        system_prompt="",
        background_model_client=penny.background_model_client,
        foreground_model_client=penny.foreground_model_client,
        tools=[],
        db=penny.db,
        max_steps=1,
        tool_timeout=config.tool_timeout,
        config=config,
    )
    agent.set_channel(penny.channel)
    return agent


@pytest.mark.asyncio
async def test_notification_sends_highest_interest_entity(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Notification agent picks the entity with the highest interest score."""
    config = make_config()

    def handler(request: dict, count: int) -> dict:
        messages = request.get("messages", [])
        prompt = messages[-1]["content"] if messages else ""
        if "came across" in prompt:
            return mock_ollama._make_text_response(request, prompt)
        return mock_ollama._make_text_response(request, "ok")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        msg_id = penny.db.log_message(direction="incoming", sender=TEST_SENDER, content="hello")
        penny.db.mark_messages_processed([msg_id])

        # Create two entities — one with higher interest
        low_entity = penny.db.get_or_create_entity(TEST_SENDER, "low interest thing")
        high_entity = penny.db.get_or_create_entity(TEST_SENDER, "high interest thing")
        assert low_entity is not None and low_entity.id is not None
        assert high_entity is not None and high_entity.id is not None

        # Low interest: weak engagement
        penny.db.add_engagement(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.MESSAGE_MENTION,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.1,
            entity_id=low_entity.id,
        )
        # High interest: strong engagement
        penny.db.add_engagement(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.USER_SEARCH,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=1.0,
            entity_id=high_entity.id,
        )

        # Add un-notified facts to both
        penny.db.add_fact(low_entity.id, "Low interest fact")
        penny.db.add_fact(high_entity.id, "High interest fact")

        agent = _create_notification_agent(penny, config)
        signal_server.outgoing_messages.clear()
        result = await agent.execute()
        assert result is True

        # Should notify about the high-interest entity
        msgs = signal_server.outgoing_messages
        assert len(msgs) == 1
        assert "high interest thing" in msgs[0]["message"]


@pytest.mark.asyncio
async def test_notification_marks_facts_notified(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """After sending a notification, facts are marked with notified_at."""
    config = make_config()

    def handler(request: dict, count: int) -> dict:
        msg = (
            "Hey, I just came across some really interesting stuff"
            " about this topic that I think you'd like!"
        )
        return mock_ollama._make_text_response(request, msg)

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        msg_id = penny.db.log_message(direction="incoming", sender=TEST_SENDER, content="hello")
        penny.db.mark_messages_processed([msg_id])

        entity = penny.db.get_or_create_entity(TEST_SENDER, "test entity")
        assert entity is not None and entity.id is not None
        penny.db.add_fact(entity.id, "Fact one")
        penny.db.add_fact(entity.id, "Fact two")

        # Verify facts start un-notified
        facts_before = penny.db.get_entity_facts(entity.id)
        assert all(f.notified_at is None for f in facts_before)

        agent = _create_notification_agent(penny, config)
        await agent.execute()

        # Facts should now be marked as notified
        facts_after = penny.db.get_entity_facts(entity.id)
        assert all(f.notified_at is not None for f in facts_after)


@pytest.mark.asyncio
async def test_notification_one_per_cycle(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Only one notification is sent per execute() cycle."""
    config = make_config()

    def handler(request: dict, count: int) -> dict:
        return mock_ollama._make_text_response(
            request,
            "Here's what I discovered — really interesting new findings about this topic!",
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        msg_id = penny.db.log_message(direction="incoming", sender=TEST_SENDER, content="hello")
        penny.db.mark_messages_processed([msg_id])

        # Create two entities with un-notified facts
        e1 = penny.db.get_or_create_entity(TEST_SENDER, "entity one")
        e2 = penny.db.get_or_create_entity(TEST_SENDER, "entity two")
        assert e1 is not None and e1.id is not None
        assert e2 is not None and e2.id is not None
        penny.db.add_fact(e1.id, "Fact for entity one")
        penny.db.add_fact(e2.id, "Fact for entity two")

        agent = _create_notification_agent(penny, config)
        signal_server.outgoing_messages.clear()
        await agent.execute()

        # Only one notification sent
        assert len(signal_server.outgoing_messages) == 1


@pytest.mark.asyncio
async def test_notification_skips_user_message_facts(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Facts pre-marked as notified (user-sourced) don't trigger notifications."""
    config = make_config()
    mock_ollama.set_default_flow(search_query="test", final_response="test response")

    async with running_penny(config) as penny:
        msg_id = penny.db.log_message(direction="incoming", sender=TEST_SENDER, content="hello")
        penny.db.mark_messages_processed([msg_id])

        entity = penny.db.get_or_create_entity(TEST_SENDER, "pre-notified entity")
        assert entity is not None and entity.id is not None
        # Pre-mark as notified (simulates user-sourced facts)
        penny.db.add_fact(entity.id, "Already notified fact", notified_at=datetime.now(UTC))

        agent = _create_notification_agent(penny, config)
        signal_server.outgoing_messages.clear()
        result = await agent.execute()

        assert result is False
        assert len(signal_server.outgoing_messages) == 0


@pytest.mark.asyncio
async def test_notification_mentions_learn_topic(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """When facts come from a /learn search, the notification mentions the learn topic."""
    config = make_config()

    captured_prompts: list[str] = []

    def handler(request: dict, count: int) -> dict:
        messages = request.get("messages", [])
        prompt = messages[-1]["content"] if messages else ""
        captured_prompts.append(prompt)
        return mock_ollama._make_text_response(
            request,
            "While looking into audiophile gear, I found out about KEF LS50 Meta — cool stuff!",
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        msg_id = penny.db.log_message(direction="incoming", sender=TEST_SENDER, content="hello")
        penny.db.mark_messages_processed([msg_id])

        # Create the full chain: LearnPrompt → SearchLog → Fact
        learn_prompt = penny.db.create_learn_prompt(
            user=TEST_SENDER,
            prompt_text="audiophile gear",
            searches_remaining=0,
        )
        assert learn_prompt is not None and learn_prompt.id is not None

        penny.db.log_search(
            query="audiophile gear",
            response="KEF LS50 Meta is a popular bookshelf speaker...",
            trigger="learn_command",
            learn_prompt_id=learn_prompt.id,
        )
        search_logs = penny.db.get_search_logs_by_learn_prompt(learn_prompt.id)
        assert len(search_logs) == 1

        entity = penny.db.get_or_create_entity(TEST_SENDER, "kef ls50 meta")
        assert entity is not None and entity.id is not None
        penny.db.add_fact(
            entity.id,
            "KEF LS50 Meta uses Metamaterial Absorption Technology",
            source_search_log_id=search_logs[0].id,
        )

        agent = _create_notification_agent(penny, config)
        signal_server.outgoing_messages.clear()
        result = await agent.execute()
        assert result is True

        # The prompt sent to the LLM should mention the learn topic
        assert any("audiophile gear" in p for p in captured_prompts)


@pytest.mark.asyncio
async def test_notification_backoff_and_reset(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Notification respects backoff: send, suppress, reset on user message, send again."""
    config = make_config()

    def handler(request: dict, count: int) -> dict:
        return mock_ollama._make_text_response(
            request,
            "Here's an interesting discovery — some really great new facts about this topic!",
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        msg_id = penny.db.log_message(direction="incoming", sender=TEST_SENDER, content="hello")
        penny.db.mark_messages_processed([msg_id])

        agent = _create_notification_agent(penny, config)

        # --- Cycle 1: notification sent (no backoff) ---
        e1 = penny.db.get_or_create_entity(TEST_SENDER, "backoff entity 1")
        assert e1 is not None and e1.id is not None
        penny.db.add_fact(e1.id, "Fact for backoff test 1")

        signal_server.outgoing_messages.clear()
        result1 = await agent.execute()
        assert result1 is True
        assert len(signal_server.outgoing_messages) == 1

        # --- Cycle 2: suppressed (backoff active, no user reply) ---
        e2 = penny.db.get_or_create_entity(TEST_SENDER, "backoff entity 2")
        assert e2 is not None and e2.id is not None
        penny.db.add_fact(e2.id, "Fact for backoff test 2")

        signal_server.outgoing_messages.clear()
        result2 = await agent.execute()
        assert result2 is False
        assert len(signal_server.outgoing_messages) == 0

        # --- User sends message → resets backoff ---
        msg_id2 = penny.db.log_message(direction="incoming", sender=TEST_SENDER, content="thanks!")
        penny.db.mark_messages_processed([msg_id2])

        # --- Cycle 3: notification sent (backoff reset) ---
        signal_server.outgoing_messages.clear()
        result3 = await agent.execute()
        assert result3 is True
        assert len(signal_server.outgoing_messages) == 1


@pytest.mark.asyncio
async def test_notification_command_resets_backoff(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Commands (like /learn) reset notification backoff, not just regular messages."""
    config = make_config()

    def handler(request: dict, count: int) -> dict:
        return mock_ollama._make_text_response(
            request,
            "Here's an interesting discovery — some really great new facts about this topic!",
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        msg_id = penny.db.log_message(direction="incoming", sender=TEST_SENDER, content="hello")
        penny.db.mark_messages_processed([msg_id])

        agent = _create_notification_agent(penny, config)

        # --- Cycle 1: notification sent (no backoff) ---
        e1 = penny.db.get_or_create_entity(TEST_SENDER, "command backoff entity 1")
        assert e1 is not None and e1.id is not None
        penny.db.add_fact(e1.id, "Fact for command backoff test 1")

        signal_server.outgoing_messages.clear()
        result1 = await agent.execute()
        assert result1 is True

        # --- Cycle 2: suppressed (backoff active) ---
        e2 = penny.db.get_or_create_entity(TEST_SENDER, "command backoff entity 2")
        assert e2 is not None and e2.id is not None
        penny.db.add_fact(e2.id, "Fact for command backoff test 2")

        signal_server.outgoing_messages.clear()
        result2 = await agent.execute()
        assert result2 is False

        # --- User sends a command (not a message) → should also reset backoff ---
        penny.db.log_command(
            user=TEST_SENDER,
            channel_type="signal",
            command_name="learn",
            command_args="kef speakers",
            response="Okay, I'll learn more about kef speakers",
        )

        # --- Cycle 3: notification sent (backoff reset by command) ---
        signal_server.outgoing_messages.clear()
        result3 = await agent.execute()
        assert result3 is True
        assert len(signal_server.outgoing_messages) == 1


@pytest.mark.asyncio
async def test_notification_expired_backoff_stays_at_cadence(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """At max backoff, _mark_proactive_sent doubles (clamped at max), not reset to initial.

    Tests the backoff state machine directly: when backoff expires and a notification
    fires, the next backoff should double from the current value.
    """
    from datetime import timedelta

    from penny.agents.backoff import BackoffState

    config = make_config()
    max_backoff = config.runtime.NOTIFICATION_MAX_BACKOFF
    initial_backoff = config.runtime.NOTIFICATION_INITIAL_BACKOFF

    async with running_penny(config) as penny:
        agent = _create_notification_agent(penny, config)

        # Simulate state: backoff at 480s, expired (last send was 481s ago)
        state = BackoffState()
        state.backoff_seconds = 480.0
        state.last_action_time = datetime.now(UTC) - timedelta(seconds=481)
        agent._backoff_state[TEST_SENDER] = state

        # _should_send should return True (backoff expired)
        assert agent._should_send(TEST_SENDER) is True
        # But backoff value should NOT have been reset to 0
        assert state.backoff_seconds == 480.0

        # After sending, _mark_proactive_sent doubles from 480 → 960
        agent._mark_proactive_sent(TEST_SENDER)
        assert state.backoff_seconds == 960.0

        # At max backoff, stays clamped
        state.backoff_seconds = max_backoff
        agent._mark_proactive_sent(TEST_SENDER)
        assert state.backoff_seconds == max_backoff

        # User interaction resets to eager (0), then first send starts fresh
        state.backoff_seconds = 0.0
        agent._mark_proactive_sent(TEST_SENDER)
        assert state.backoff_seconds == initial_backoff


@pytest.mark.asyncio
async def test_learn_completion_announcement(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Completion announcement sent when learn prompt is completed and all search logs extracted."""
    config = make_config()

    captured_prompts: list[str] = []

    def handler(request: dict, count: int) -> dict:
        messages = request.get("messages", [])
        prompt = messages[-1]["content"] if messages else ""
        captured_prompts.append(prompt)
        return mock_ollama._make_text_response(
            request,
            "I finished researching **kef speakers** and found some cool stuff about "
            "**kef ls50 meta** — they use Metamaterial Absorption Technology!",
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        msg_id = penny.db.log_message(direction="incoming", sender=TEST_SENDER, content="hello")
        penny.db.mark_messages_processed([msg_id])

        # Create a completed learn prompt with extracted search logs
        lp = penny.db.create_learn_prompt(
            user=TEST_SENDER,
            prompt_text="kef speakers",
            searches_remaining=0,
        )
        assert lp is not None and lp.id is not None
        penny.db.update_learn_prompt_status(lp.id, PennyConstants.LearnPromptStatus.COMPLETED)

        penny.db.log_search(
            query="kef speakers overview",
            response="KEF makes great speakers...",
            trigger="learn_command",
            learn_prompt_id=lp.id,
        )
        search_logs = penny.db.get_search_logs_by_learn_prompt(lp.id)
        assert len(search_logs) == 1
        # Mark as extracted
        assert search_logs[0].id is not None
        penny.db.mark_search_extracted(search_logs[0].id)

        # Create entity and fact linked to the search log
        entity = penny.db.get_or_create_entity(TEST_SENDER, "kef ls50 meta")
        assert entity is not None and entity.id is not None
        penny.db.add_engagement(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.USER_SEARCH,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=1.0,
            entity_id=entity.id,
        )
        penny.db.add_fact(
            entity.id,
            "KEF LS50 Meta uses Metamaterial Absorption Technology",
            source_search_log_id=search_logs[0].id,
        )

        agent = _create_notification_agent(penny, config)
        signal_server.outgoing_messages.clear()
        result = await agent.execute()
        assert result is True

        # Model was given the topic and entity facts to compose a summary
        assert any("kef speakers" in p for p in captured_prompts)
        assert any("kef ls50 meta" in p for p in captured_prompts)
        assert any("Metamaterial Absorption Technology" in p for p in captured_prompts)

        # Announcement was sent
        msgs = signal_server.outgoing_messages
        assert len(msgs) == 1

        # LearnPrompt should be marked as announced
        updated_lp = penny.db.get_learn_prompt(lp.id)
        assert updated_lp is not None
        assert updated_lp.announced_at is not None


@pytest.mark.asyncio
async def test_learn_completion_not_sent_when_unextracted(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Completion announcement NOT sent when search logs are still unextracted."""
    config = make_config()
    mock_ollama.set_default_flow(search_query="test", final_response="ok")

    async with running_penny(config) as penny:
        msg_id = penny.db.log_message(direction="incoming", sender=TEST_SENDER, content="hello")
        penny.db.mark_messages_processed([msg_id])

        # Create a completed learn prompt with UN-extracted search log
        lp = penny.db.create_learn_prompt(
            user=TEST_SENDER,
            prompt_text="kef speakers",
            searches_remaining=0,
        )
        assert lp is not None and lp.id is not None
        penny.db.update_learn_prompt_status(lp.id, PennyConstants.LearnPromptStatus.COMPLETED)

        penny.db.log_search(
            query="kef speakers overview",
            response="KEF makes great speakers...",
            trigger="learn_command",
            learn_prompt_id=lp.id,
        )
        # Do NOT mark as extracted

        agent = _create_notification_agent(penny, config)
        signal_server.outgoing_messages.clear()
        result = await agent.execute()
        assert result is False
        assert len(signal_server.outgoing_messages) == 0

        # LearnPrompt should NOT be announced
        updated_lp = penny.db.get_learn_prompt(lp.id)
        assert updated_lp is not None
        assert updated_lp.announced_at is None


@pytest.mark.asyncio
async def test_learn_completion_marks_facts_notified(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """After learn completion announcement, un-notified facts from the learn prompt are marked."""
    config = make_config()

    def handler(request: dict, count: int) -> dict:
        return mock_ollama._make_text_response(
            request,
            "I finished researching kef speakers and found some interesting stuff!",
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        msg_id = penny.db.log_message(direction="incoming", sender=TEST_SENDER, content="hello")
        penny.db.mark_messages_processed([msg_id])

        lp = penny.db.create_learn_prompt(
            user=TEST_SENDER,
            prompt_text="kef speakers",
            searches_remaining=0,
        )
        assert lp is not None and lp.id is not None
        penny.db.update_learn_prompt_status(lp.id, PennyConstants.LearnPromptStatus.COMPLETED)

        penny.db.log_search(
            query="kef speakers overview",
            response="KEF makes great speakers...",
            trigger="learn_command",
            learn_prompt_id=lp.id,
        )
        search_logs = penny.db.get_search_logs_by_learn_prompt(lp.id)
        assert search_logs[0].id is not None
        penny.db.mark_search_extracted(search_logs[0].id)

        entity = penny.db.get_or_create_entity(TEST_SENDER, "kef ls50 meta")
        assert entity is not None and entity.id is not None
        penny.db.add_fact(
            entity.id,
            "KEF LS50 Meta costs $1,599",
            source_search_log_id=search_logs[0].id,
        )
        penny.db.add_fact(
            entity.id,
            "KEF LS50 Meta uses MAT technology",
            source_search_log_id=search_logs[0].id,
        )

        # Verify facts start un-notified
        facts_before = penny.db.get_entity_facts(entity.id)
        assert all(f.notified_at is None for f in facts_before)

        agent = _create_notification_agent(penny, config)
        await agent.execute()

        # Facts should now be marked as notified
        facts_after = penny.db.get_entity_facts(entity.id)
        assert all(f.notified_at is not None for f in facts_after)


@pytest.mark.asyncio
async def test_learn_completion_sends_one_per_cycle(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Multiple completed learn prompts are announced one per cycle, not all at once."""
    config = make_config()

    def handler(request: dict, count: int) -> dict:
        return mock_ollama._make_text_response(request, "Here's what I found!")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        msg_id = penny.db.log_message(direction="incoming", sender=TEST_SENDER, content="hello")
        penny.db.mark_messages_processed([msg_id])

        # Create two completed learn prompts, both fully extracted
        for topic in ("kef speakers", "nvidia gpus"):
            lp = penny.db.create_learn_prompt(
                user=TEST_SENDER, prompt_text=topic, searches_remaining=0
            )
            assert lp is not None and lp.id is not None
            penny.db.update_learn_prompt_status(lp.id, PennyConstants.LearnPromptStatus.COMPLETED)
            penny.db.log_search(
                query=f"{topic} overview",
                response=f"Info about {topic}...",
                trigger="learn_command",
                learn_prompt_id=lp.id,
            )
            search_logs = penny.db.get_search_logs_by_learn_prompt(lp.id)
            assert search_logs[0].id is not None
            penny.db.mark_search_extracted(search_logs[0].id)

            entity = penny.db.get_or_create_entity(TEST_SENDER, topic)
            assert entity is not None and entity.id is not None
            penny.db.add_fact(
                entity.id, f"Fact about {topic}", source_search_log_id=search_logs[0].id
            )

        agent = _create_notification_agent(penny, config)

        # First cycle: only one announcement sent
        signal_server.outgoing_messages.clear()
        result = await agent.execute()
        assert result is True
        assert len(signal_server.outgoing_messages) == 1

        # One learn prompt announced, one still pending
        prompts = penny.db.get_unannounced_completed_learn_prompts(TEST_SENDER)
        assert len(prompts) == 1

        # Second cycle: the other announcement sent
        signal_server.outgoing_messages.clear()
        result = await agent.execute()
        assert result is True
        assert len(signal_server.outgoing_messages) == 1

        # Both now announced
        prompts = penny.db.get_unannounced_completed_learn_prompts(TEST_SENDER)
        assert len(prompts) == 0

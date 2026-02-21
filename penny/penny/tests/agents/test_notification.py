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
            engagement_type=PennyConstants.EngagementType.LEARN_COMMAND,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=PennyConstants.ENGAGEMENT_STRENGTH_LEARN_COMMAND,
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

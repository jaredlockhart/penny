"""Integration tests for the NotificationAgent."""

from datetime import UTC, datetime

import pytest

from penny.agents.notification import NotificationAgent
from penny.constants import PennyConstants
from penny.ollama.embeddings import serialize_embedding
from penny.tests.conftest import TEST_SENDER, wait_until


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
async def test_notification_prefers_higher_interest_entity(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Notification agent scores entities by interest + enrichment volume.

    With pool_size=1, selection is deterministic (always picks highest score).
    An entity with user engagement should outscore one without.
    """
    config = make_config(notification_pool_size=1)

    captured_prompts: list[str] = []

    def handler(request: dict, count: int) -> dict:
        messages = request.get("messages", [])
        prompt = messages[-1]["content"] if messages else ""
        captured_prompts.append(prompt)
        if "came across" in prompt:
            return mock_ollama._make_text_response(
                request,
                "Hey, I came across **interesting entity** recently and found some"
                " really interesting stuff worth sharing!",
            )
        return mock_ollama._make_text_response(request, "ok")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        msg_id = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="hello"
        )
        penny.db.messages.mark_processed([msg_id])

        # Create two entities — one with interest (engagement), one without
        boring_entity = penny.db.entities.get_or_create(TEST_SENDER, "boring entity")
        interesting_entity = penny.db.entities.get_or_create(TEST_SENDER, "interesting entity")
        assert boring_entity is not None and boring_entity.id is not None
        assert interesting_entity is not None and interesting_entity.id is not None

        # Give interesting_entity an explicit engagement (emoji reaction) to boost score.
        # user_search is filtered out of notification scoring — only explicit signals count.
        penny.db.engagements.add(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.EMOJI_REACTION,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.5,
            entity_id=interesting_entity.id,
        )

        # Both get one fact (same enrichment volume)
        penny.db.facts.add(boring_entity.id, "Boring fact")
        penny.db.facts.add(interesting_entity.id, "Interesting fact")

        agent = _create_notification_agent(penny, config)
        signal_server.outgoing_messages.clear()
        result = await agent.execute()
        assert result is True

        # Should notify about the interesting entity (higher interest score)
        msgs = signal_server.outgoing_messages
        assert len(msgs) == 1
        assert "interesting entity" in msgs[0]["message"]

        # Prompt sent to model should instruct it to synthesize, not echo raw facts
        assert any("Synthesize" in p for p in captured_prompts)


@pytest.mark.asyncio
async def test_notification_vetoes_entity_with_negative_emoji(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Entity with a negative emoji reaction is hard-vetoed from notifications.

    Even if the entity has positive engagements, a single thumbs-down
    completely excludes it. The fallback entity (no engagement) gets picked.
    """
    config = make_config(notification_pool_size=1)

    def handler(request: dict, count: int) -> dict:
        return mock_ollama._make_text_response(
            request,
            "Hey, I came across something about fallback entity that you might find interesting!",
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        msg_id = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="hello"
        )
        penny.db.messages.mark_processed([msg_id])

        # Create two entities — vetoed one has positive AND negative signals
        vetoed = penny.db.entities.get_or_create(TEST_SENDER, "vetoed entity")
        fallback = penny.db.entities.get_or_create(TEST_SENDER, "fallback entity")
        assert vetoed is not None and vetoed.id is not None
        assert fallback is not None and fallback.id is not None

        # Vetoed entity: positive follow-up but negative emoji → hard veto
        penny.db.engagements.add(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.FOLLOW_UP_QUESTION,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.5,
            entity_id=vetoed.id,
        )
        penny.db.engagements.add(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.EMOJI_REACTION,
            valence=PennyConstants.EngagementValence.NEGATIVE,
            strength=0.8,
            entity_id=vetoed.id,
        )

        penny.db.facts.add(vetoed.id, "Vetoed fact")
        penny.db.facts.add(fallback.id, "Fallback fact")

        agent = _create_notification_agent(penny, config)
        signal_server.outgoing_messages.clear()
        result = await agent.execute()
        assert result is True

        # Should pick the fallback, not the vetoed entity
        msgs = signal_server.outgoing_messages
        assert len(msgs) == 1
        assert "fallback entity" in msgs[0]["message"].lower()


@pytest.mark.asyncio
async def test_notification_fatigue_penalizes_over_notified_entity(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Entity with many previously-notified facts is penalized by fatigue.

    Two entities with the same emoji engagement, but the over-notified one
    (many pre-marked facts) should score lower due to fatigue divisor.
    """
    config = make_config(notification_pool_size=1)

    def handler(request: dict, count: int) -> dict:
        return mock_ollama._make_text_response(
            request,
            "Hey, I found something new about fresh entity that you might find really interesting!",
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        msg_id = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="hello"
        )
        penny.db.messages.mark_processed([msg_id])

        fatigued = penny.db.entities.get_or_create(TEST_SENDER, "fatigued entity")
        fresh = penny.db.entities.get_or_create(TEST_SENDER, "fresh entity")
        assert fatigued is not None and fatigued.id is not None
        assert fresh is not None and fresh.id is not None

        # Same emoji engagement for both
        for entity in [fatigued, fresh]:
            penny.db.engagements.add(
                user=TEST_SENDER,
                engagement_type=PennyConstants.EngagementType.EMOJI_REACTION,
                valence=PennyConstants.EngagementValence.POSITIVE,
                strength=0.5,
                entity_id=entity.id,
            )

        # Fatigued entity: 50 pre-notified facts (already told user about it many times)
        now = datetime.now(UTC)
        for i in range(50):
            penny.db.facts.add(fatigued.id, f"Old fact {i}", notified_at=now)

        # Both get one new un-notified fact
        penny.db.facts.add(fatigued.id, "New fatigued fact")
        penny.db.facts.add(fresh.id, "New fresh fact")

        agent = _create_notification_agent(penny, config)
        signal_server.outgoing_messages.clear()
        result = await agent.execute()
        assert result is True

        # Fresh entity should win (lower fatigue divisor)
        msgs = signal_server.outgoing_messages
        assert len(msgs) == 1
        assert "fresh entity" in msgs[0]["message"].lower()


@pytest.mark.asyncio
async def test_notification_auto_tuning_records_ignored(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """When a notification gets no engagement, the next cycle records NOTIFICATION_IGNORED.

    First cycle: sends notification for entity A.
    Second cycle: no engagement since → records NOTIFICATION_IGNORED for A.
    """
    config = make_config(
        notification_pool_size=1,
        notification_initial_backoff=0,
        notification_entity_cooldown=0,
    )

    def handler(request: dict, count: int) -> dict:
        return mock_ollama._make_text_response(
            request,
            "Hey, I came across some interesting news about this entity that is worth sharing!",
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        msg_id = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="hello"
        )
        penny.db.messages.mark_processed([msg_id])

        entity_a = penny.db.entities.get_or_create(TEST_SENDER, "entity a")
        entity_b = penny.db.entities.get_or_create(TEST_SENDER, "entity b")
        assert entity_a is not None and entity_a.id is not None
        assert entity_b is not None and entity_b.id is not None

        penny.db.facts.add(entity_a.id, "Fact for A")
        penny.db.facts.add(entity_b.id, "Fact for B")

        agent = _create_notification_agent(penny, config)

        # Cycle 1: sends notification (picks one entity)
        signal_server.outgoing_messages.clear()
        result1 = await agent.execute()
        assert result1 is True

        # No engagement happens between cycles — user ignores the notification

        # Simulate user message to reset backoff (so cycle 2 can fire)
        msg_id2 = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="something"
        )
        penny.db.messages.mark_processed([msg_id2])

        # Cycle 2: should record NOTIFICATION_IGNORED for the entity from cycle 1
        await agent.execute()

        # Check that a NOTIFICATION_IGNORED engagement was created
        # The ignored engagement is for the FIRST notification's entity
        # (recorded at the start of cycle 2, before picking the new entity)
        all_user_engs = penny.db.engagements.get_for_user(TEST_SENDER)
        all_ignored = [
            e
            for e in all_user_engs
            if e.engagement_type == PennyConstants.EngagementType.NOTIFICATION_IGNORED
        ]
        assert len(all_ignored) >= 1


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
        msg_id = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="hello"
        )
        penny.db.messages.mark_processed([msg_id])

        entity = penny.db.entities.get_or_create(TEST_SENDER, "test entity")
        assert entity is not None and entity.id is not None
        penny.db.facts.add(entity.id, "Fact one")
        penny.db.facts.add(entity.id, "Fact two")

        # Verify facts start un-notified
        facts_before = penny.db.facts.get_for_entity(entity.id)
        assert all(f.notified_at is None for f in facts_before)

        agent = _create_notification_agent(penny, config)
        await agent.execute()

        # Facts should now be marked as notified
        facts_after = penny.db.facts.get_for_entity(entity.id)
        assert all(f.notified_at is not None for f in facts_after)

        # Entity should have last_notified_at set
        updated_entity = penny.db.entities.get(entity.id)
        assert updated_entity is not None
        assert updated_entity.last_notified_at is not None


@pytest.mark.asyncio
async def test_notification_entity_cooldown(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """After notifying about an entity, it enters cooldown and other entities are picked instead."""
    # Use small initial_backoff (50ms) so the second notification fires quickly after user message.
    # pool_size=1 makes selection deterministic (always picks highest score).
    config = make_config(notification_initial_backoff=0.05, notification_pool_size=1)

    def handler(request: dict, count: int) -> dict:
        return mock_ollama._make_text_response(
            request,
            "Here's an interesting discovery — some really great new facts about this topic!",
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        msg_id = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="hello"
        )
        penny.db.messages.mark_processed([msg_id])

        # Create two entities — give entity A higher interest so it's picked first
        entity_a = penny.db.entities.get_or_create(TEST_SENDER, "entity alpha")
        entity_b = penny.db.entities.get_or_create(TEST_SENDER, "entity beta")
        assert entity_a is not None and entity_a.id is not None
        assert entity_b is not None and entity_b.id is not None

        penny.db.engagements.add(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.USER_SEARCH,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=1.0,
            entity_id=entity_a.id,
        )

        penny.db.facts.add(entity_b.id, "Fact for beta")
        penny.db.facts.add(entity_a.id, "Fact for alpha")

        agent = _create_notification_agent(penny, config)

        # Cycle 1: entity A picked (highest interest score)
        signal_server.outgoing_messages.clear()
        result1 = await agent.execute()
        assert result1 is True
        # Verify entity A was notified (its fact is marked, its last_notified_at is set)
        facts_a = penny.db.facts.get_for_entity(entity_a.id)
        assert any(f.notified_at is not None for f in facts_a)
        entity_a_refreshed = penny.db.entities.get(entity_a.id)
        assert entity_a_refreshed is not None and entity_a_refreshed.last_notified_at is not None

        # Add new facts to both
        penny.db.facts.add(entity_b.id, "Another beta fact")
        penny.db.facts.add(entity_a.id, "Another alpha fact")

        # User sends message → resets backoff (to initial_backoff=50ms from interaction time)
        msg_id2 = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="thanks"
        )
        penny.db.messages.mark_processed([msg_id2])

        # Wait for initial_backoff (50ms) to elapse from interaction time
        interaction_recorded = datetime.now(UTC)
        await wait_until(lambda: (datetime.now(UTC) - interaction_recorded).total_seconds() >= 0.1)

        # Cycle 2: entity A is in cooldown, so entity B is picked instead
        signal_server.outgoing_messages.clear()
        result2 = await agent.execute()
        assert result2 is True
        # Verify entity B was notified this time (cooldown forced rotation)
        facts_b = penny.db.facts.get_for_entity(entity_b.id)
        assert any(f.notified_at is not None for f in facts_b)
        entity_b_refreshed = penny.db.entities.get(entity_b.id)
        assert entity_b_refreshed is not None and entity_b_refreshed.last_notified_at is not None


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
        msg_id = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="hello"
        )
        penny.db.messages.mark_processed([msg_id])

        # Create two entities with un-notified facts
        e1 = penny.db.entities.get_or_create(TEST_SENDER, "entity one")
        e2 = penny.db.entities.get_or_create(TEST_SENDER, "entity two")
        assert e1 is not None and e1.id is not None
        assert e2 is not None and e2.id is not None
        penny.db.facts.add(e1.id, "Fact for entity one")
        penny.db.facts.add(e2.id, "Fact for entity two")

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
        msg_id = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="hello"
        )
        penny.db.messages.mark_processed([msg_id])

        entity = penny.db.entities.get_or_create(TEST_SENDER, "pre-notified entity")
        assert entity is not None and entity.id is not None
        # Pre-mark as notified (simulates user-sourced facts)
        penny.db.facts.add(entity.id, "Already notified fact", notified_at=datetime.now(UTC))

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
        msg_id = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="hello"
        )
        penny.db.messages.mark_processed([msg_id])

        # Create the full chain: LearnPrompt → SearchLog → Fact
        learn_prompt = penny.db.learn_prompts.create(
            user=TEST_SENDER,
            prompt_text="audiophile gear",
            searches_remaining=0,
        )
        assert learn_prompt is not None and learn_prompt.id is not None

        penny.db.searches.log(
            query="audiophile gear",
            response="KEF LS50 Meta is a popular bookshelf speaker...",
            trigger="learn_command",
            learn_prompt_id=learn_prompt.id,
        )
        search_logs = penny.db.searches.get_by_learn_prompt(learn_prompt.id)
        assert len(search_logs) == 1

        entity = penny.db.entities.get_or_create(TEST_SENDER, "kef ls50 meta")
        assert entity is not None and entity.id is not None
        penny.db.facts.add(
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
    """Notification respects backoff: send, then suppress until initial_backoff expires.

    After user interaction, the next notification requires idle + initial_backoff
    before firing — not immediately on the first idle scheduler tick.
    """
    config = make_config()

    def handler(request: dict, count: int) -> dict:
        return mock_ollama._make_text_response(
            request,
            "Here's an interesting discovery — some really great new facts about this topic!",
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        msg_id = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="hello"
        )
        penny.db.messages.mark_processed([msg_id])

        agent = _create_notification_agent(penny, config)

        # --- Cycle 1: notification sent (no backoff — never acted before) ---
        e1 = penny.db.entities.get_or_create(TEST_SENDER, "backoff entity 1")
        assert e1 is not None and e1.id is not None
        penny.db.facts.add(e1.id, "Fact for backoff test 1")

        signal_server.outgoing_messages.clear()
        result1 = await agent.execute()
        assert result1 is True
        assert len(signal_server.outgoing_messages) == 1

        # --- Cycle 2: suppressed (backoff active, no user reply) ---
        e2 = penny.db.entities.get_or_create(TEST_SENDER, "backoff entity 2")
        assert e2 is not None and e2.id is not None
        penny.db.facts.add(e2.id, "Fact for backoff test 2")

        signal_server.outgoing_messages.clear()
        result2 = await agent.execute()
        assert result2 is False
        assert len(signal_server.outgoing_messages) == 0

        # --- User sends message → resets backoff ---
        msg_id2 = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="thanks!"
        )
        penny.db.messages.mark_processed([msg_id2])

        # Cycle 3: immediately after user message, still suppressed.
        # The fix: initial_backoff must elapse from interaction time before notification fires.
        signal_server.outgoing_messages.clear()
        result3 = await agent.execute()
        assert result3 is False
        assert len(signal_server.outgoing_messages) == 0


@pytest.mark.asyncio
async def test_notification_fires_after_initial_backoff_from_user_message(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """After user interaction, notification fires once initial_backoff has elapsed.

    Uses a small initial_backoff so the test completes quickly while still
    verifying the backoff fires correctly after the wait.
    """
    # Use a small initial_backoff (50ms) — reliably larger than Python/DB overhead
    # but small enough that wait_until (50ms poll) catches it in one or two ticks.
    config = make_config(notification_initial_backoff=0.05)

    def handler(request: dict, count: int) -> dict:
        return mock_ollama._make_text_response(
            request,
            "Here's an interesting discovery — some really great new facts about this topic!",
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Log and process initial message to establish a baseline interaction time
        msg_id = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="hello"
        )
        penny.db.messages.mark_processed([msg_id])

        agent = _create_notification_agent(penny, config)

        # --- Cycle 1: first notification fires (no prior state) ---
        e1 = penny.db.entities.get_or_create(TEST_SENDER, "entity for initial backoff test")
        assert e1 is not None and e1.id is not None
        penny.db.facts.add(e1.id, "Fact for initial backoff test")

        signal_server.outgoing_messages.clear()
        result1 = await agent.execute()
        assert result1 is True

        # --- User sends message → resets backoff (initial_backoff = 50ms from interaction) ---
        msg_id2 = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="thanks!"
        )
        penny.db.messages.mark_processed([msg_id2])

        # Add another entity/fact to notify about
        e2 = penny.db.entities.get_or_create(TEST_SENDER, "entity for initial backoff test 2")
        assert e2 is not None and e2.id is not None
        penny.db.facts.add(e2.id, "Fact for initial backoff test 2")

        # Immediately after interaction: suppressed (initial_backoff of 50ms not yet elapsed)
        signal_server.outgoing_messages.clear()
        result2_immediate = await agent.execute()
        assert result2_immediate is False

        # After initial_backoff (50ms) elapses from the interaction time, it fires.
        # wait_until polls every 50ms, so this resolves in at most 2 poll cycles.
        interaction_recorded = datetime.now(UTC)
        await wait_until(lambda: (datetime.now(UTC) - interaction_recorded).total_seconds() >= 0.1)
        signal_server.outgoing_messages.clear()
        result2_after = await agent.execute()
        assert result2_after is True
        assert len(signal_server.outgoing_messages) == 1


@pytest.mark.asyncio
async def test_notification_command_does_not_reset_backoff(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Commands (like /learn) should NOT reset notification backoff — only real messages do."""
    config = make_config()

    def handler(request: dict, count: int) -> dict:
        return mock_ollama._make_text_response(
            request,
            "Here's an interesting discovery — some really great new facts about this topic!",
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        msg_id = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="hello"
        )
        penny.db.messages.mark_processed([msg_id])

        agent = _create_notification_agent(penny, config)

        # --- Cycle 1: notification sent (no backoff) ---
        e1 = penny.db.entities.get_or_create(TEST_SENDER, "command backoff entity 1")
        assert e1 is not None and e1.id is not None
        penny.db.facts.add(e1.id, "Fact for command backoff test 1")

        signal_server.outgoing_messages.clear()
        result1 = await agent.execute()
        assert result1 is True

        # --- Cycle 2: suppressed (backoff active) ---
        e2 = penny.db.entities.get_or_create(TEST_SENDER, "command backoff entity 2")
        assert e2 is not None and e2.id is not None
        penny.db.facts.add(e2.id, "Fact for command backoff test 2")

        signal_server.outgoing_messages.clear()
        result2 = await agent.execute()
        assert result2 is False

        # --- User sends a command (not a message) → should NOT reset backoff ---
        penny.db.messages.log_command(
            user=TEST_SENDER,
            channel_type="signal",
            command_name="learn",
            command_args="kef speakers",
            response="Okay, I'll learn more about kef speakers",
        )

        # --- Cycle 3: still suppressed (command does not reset backoff) ---
        signal_server.outgoing_messages.clear()
        result3 = await agent.execute()
        assert result3 is False


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
        msg_id = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="hello"
        )
        penny.db.messages.mark_processed([msg_id])

        # Create a completed learn prompt with extracted search logs
        lp = penny.db.learn_prompts.create(
            user=TEST_SENDER,
            prompt_text="kef speakers",
            searches_remaining=0,
        )
        assert lp is not None and lp.id is not None
        penny.db.learn_prompts.update_status(lp.id, PennyConstants.LearnPromptStatus.COMPLETED)

        penny.db.searches.log(
            query="kef speakers overview",
            response="KEF makes great speakers...",
            trigger="learn_command",
            learn_prompt_id=lp.id,
        )
        search_logs = penny.db.searches.get_by_learn_prompt(lp.id)
        assert len(search_logs) == 1
        # Mark as extracted
        assert search_logs[0].id is not None
        penny.db.searches.mark_extracted(search_logs[0].id)

        # Create entity and fact linked to the search log
        entity = penny.db.entities.get_or_create(TEST_SENDER, "kef ls50 meta")
        assert entity is not None and entity.id is not None
        penny.db.engagements.add(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.USER_SEARCH,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=1.0,
            entity_id=entity.id,
        )
        penny.db.facts.add(
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
        updated_lp = penny.db.learn_prompts.get(lp.id)
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
        msg_id = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="hello"
        )
        penny.db.messages.mark_processed([msg_id])

        # Create a completed learn prompt with UN-extracted search log
        lp = penny.db.learn_prompts.create(
            user=TEST_SENDER,
            prompt_text="kef speakers",
            searches_remaining=0,
        )
        assert lp is not None and lp.id is not None
        penny.db.learn_prompts.update_status(lp.id, PennyConstants.LearnPromptStatus.COMPLETED)

        penny.db.searches.log(
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
        updated_lp = penny.db.learn_prompts.get(lp.id)
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
        msg_id = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="hello"
        )
        penny.db.messages.mark_processed([msg_id])

        lp = penny.db.learn_prompts.create(
            user=TEST_SENDER,
            prompt_text="kef speakers",
            searches_remaining=0,
        )
        assert lp is not None and lp.id is not None
        penny.db.learn_prompts.update_status(lp.id, PennyConstants.LearnPromptStatus.COMPLETED)

        penny.db.searches.log(
            query="kef speakers overview",
            response="KEF makes great speakers...",
            trigger="learn_command",
            learn_prompt_id=lp.id,
        )
        search_logs = penny.db.searches.get_by_learn_prompt(lp.id)
        assert search_logs[0].id is not None
        penny.db.searches.mark_extracted(search_logs[0].id)

        entity = penny.db.entities.get_or_create(TEST_SENDER, "kef ls50 meta")
        assert entity is not None and entity.id is not None
        penny.db.facts.add(
            entity.id,
            "KEF LS50 Meta costs $1,599",
            source_search_log_id=search_logs[0].id,
        )
        penny.db.facts.add(
            entity.id,
            "KEF LS50 Meta uses MAT technology",
            source_search_log_id=search_logs[0].id,
        )

        # Verify facts start un-notified
        facts_before = penny.db.facts.get_for_entity(entity.id)
        assert all(f.notified_at is None for f in facts_before)

        agent = _create_notification_agent(penny, config)
        await agent.execute()

        # Facts should now be marked as notified
        facts_after = penny.db.facts.get_for_entity(entity.id)
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
        msg_id = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="hello"
        )
        penny.db.messages.mark_processed([msg_id])

        # Create two completed learn prompts, both fully extracted
        for topic in ("kef speakers", "nvidia gpus"):
            lp = penny.db.learn_prompts.create(
                user=TEST_SENDER, prompt_text=topic, searches_remaining=0
            )
            assert lp is not None and lp.id is not None
            penny.db.learn_prompts.update_status(lp.id, PennyConstants.LearnPromptStatus.COMPLETED)
            penny.db.searches.log(
                query=f"{topic} overview",
                response=f"Info about {topic}...",
                trigger="learn_command",
                learn_prompt_id=lp.id,
            )
            search_logs = penny.db.searches.get_by_learn_prompt(lp.id)
            assert search_logs[0].id is not None
            penny.db.searches.mark_extracted(search_logs[0].id)

            entity = penny.db.entities.get_or_create(TEST_SENDER, topic)
            assert entity is not None and entity.id is not None
            penny.db.facts.add(
                entity.id, f"Fact about {topic}", source_search_log_id=search_logs[0].id
            )

        agent = _create_notification_agent(penny, config)

        # First cycle: only one announcement sent
        signal_server.outgoing_messages.clear()
        result = await agent.execute()
        assert result is True
        assert len(signal_server.outgoing_messages) == 1

        # One learn prompt announced, one still pending
        prompts = penny.db.learn_prompts.get_unannounced_completed(TEST_SENDER)
        assert len(prompts) == 1

        # Second cycle: the other announcement sent
        signal_server.outgoing_messages.clear()
        result = await agent.execute()
        assert result is True
        assert len(signal_server.outgoing_messages) == 1

        # Both now announced
        prompts = penny.db.learn_prompts.get_unannounced_completed(TEST_SENDER)
        assert len(prompts) == 0


@pytest.mark.asyncio
async def test_notification_skips_same_learn_topic_after_notifying(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """After notifying about entity B from learn topic X, skip entity A from same topic X.

    When two entities share the same learn_prompt_id, notifying about one should
    suppress notifications for the other on the very next cycle — the learn completion
    announcement will surface all entities from the topic together.
    """
    # pool_size=1 makes selection deterministic (always picks highest score)
    config = make_config(notification_initial_backoff=0.05, notification_pool_size=1)

    def handler(request: dict, count: int) -> dict:
        return mock_ollama._make_text_response(
            request,
            "Here's an interesting discovery — some really great new facts about this topic!",
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        msg_id = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="hello"
        )
        penny.db.messages.mark_processed([msg_id])

        # Create a learn prompt and two entities with facts from it
        lp = penny.db.learn_prompts.create(
            user=TEST_SENDER,
            prompt_text="audiophile speakers",
            searches_remaining=0,
        )
        assert lp is not None and lp.id is not None

        penny.db.searches.log(
            query="audiophile speakers",
            response="KEF and Focal make great speakers...",
            trigger="learn_command",
            learn_prompt_id=lp.id,
        )
        search_logs = penny.db.searches.get_by_learn_prompt(lp.id)
        assert len(search_logs) == 1
        sl_id = search_logs[0].id

        # Entity A: from this learn topic
        entity_a = penny.db.entities.get_or_create(TEST_SENDER, "kef ls50 meta")
        assert entity_a is not None and entity_a.id is not None
        penny.db.facts.add(
            entity_a.id, "KEF LS50 Meta uses MAT technology", source_search_log_id=sl_id
        )

        # Entity B: from the SAME learn topic, with higher interest so it's picked first
        entity_b = penny.db.entities.get_or_create(TEST_SENDER, "focal clear mg")
        assert entity_b is not None and entity_b.id is not None
        penny.db.engagements.add(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.USER_SEARCH,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=1.0,
            entity_id=entity_b.id,
        )
        penny.db.facts.add(
            entity_b.id, "Focal Clear MG uses magnesium drivers", source_search_log_id=sl_id
        )

        agent = _create_notification_agent(penny, config)

        # Cycle 1: entity B picked (highest interest score) and notified
        signal_server.outgoing_messages.clear()
        result1 = await agent.execute()
        assert result1 is True
        assert len(signal_server.outgoing_messages) == 1

        # Verify entity B was the one notified (highest score)
        facts_b = penny.db.facts.get_for_entity(entity_b.id)
        assert any(f.notified_at is not None for f in facts_b)

        # Wait for initial_backoff (50ms) to pass before next cycle
        interaction_recorded = datetime.now(UTC)
        await wait_until(lambda: (datetime.now(UTC) - interaction_recorded).total_seconds() >= 0.1)

        # Cycle 2: entity A shares the same learn topic as entity B (just notified)
        # → should be SKIPPED even though it has unnotified facts
        signal_server.outgoing_messages.clear()
        result2 = await agent.execute()
        assert result2 is False
        assert len(signal_server.outgoing_messages) == 0

        # Entity A still has unnotified facts (it was skipped, not notified)
        facts_a = penny.db.facts.get_for_entity(entity_a.id)
        assert all(f.notified_at is None for f in facts_a)


@pytest.mark.asyncio
async def test_event_notification_sends_and_marks_notified(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Unnotified events trigger a notification and get marked as notified."""
    config = make_config()

    def handler(request: dict, count: int) -> dict:
        return mock_ollama._make_text_response(
            request,
            "Heads up — **SpaceX** just launched their Starship rocket today!"
            " Pretty cool development worth keeping an eye on.",
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        msg_id = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="hello"
        )
        penny.db.messages.mark_processed([msg_id])

        # Create a follow prompt — events need one for per-prompt cadence
        fp = penny.db.follow_prompts.create(
            user=TEST_SENDER,
            prompt_text="space launches",
            query_terms='["spacex"]',
        )
        assert fp is not None and fp.id is not None

        # Create an unnotified event linked to the follow prompt
        event = penny.db.events.add(
            user=TEST_SENDER,
            headline="SpaceX launches Starship",
            summary="SpaceX successfully launched its Starship rocket today.",
            occurred_at=datetime.now(UTC),
            source_type=PennyConstants.EventSourceType.NEWS_API,
            source_url="https://example.com/spacex",
            external_id="https://example.com/spacex",
            follow_prompt_id=fp.id,
        )
        assert event is not None and event.id is not None

        agent = _create_notification_agent(penny, config)
        signal_server.outgoing_messages.clear()
        result = await agent.execute()

        assert result is True
        assert len(signal_server.outgoing_messages) == 1
        assert "SpaceX" in signal_server.outgoing_messages[0]["message"]

        # Event should be marked as notified
        updated = penny.db.events.get(event.id)
        assert updated is not None
        assert updated.notified_at is not None

        # Follow prompt's last_notified_at should be set
        updated_fp = penny.db.follow_prompts.get(fp.id)
        assert updated_fp is not None
        assert updated_fp.last_notified_at is not None


@pytest.mark.asyncio
async def test_event_notification_respects_cadence(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Event notifications respect per-prompt cadence — suppressed until cadence elapses."""
    config = make_config()

    def handler(request: dict, count: int) -> dict:
        return mock_ollama._make_text_response(
            request,
            "Here's a heads up about some interesting news I saw recently!"
            " This looks like something you'd want to know about.",
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        msg_id = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="hello"
        )
        penny.db.messages.mark_processed([msg_id])

        # Create follow prompt with daily cadence
        fp = penny.db.follow_prompts.create(
            user=TEST_SENDER,
            prompt_text="tech news",
            query_terms='["tech"]',
            cadence="daily",
        )
        assert fp is not None and fp.id is not None

        agent = _create_notification_agent(penny, config)

        # Create and send first event notification
        penny.db.events.add(
            user=TEST_SENDER,
            headline="Event one",
            summary="First event.",
            occurred_at=datetime.now(UTC),
            source_type=PennyConstants.EventSourceType.NEWS_API,
            source_url="https://example.com/1",
            external_id="https://example.com/1",
            follow_prompt_id=fp.id,
        )

        signal_server.outgoing_messages.clear()
        result1 = await agent.execute()
        assert result1 is True

        # Second event — should be suppressed by cadence (daily = 86400s)
        penny.db.events.add(
            user=TEST_SENDER,
            headline="Event two",
            summary="Second event.",
            occurred_at=datetime.now(UTC),
            source_type=PennyConstants.EventSourceType.NEWS_API,
            source_url="https://example.com/2",
            external_id="https://example.com/2",
            follow_prompt_id=fp.id,
        )

        signal_server.outgoing_messages.clear()
        result2 = await agent.execute()
        # Event notification suppressed, but fact notification may or may not fire
        # (there are no unnotified facts here, so it's False)
        assert result2 is False


@pytest.mark.asyncio
async def test_event_and_fact_notifications_independent(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Event and fact notifications fire independently in the same cycle."""
    config = make_config()

    def handler(request: dict, count: int) -> dict:
        return mock_ollama._make_text_response(
            request,
            "Here's what I noticed — some really interesting recent news about this topic!"
            " Thought you'd want to know about these developments.",
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        msg_id = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="hello"
        )
        penny.db.messages.mark_processed([msg_id])

        # Create a follow prompt with an unnotified event
        fp = penny.db.follow_prompts.create(
            user=TEST_SENDER,
            prompt_text="tech news",
            query_terms='["tech"]',
        )
        assert fp is not None and fp.id is not None

        penny.db.events.add(
            user=TEST_SENDER,
            headline="Breaking: test entity news",
            summary="Something happened about test entity.",
            occurred_at=datetime.now(UTC),
            source_type=PennyConstants.EventSourceType.NEWS_API,
            source_url="https://example.com/event",
            external_id="https://example.com/event",
            follow_prompt_id=fp.id,
        )

        # Also create an unnotified fact
        entity = penny.db.entities.get_or_create(TEST_SENDER, "test entity")
        assert entity is not None and entity.id is not None
        penny.db.facts.add(entity.id, "Some new fact about test entity")

        agent = _create_notification_agent(penny, config)
        signal_server.outgoing_messages.clear()
        result = await agent.execute()

        assert result is True
        # Both event and fact notifications should fire independently
        assert len(signal_server.outgoing_messages) == 2


@pytest.mark.asyncio
async def test_notification_neighbor_boost_shifts_selection(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Embedding neighbor boost lifts entities near positively-engaged neighbors.

    Entity A has low base interest but its embedding is close to a neighbor
    entity with strong positive engagement. Entity B has slightly higher base
    interest but no engaged neighbors. With neighbor boost, A should outscore B.
    """
    config = make_config(notification_pool_size=1)

    captured_prompts: list[str] = []

    def handler(request: dict, count: int) -> dict:
        messages = request.get("messages", [])
        prompt = messages[-1]["content"] if messages else ""
        captured_prompts.append(prompt)
        if "boosted" in prompt:
            return mock_ollama._make_text_response(
                request,
                "Hey, I found something really interesting about boosted entity"
                " that I think you would enjoy hearing about!",
            )
        return mock_ollama._make_text_response(
            request,
            "Hey, I found something really interesting about baseline entity"
            " that I think you would enjoy hearing about!",
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        msg_id = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="hello"
        )
        penny.db.messages.mark_processed([msg_id])

        # Neighbor entity: strong positive engagement, similar embedding to A
        neighbor = penny.db.entities.get_or_create(TEST_SENDER, "neighbor entity")
        assert neighbor is not None and neighbor.id is not None
        penny.db.engagements.add(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.EMOJI_REACTION,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.8,
            entity_id=neighbor.id,
        )
        # Embedding: [1, 0, 0, 0] — close to entity A
        penny.db.entities.update_embedding(neighbor.id, serialize_embedding([1.0, 0.0, 0.0, 0.0]))

        # Entity A: low base interest, but embedding [0.9, 0.1, 0, 0] is
        # very similar to neighbor (cosine ~0.99)
        entity_a = penny.db.entities.get_or_create(TEST_SENDER, "boosted entity")
        assert entity_a is not None and entity_a.id is not None
        penny.db.entities.update_embedding(entity_a.id, serialize_embedding([0.9, 0.1, 0.0, 0.0]))
        penny.db.facts.add(entity_a.id, "Fact about boosted entity")

        # Entity B: slightly higher base interest (follow-up question),
        # but embedding [0, 0, 1, 0] is far from any engaged entity
        entity_b = penny.db.entities.get_or_create(TEST_SENDER, "baseline entity")
        assert entity_b is not None and entity_b.id is not None
        penny.db.engagements.add(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.FOLLOW_UP_QUESTION,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.1,
            entity_id=entity_b.id,
        )
        penny.db.entities.update_embedding(entity_b.id, serialize_embedding([0.0, 0.0, 1.0, 0.0]))
        penny.db.facts.add(entity_b.id, "Fact about baseline entity")

        agent = _create_notification_agent(penny, config)
        signal_server.outgoing_messages.clear()
        result = await agent.execute()
        assert result is True

        # Entity A (boosted) should be picked over entity B (baseline)
        msgs = signal_server.outgoing_messages
        assert len(msgs) == 1
        assert "boosted entity" in msgs[0]["message"].lower()

"""Integration tests for the NotificationAgent."""

from datetime import UTC, datetime, timedelta

import pytest

from penny.constants import PennyConstants
from penny.tests.conftest import TEST_SENDER, wait_until


def _set_entity_heat(penny, entity_id, heat):
    """Set an entity's heat so it's eligible for notification."""
    penny.db.entities.update_heat(entity_id, heat)


@pytest.mark.asyncio
async def test_notification_prefers_higher_heat_entity(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Notification agent picks the entity with higher heat deterministically."""
    config = make_config()

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

        # Create two entities — one with high heat, one with low heat
        boring_entity = penny.db.entities.get_or_create(TEST_SENDER, "boring entity")
        interesting_entity = penny.db.entities.get_or_create(TEST_SENDER, "interesting entity")
        assert boring_entity is not None and boring_entity.id is not None
        assert interesting_entity is not None and interesting_entity.id is not None

        # Give interesting_entity higher heat
        _set_entity_heat(penny, boring_entity.id, 1.0)
        _set_entity_heat(penny, interesting_entity.id, 5.0)

        # Both get one fact
        penny.db.facts.add(boring_entity.id, "Boring fact")
        penny.db.facts.add(interesting_entity.id, "Interesting fact")

        signal_server.outgoing_messages.clear()
        result = await penny.notification_agent.execute()
        assert result is True

        # Should notify about the interesting entity (higher heat)
        msgs = signal_server.outgoing_messages
        assert len(msgs) == 1
        assert "interesting entity" in msgs[0]["message"]

        # Prompt sent to model should instruct it to synthesize, not echo raw facts
        assert any("Synthesize" in p for p in captured_prompts)


@pytest.mark.asyncio
async def test_notification_skips_zero_heat_entity(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Entity with zero heat (e.g., vetoed) is excluded from notifications.

    A thumbs-down zeroes heat, so the entity naturally sinks to ineligible.
    The fallback entity (with heat) gets picked instead.
    """
    config = make_config()

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

        # Create two entities — vetoed one has zero heat
        vetoed = penny.db.entities.get_or_create(TEST_SENDER, "vetoed entity")
        fallback = penny.db.entities.get_or_create(TEST_SENDER, "fallback entity")
        assert vetoed is not None and vetoed.id is not None
        assert fallback is not None and fallback.id is not None

        # Vetoed entity: heat = 0 (as if thumbs-downed)
        _set_entity_heat(penny, vetoed.id, 0.0)
        _set_entity_heat(penny, fallback.id, 3.0)

        penny.db.facts.add(vetoed.id, "Vetoed fact")
        penny.db.facts.add(fallback.id, "Fallback fact")

        signal_server.outgoing_messages.clear()
        result = await penny.notification_agent.execute()
        assert result is True

        # Should pick the fallback, not the vetoed entity
        msgs = signal_server.outgoing_messages
        assert len(msgs) == 1
        assert "fallback entity" in msgs[0]["message"].lower()


@pytest.mark.asyncio
async def test_notification_ignore_penalty_reduces_heat(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """When a notification gets no engagement, the next cycle penalizes heat.

    First cycle: sends notification for entity A.
    Second cycle: no engagement since → heat penalty applied to A.
    """
    config = make_config(
        notification_initial_backoff=0,
        HEAT_COOLDOWN_CYCLES=0,
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

        _set_entity_heat(penny, entity_a.id, 5.0)
        _set_entity_heat(penny, entity_b.id, 3.0)

        penny.db.facts.add(entity_a.id, "Fact for A")
        penny.db.facts.add(entity_b.id, "Fact for B")

        agent = penny.notification_agent

        # Cycle 1: sends notification (picks entity A — higher heat)
        signal_server.outgoing_messages.clear()
        result1 = await agent.execute()
        assert result1 is True

        # Record A's heat before ignore penalty
        heat_before = penny.db.entities.get(entity_a.id).heat

        # No engagement happens between cycles — user ignores the notification

        # Simulate user message to reset backoff (so cycle 2 can fire)
        msg_id2 = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="something"
        )
        penny.db.messages.mark_processed([msg_id2])

        # Add new facts so there's something to notify about
        penny.db.facts.add(entity_a.id, "New fact for A")
        penny.db.facts.add(entity_b.id, "New fact for B")

        # Cycle 2: should penalize heat for ignored entity A
        await agent.execute()

        # Entity A's heat should have been reduced by ignore penalty
        heat_after = penny.db.entities.get(entity_a.id).heat
        assert heat_after < heat_before


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
        _set_entity_heat(penny, entity.id, 5.0)
        penny.db.facts.add(entity.id, "Fact one")
        penny.db.facts.add(entity.id, "Fact two")

        # Verify facts start un-notified
        facts_before = penny.db.facts.get_for_entity(entity.id)
        assert all(f.notified_at is None for f in facts_before)

        await penny.notification_agent.execute()

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
    config = make_config(notification_initial_backoff=0.05)

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

        # Create two entities — give entity A higher heat so it's picked first
        entity_a = penny.db.entities.get_or_create(TEST_SENDER, "entity alpha")
        entity_b = penny.db.entities.get_or_create(TEST_SENDER, "entity beta")
        assert entity_a is not None and entity_a.id is not None
        assert entity_b is not None and entity_b.id is not None

        _set_entity_heat(penny, entity_a.id, 8.0)
        _set_entity_heat(penny, entity_b.id, 3.0)

        penny.db.facts.add(entity_b.id, "Fact for beta")
        penny.db.facts.add(entity_a.id, "Fact for alpha")

        agent = penny.notification_agent

        # Cycle 1: entity A picked (highest heat)
        signal_server.outgoing_messages.clear()
        result1 = await agent.execute()
        assert result1 is True
        # Verify entity A was notified and has cooldown set
        facts_a = penny.db.facts.get_for_entity(entity_a.id)
        assert any(f.notified_at is not None for f in facts_a)
        entity_a_refreshed = penny.db.entities.get(entity_a.id)
        assert entity_a_refreshed is not None
        assert entity_a_refreshed.heat_cooldown > 0

        # Add new facts to both
        penny.db.facts.add(entity_b.id, "Another beta fact")
        penny.db.facts.add(entity_a.id, "Another alpha fact")

        # User sends message → resets backoff
        msg_id2 = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="thanks"
        )
        penny.db.messages.mark_processed([msg_id2])

        # Wait for initial_backoff (50ms) to elapse
        interaction_recorded = datetime.now(UTC)
        await wait_until(lambda: (datetime.now(UTC) - interaction_recorded).total_seconds() >= 0.1)

        # Cycle 2: entity A is on cooldown, so entity B is picked instead
        signal_server.outgoing_messages.clear()
        result2 = await agent.execute()
        assert result2 is True
        # Verify entity B was notified this time (cooldown forced rotation)
        facts_b = penny.db.facts.get_for_entity(entity_b.id)
        assert any(f.notified_at is not None for f in facts_b)


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
        _set_entity_heat(penny, e1.id, 5.0)
        _set_entity_heat(penny, e2.id, 3.0)
        penny.db.facts.add(e1.id, "Fact for entity one")
        penny.db.facts.add(e2.id, "Fact for entity two")

        signal_server.outgoing_messages.clear()
        await penny.notification_agent.execute()

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
        _set_entity_heat(penny, entity.id, 5.0)
        # Pre-mark as notified (simulates user-sourced facts)
        penny.db.facts.add(entity.id, "Already notified fact", notified_at=datetime.now(UTC))

        signal_server.outgoing_messages.clear()
        result = await penny.notification_agent.execute()

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
        _set_entity_heat(penny, entity.id, 5.0)
        penny.db.facts.add(
            entity.id,
            "KEF LS50 Meta uses Metamaterial Absorption Technology",
            source_search_log_id=search_logs[0].id,
        )

        signal_server.outgoing_messages.clear()
        result = await penny.notification_agent.execute()
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

        agent = penny.notification_agent

        # --- Cycle 1: notification sent (no backoff — never acted before) ---
        e1 = penny.db.entities.get_or_create(TEST_SENDER, "backoff entity 1")
        assert e1 is not None and e1.id is not None
        _set_entity_heat(penny, e1.id, 5.0)
        penny.db.facts.add(e1.id, "Fact for backoff test 1")

        signal_server.outgoing_messages.clear()
        result1 = await agent.execute()
        assert result1 is True
        assert len(signal_server.outgoing_messages) == 1

        # --- Cycle 2: suppressed (backoff active, no user reply) ---
        e2 = penny.db.entities.get_or_create(TEST_SENDER, "backoff entity 2")
        assert e2 is not None and e2.id is not None
        _set_entity_heat(penny, e2.id, 5.0)
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
    """After user interaction, notification fires once initial_backoff has elapsed."""
    config = make_config(notification_initial_backoff=0.05)

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

        agent = penny.notification_agent

        # --- Cycle 1: first notification fires (no prior state) ---
        e1 = penny.db.entities.get_or_create(TEST_SENDER, "entity for initial backoff test")
        assert e1 is not None and e1.id is not None
        _set_entity_heat(penny, e1.id, 5.0)
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
        _set_entity_heat(penny, e2.id, 5.0)
        penny.db.facts.add(e2.id, "Fact for initial backoff test 2")

        # Immediately after interaction: suppressed (initial_backoff of 50ms not yet elapsed)
        signal_server.outgoing_messages.clear()
        result2_immediate = await agent.execute()
        assert result2_immediate is False

        # After initial_backoff (50ms) elapses from the interaction time, it fires.
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

        agent = penny.notification_agent

        # --- Cycle 1: notification sent (no backoff) ---
        e1 = penny.db.entities.get_or_create(TEST_SENDER, "command backoff entity 1")
        assert e1 is not None and e1.id is not None
        _set_entity_heat(penny, e1.id, 5.0)
        penny.db.facts.add(e1.id, "Fact for command backoff test 1")

        signal_server.outgoing_messages.clear()
        result1 = await agent.execute()
        assert result1 is True

        # --- Cycle 2: suppressed (backoff active) ---
        e2 = penny.db.entities.get_or_create(TEST_SENDER, "command backoff entity 2")
        assert e2 is not None and e2.id is not None
        _set_entity_heat(penny, e2.id, 5.0)
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
    """At max backoff, _mark_proactive_sent doubles (clamped at max), not reset to initial."""
    from datetime import timedelta

    from penny.agents.backoff import BackoffState

    config = make_config()
    max_backoff = config.runtime.NOTIFICATION_MAX_BACKOFF
    initial_backoff = config.runtime.NOTIFICATION_INITIAL_BACKOFF

    async with running_penny(config) as penny:
        agent = penny.notification_agent

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
        search_log_id = search_logs[0].id
        assert search_log_id is not None
        penny.db.searches.mark_extracted(search_log_id)

        entity = penny.db.entities.get_or_create(TEST_SENDER, "kef ls50 meta")
        assert entity is not None and entity.id is not None
        penny.db.facts.add(
            entity.id,
            "KEF LS50 Meta uses Metamaterial Absorption Technology",
            source_search_log_id=search_log_id,
        )

        signal_server.outgoing_messages.clear()
        result = await penny.notification_agent.execute()
        assert result is True

        msgs = signal_server.outgoing_messages
        assert len(msgs) == 1
        assert "kef" in msgs[0]["message"].lower()

        # Facts from the learn prompt should be marked as notified
        facts = penny.db.facts.get_for_entity(entity.id)
        assert all(f.notified_at is not None for f in facts)

        # Learn prompt should be marked as announced
        refreshed_lp = penny.db.learn_prompts.get(lp.id)
        assert refreshed_lp is not None and refreshed_lp.announced_at is not None


@pytest.mark.asyncio
async def test_learn_completion_exclusive(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Learn completion blocks other notifications in the same cycle."""
    config = make_config()

    def handler(request: dict, count: int) -> dict:
        return mock_ollama._make_text_response(
            request,
            "I finished researching kef speakers — great find!",
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        msg_id = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="hello"
        )
        penny.db.messages.mark_processed([msg_id])

        # Create a completed learn prompt
        lp = penny.db.learn_prompts.create(
            user=TEST_SENDER,
            prompt_text="kef speakers",
            searches_remaining=0,
        )
        assert lp is not None and lp.id is not None
        penny.db.learn_prompts.update_status(lp.id, PennyConstants.LearnPromptStatus.COMPLETED)

        penny.db.searches.log(
            query="kef speakers",
            response="KEF makes speakers...",
            trigger="learn_command",
            learn_prompt_id=lp.id,
        )
        search_logs = penny.db.searches.get_by_learn_prompt(lp.id)
        assert search_logs and search_logs[0].id is not None
        penny.db.searches.mark_extracted(search_logs[0].id)

        entity_learn = penny.db.entities.get_or_create(TEST_SENDER, "kef ls50")
        assert entity_learn is not None and entity_learn.id is not None
        penny.db.facts.add(
            entity_learn.id,
            "KEF fact",
            source_search_log_id=search_logs[0].id,
        )

        # Also create a regular entity with an unnotified fact
        entity_regular = penny.db.entities.get_or_create(TEST_SENDER, "regular entity")
        assert entity_regular is not None and entity_regular.id is not None
        _set_entity_heat(penny, entity_regular.id, 5.0)
        penny.db.facts.add(entity_regular.id, "Regular fact")

        signal_server.outgoing_messages.clear()
        result = await penny.notification_agent.execute()
        assert result is True

        # Only ONE message sent (learn completion is exclusive)
        assert len(signal_server.outgoing_messages) == 1


@pytest.mark.asyncio
async def test_event_digest_marks_all_notified(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Event digest sends one message and marks all unnotified events as notified."""
    config = make_config()

    def handler(request: dict, count: int) -> dict:
        return mock_ollama._make_text_response(
            request,
            "Here's your latest update on **space launches** — "
            "SpaceX launched Starship, NASA updated Artemis, and Blue Origin tested New Glenn!",
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        msg_id = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="hello"
        )
        penny.db.messages.mark_processed([msg_id])

        # Create follow prompt with cron that's overdue
        fp = penny.db.follow_prompts.create(
            user=TEST_SENDER,
            prompt_text="space launches",
            query_terms='["spacex"]',
            cron_expression="0 * * * *",
            timing_description="hourly",
        )
        assert fp is not None and fp.id is not None

        # Create 3 unnotified events for this follow prompt
        for i, headline in enumerate(
            ["SpaceX launches Starship", "NASA Artemis update", "Blue Origin tests New Glenn"]
        ):
            penny.db.events.add(
                user=TEST_SENDER,
                headline=headline,
                summary=f"Details about {headline}.",
                occurred_at=datetime.now(UTC) - timedelta(hours=i),
                source_type=PennyConstants.EventSourceType.NEWS_API,
                source_url=f"https://example.com/story-{i}",
                external_id=f"https://example.com/story-{i}",
                follow_prompt_id=fp.id,
            )

        # Verify events start unnotified
        unnotified = penny.db.events.get_unnotified_for_follow_prompt(fp.id)
        assert len(unnotified) == 3

        signal_server.outgoing_messages.clear()
        result = await penny.notification_agent.execute()
        assert result is True

        # One digest message sent (not 3 individual ones)
        assert len(signal_server.outgoing_messages) == 1

        # All 3 events marked as notified
        still_unnotified = penny.db.events.get_unnotified_for_follow_prompt(fp.id)
        assert len(still_unnotified) == 0

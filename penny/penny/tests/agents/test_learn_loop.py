"""Integration tests for the LearnLoopAgent."""

import json

import pytest

from penny.agents.learn_loop import LearnLoopAgent
from penny.constants import PennyConstants
from penny.tests.conftest import TEST_SENDER


@pytest.mark.asyncio
async def test_learn_loop_enrichment(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Entity with few facts and positive interest → enrichment search."""
    config = make_config()

    def handler(request: dict, count: int) -> dict:
        # Fact extraction call — return structured JSON
        return mock_ollama._make_text_response(
            request,
            json.dumps({"facts": ["Costs $1,599 per pair", "Uses MAT driver technology"]}),
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Send a message to create a sender
        await signal_server.push_message(sender=TEST_SENDER, content="hello")
        await signal_server.wait_for_message(timeout=10.0)

        # Create entity with positive interest
        entity = penny.db.get_or_create_entity(TEST_SENDER, "kef ls50 meta")
        assert entity is not None and entity.id is not None
        penny.db.add_engagement(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.LEARN_COMMAND,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=PennyConstants.ENGAGEMENT_STRENGTH_LEARN_COMMAND,
            entity_id=entity.id,
        )

        # Clear previous mock state
        mock_ollama.requests.clear()

        # Create and run learn loop agent
        agent = LearnLoopAgent(
            search_tool=penny.message_agent.tools[0] if penny.message_agent.tools else None,
            system_prompt="",
            background_model_client=penny.background_model_client,
            foreground_model_client=penny.foreground_model_client,
            tools=[],
            db=penny.db,
            config=config,
            max_steps=1,
            tool_timeout=config.tool_timeout,
        )

        result = await agent.execute()
        assert result is True

        # Verify facts were stored with notified_at=NULL (notification agent's job)
        facts = penny.db.get_entity_facts(entity.id)
        assert len(facts) >= 1
        fact_texts = [f.content for f in facts]
        assert any("1,599" in t for t in fact_texts)
        assert all(f.notified_at is None for f in facts)

        # Verify search was tagged as penny_enrichment
        search_logs = penny.db.get_unprocessed_search_logs(limit=10)
        assert len(search_logs) >= 1
        assert search_logs[0].trigger == PennyConstants.SearchTrigger.PENNY_ENRICHMENT


@pytest.mark.asyncio
async def test_learn_loop_skips_negative_interest(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Entity with negative interest is skipped."""
    config = make_config()
    mock_ollama.set_default_flow(
        search_query="test",
        final_response="test response",
    )

    async with running_penny(config) as penny:
        # Create sender
        await signal_server.push_message(sender=TEST_SENDER, content="hello")
        await signal_server.wait_for_message(timeout=10.0)

        # Create entity with negative interest (thumbs down on notification)
        entity = penny.db.get_or_create_entity(TEST_SENDER, "sports")
        assert entity is not None and entity.id is not None
        penny.db.add_engagement(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.EMOJI_REACTION,
            valence=PennyConstants.EngagementValence.NEGATIVE,
            strength=PennyConstants.ENGAGEMENT_STRENGTH_EMOJI_REACTION_PROACTIVE_NEGATIVE,
            entity_id=entity.id,
        )

        agent = LearnLoopAgent(
            search_tool=penny.message_agent.tools[0] if penny.message_agent.tools else None,
            system_prompt="",
            background_model_client=penny.background_model_client,
            foreground_model_client=penny.foreground_model_client,
            tools=[],
            db=penny.db,
            config=config,
            max_steps=1,
            tool_timeout=config.tool_timeout,
        )

        result = await agent.execute()
        assert result is False


@pytest.mark.asyncio
async def test_learn_loop_no_entities(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """No entities for any user → returns False."""
    config = make_config()

    async with running_penny(config) as penny:
        agent = LearnLoopAgent(
            search_tool=penny.message_agent.tools[0] if penny.message_agent.tools else None,
            system_prompt="",
            background_model_client=penny.background_model_client,
            foreground_model_client=penny.foreground_model_client,
            tools=[],
            db=penny.db,
            config=config,
            max_steps=1,
            tool_timeout=config.tool_timeout,
        )

        result = await agent.execute()
        assert result is False


@pytest.mark.asyncio
async def test_learn_loop_no_search_tool(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Returns False when no search tool is configured."""
    config = make_config()

    async with running_penny(config) as penny:
        agent = LearnLoopAgent(
            search_tool=None,
            system_prompt="",
            background_model_client=penny.background_model_client,
            foreground_model_client=penny.foreground_model_client,
            tools=[],
            db=penny.db,
            config=config,
            max_steps=1,
            tool_timeout=config.tool_timeout,
        )

        result = await agent.execute()
        assert result is False


@pytest.mark.asyncio
async def test_learn_loop_dedup_facts(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """New facts that match existing ones are deduplicated."""
    config = make_config()

    def handler(request: dict, count: int) -> dict:
        messages = request.get("messages", [])
        last_content = messages[-1].get("content", "") if messages else ""

        if "Extract specific" in last_content:
            # Return a mix of existing and new facts
            return mock_ollama._make_text_response(
                request,
                json.dumps(
                    {
                        "facts": [
                            "Costs $1,599 per pair",  # duplicate of existing
                            "Won What Hi-Fi award in 2023",  # new
                        ]
                    }
                ),
            )
        else:
            return mock_ollama._make_text_response(request, "The KEF LS50 Meta just won an award!")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Create sender
        await signal_server.push_message(sender=TEST_SENDER, content="hello")
        await signal_server.wait_for_message(timeout=10.0)

        # Create entity with existing fact and positive interest
        entity = penny.db.get_or_create_entity(TEST_SENDER, "kef ls50 meta")
        assert entity is not None and entity.id is not None
        existing_fact = penny.db.add_fact(entity_id=entity.id, content="Costs $1,599 per pair")
        assert existing_fact is not None

        penny.db.add_engagement(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.LEARN_COMMAND,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=PennyConstants.ENGAGEMENT_STRENGTH_LEARN_COMMAND,
            entity_id=entity.id,
        )

        mock_ollama.requests.clear()

        agent = LearnLoopAgent(
            search_tool=penny.message_agent.tools[0] if penny.message_agent.tools else None,
            system_prompt="",
            background_model_client=penny.background_model_client,
            foreground_model_client=penny.foreground_model_client,
            tools=[],
            db=penny.db,
            config=config,
            max_steps=1,
            tool_timeout=config.tool_timeout,
        )

        result = await agent.execute()
        assert result is True

        # Should have 2 facts total (1 existing + 1 new), not 3
        facts = penny.db.get_entity_facts(entity.id)
        assert len(facts) == 2
        fact_texts = [f.content for f in facts]
        assert "Costs $1,599 per pair" in fact_texts
        assert "Won What Hi-Fi award in 2023" in fact_texts


@pytest.mark.asyncio
async def test_learn_loop_semantic_interest_priority(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Entity with higher SEARCH_DISCOVERY strength is prioritized.

    Two entities with identical SEARCH_INITIATED engagement — entity A has a
    SEARCH_DISCOVERY engagement with high strength (0.9), entity B with low
    strength (0.6). Entity A should be selected first.
    """
    config = make_config()

    def handler(request: dict, count: int) -> dict:
        return mock_ollama._make_text_response(request, json.dumps({"facts": ["Some new fact"]}))

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Create sender
        await signal_server.push_message(sender=TEST_SENDER, content="hello")
        await signal_server.wait_for_message(timeout=10.0)

        # Create two entities with identical SEARCH_INITIATED engagement
        entity_a = penny.db.get_or_create_entity(TEST_SENDER, "aamas")
        entity_b = penny.db.get_or_create_entity(TEST_SENDER, "coral beach hotel")
        assert entity_a is not None and entity_a.id is not None
        assert entity_b is not None and entity_b.id is not None

        for eid in (entity_a.id, entity_b.id):
            penny.db.add_engagement(
                user=TEST_SENDER,
                engagement_type=PennyConstants.EngagementType.SEARCH_INITIATED,
                valence=PennyConstants.EngagementValence.POSITIVE,
                strength=PennyConstants.ENGAGEMENT_STRENGTH_SEARCH_INITIATED,
                entity_id=eid,
            )

        # Entity A gets high semantic relevance, entity B gets low
        penny.db.add_engagement(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.SEARCH_DISCOVERY,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.9,
            entity_id=entity_a.id,
        )
        penny.db.add_engagement(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.SEARCH_DISCOVERY,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.6,
            entity_id=entity_b.id,
        )

        # Both get one fact so they're eligible for enrichment
        penny.db.add_fact(entity_id=entity_a.id, content="AAMAS 2026 is held in Cyprus")
        penny.db.add_fact(entity_id=entity_b.id, content="Coral Beach Hotel hosts AAMAS 2026")

        mock_ollama.requests.clear()

        agent = LearnLoopAgent(
            search_tool=penny.message_agent.tools[0] if penny.message_agent.tools else None,
            system_prompt="",
            background_model_client=penny.background_model_client,
            foreground_model_client=penny.foreground_model_client,
            tools=[],
            db=penny.db,
            config=config,
            max_steps=1,
            tool_timeout=config.tool_timeout,
        )

        result = await agent.execute()
        assert result is True

        # Entity A (aamas) should be selected because its SEARCH_DISCOVERY
        # engagement has higher strength (0.9 vs 0.6).
        # Verify by checking which entity got new facts stored.
        facts_a = penny.db.get_entity_facts(entity_a.id)
        facts_b = penny.db.get_entity_facts(entity_b.id)
        assert len(facts_a) > 1, (
            f"Expected entity A (aamas) to receive new facts from enrichment, "
            f"but it still has {len(facts_a)} fact(s)"
        )
        assert len(facts_b) == 1, (
            f"Expected entity B (coral beach hotel) to remain at 1 fact, "
            f"but it has {len(facts_b)} fact(s)"
        )

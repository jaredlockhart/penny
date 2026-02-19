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

    # Mock: first call returns extracted facts JSON, second returns composed message
    call_count = [0]

    def handler(request: dict, count: int) -> dict:
        call_count[0] += 1
        messages = request.get("messages", [])
        last_content = messages[-1].get("content", "") if messages else ""

        if "Extract specific" in last_content or "ENTITY_FACT" in last_content:
            # Fact extraction call — return structured JSON
            return mock_ollama._make_text_response(
                request,
                json.dumps({"facts": ["Costs $1,599 per pair", "Uses MAT driver technology"]}),
            )
        else:
            # Message composition call
            return mock_ollama._make_text_response(
                request,
                "Hey! I just looked into kef ls50 meta — it costs $1,599 and uses MAT driver tech!",
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
            model=config.ollama_background_model,
            ollama_api_url=config.ollama_api_url,
            tools=[],
            db=penny.db,
            max_steps=1,
            max_retries=config.ollama_max_retries,
            retry_delay=config.ollama_retry_delay,
            tool_timeout=config.tool_timeout,
        )
        agent.set_channel(penny.channel)

        result = await agent.execute()
        assert result is True

        # Verify facts were stored
        facts = penny.db.get_entity_facts(entity.id)
        assert len(facts) >= 1
        fact_texts = [f.content for f in facts]
        assert any("1,599" in t for t in fact_texts)

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

        # Create entity with negative interest (dislike)
        entity = penny.db.get_or_create_entity(TEST_SENDER, "sports")
        assert entity is not None and entity.id is not None
        penny.db.add_engagement(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.DISLIKE_COMMAND,
            valence=PennyConstants.EngagementValence.NEGATIVE,
            strength=PennyConstants.ENGAGEMENT_STRENGTH_DISLIKE_COMMAND,
            entity_id=entity.id,
        )

        agent = LearnLoopAgent(
            search_tool=penny.message_agent.tools[0] if penny.message_agent.tools else None,
            system_prompt="",
            model=config.ollama_background_model,
            ollama_api_url=config.ollama_api_url,
            tools=[],
            db=penny.db,
            max_steps=1,
            max_retries=config.ollama_max_retries,
            retry_delay=config.ollama_retry_delay,
            tool_timeout=config.tool_timeout,
        )
        agent.set_channel(penny.channel)

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
            model=config.ollama_background_model,
            ollama_api_url=config.ollama_api_url,
            tools=[],
            db=penny.db,
            max_steps=1,
            max_retries=config.ollama_max_retries,
            retry_delay=config.ollama_retry_delay,
            tool_timeout=config.tool_timeout,
        )
        agent.set_channel(penny.channel)

        result = await agent.execute()
        assert result is False


@pytest.mark.asyncio
async def test_learn_loop_no_channel(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Returns False when no channel is set."""
    config = make_config()

    async with running_penny(config) as penny:
        agent = LearnLoopAgent(
            search_tool=penny.message_agent.tools[0] if penny.message_agent.tools else None,
            system_prompt="",
            model=config.ollama_background_model,
            ollama_api_url=config.ollama_api_url,
            tools=[],
            db=penny.db,
            max_steps=1,
            max_retries=config.ollama_max_retries,
            retry_delay=config.ollama_retry_delay,
            tool_timeout=config.tool_timeout,
        )
        # Don't set channel

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
    """New facts that match existing ones are deduplicated and last_verified is updated."""
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
            model=config.ollama_background_model,
            ollama_api_url=config.ollama_api_url,
            tools=[],
            db=penny.db,
            max_steps=1,
            max_retries=config.ollama_max_retries,
            retry_delay=config.ollama_retry_delay,
            tool_timeout=config.tool_timeout,
        )
        agent.set_channel(penny.channel)

        result = await agent.execute()
        assert result is True

        # Should have 2 facts total (1 existing + 1 new), not 3
        facts = penny.db.get_entity_facts(entity.id)
        assert len(facts) == 2
        fact_texts = [f.content for f in facts]
        assert "Costs $1,599 per pair" in fact_texts
        assert "Won What Hi-Fi award in 2023" in fact_texts

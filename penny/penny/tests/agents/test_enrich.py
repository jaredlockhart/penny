"""Integration tests for the EnrichAgent."""

import json
from datetime import UTC, datetime

import pytest

from penny.agents.enrich import EnrichAgent
from penny.constants import PennyConstants
from penny.tests.conftest import TEST_SENDER
from penny.tests.mocks.search_patches import _captured_perplexity_queries


@pytest.mark.asyncio
async def test_learn_enrichment(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Entity with few facts and positive interest -> enrichment search."""
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
        entity = penny.db.entities.get_or_create(TEST_SENDER, "kef ls50 meta")
        assert entity is not None and entity.id is not None
        penny.db.engagements.add(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.USER_SEARCH,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=1.0,
            entity_id=entity.id,
        )

        # Clear previous mock state
        mock_ollama.requests.clear()

        # Create and run learn agent
        agent = EnrichAgent(
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
        facts = penny.db.facts.get_for_entity(entity.id)
        assert len(facts) >= 1
        fact_texts = [f.content for f in facts]
        assert any("1,599" in t for t in fact_texts)
        assert all(f.notified_at is None for f in facts)

        # Verify search was tagged as penny_enrichment
        search_logs = penny.db.searches.get_unprocessed(limit=10)
        assert len(search_logs) >= 1
        assert search_logs[0].trigger == PennyConstants.SearchTrigger.PENNY_ENRICHMENT


@pytest.mark.asyncio
async def test_learn_skips_negative_interest(
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
        entity = penny.db.entities.get_or_create(TEST_SENDER, "sports")
        assert entity is not None and entity.id is not None
        penny.db.engagements.add(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.EMOJI_REACTION,
            valence=PennyConstants.EngagementValence.NEGATIVE,
            strength=0.8,
            entity_id=entity.id,
        )

        agent = EnrichAgent(
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
async def test_learn_no_entities(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """No entities for any user -> returns False."""
    config = make_config()

    async with running_penny(config) as penny:
        agent = EnrichAgent(
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
async def test_learn_no_search_tool(
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
        agent = EnrichAgent(
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
async def test_learn_dedup_facts(
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
        entity = penny.db.entities.get_or_create(TEST_SENDER, "kef ls50 meta")
        assert entity is not None and entity.id is not None
        existing_fact = penny.db.facts.add(
            entity_id=entity.id, content="Costs $1,599 per pair", notified_at=datetime.now(UTC)
        )
        assert existing_fact is not None

        penny.db.engagements.add(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.USER_SEARCH,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=1.0,
            entity_id=entity.id,
        )

        mock_ollama.requests.clear()
        _captured_perplexity_queries.clear()

        agent = EnrichAgent(
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

        # Enrichment query should include existing facts for Perplexity context
        assert len(_captured_perplexity_queries) >= 1
        search_query = _captured_perplexity_queries[0]
        assert "kef ls50 meta" in search_query.lower()
        assert "Costs $1,599 per pair" in search_query

        # Should have 2 facts total (1 existing + 1 new), not 3
        facts = penny.db.facts.get_for_entity(entity.id)
        assert len(facts) == 2
        fact_texts = [f.content for f in facts]
        assert "Costs $1,599 per pair" in fact_texts
        assert "Won What Hi-Fi award in 2023" in fact_texts


@pytest.mark.asyncio
async def test_learn_semantic_interest_priority(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Entity with higher SEARCH_DISCOVERY strength is prioritized.

    Two entities with identical USER_SEARCH engagement — entity A has a
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

        # Create two entities with identical USER_SEARCH engagement
        entity_a = penny.db.entities.get_or_create(TEST_SENDER, "aamas")
        entity_b = penny.db.entities.get_or_create(TEST_SENDER, "coral beach hotel")
        assert entity_a is not None and entity_a.id is not None
        assert entity_b is not None and entity_b.id is not None

        for eid in (entity_a.id, entity_b.id):
            penny.db.engagements.add(
                user=TEST_SENDER,
                engagement_type=PennyConstants.EngagementType.USER_SEARCH,
                valence=PennyConstants.EngagementValence.POSITIVE,
                strength=0.6,
                entity_id=eid,
            )

        # Entity A gets high semantic relevance, entity B gets low
        penny.db.engagements.add(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.SEARCH_DISCOVERY,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.9,
            entity_id=entity_a.id,
        )
        penny.db.engagements.add(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.SEARCH_DISCOVERY,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.6,
            entity_id=entity_b.id,
        )

        # Both get one announced fact so they're eligible for enrichment
        penny.db.facts.add(
            entity_id=entity_a.id,
            content="AAMAS 2026 is held in Cyprus",
            notified_at=datetime.now(UTC),
        )
        penny.db.facts.add(
            entity_id=entity_b.id,
            content="Coral Beach Hotel hosts AAMAS 2026",
            notified_at=datetime.now(UTC),
        )

        mock_ollama.requests.clear()

        agent = EnrichAgent(
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
        facts_a = penny.db.facts.get_for_entity(entity_a.id)
        facts_b = penny.db.facts.get_for_entity(entity_b.id)
        assert len(facts_a) > 1, (
            f"Expected entity A (aamas) to receive new facts from enrichment, "
            f"but it still has {len(facts_a)} fact(s)"
        )
        assert len(facts_b) == 1, (
            f"Expected entity B (coral beach hotel) to remain at 1 fact, "
            f"but it has {len(facts_b)} fact(s)"
        )


@pytest.mark.asyncio
async def test_learn_enrichment_fixed_interval(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Enrichment uses a fixed interval timer between searches.

    First execute() succeeds, second is blocked by the interval timer,
    then resetting the timer allows a third execute() to succeed.
    """
    # Disable per-entity cooldown so only the global interval is tested here
    config = make_config(enrichment_entity_cooldown=0)

    def handler(request: dict, count: int) -> dict:
        return mock_ollama._make_text_response(
            request,
            json.dumps({"facts": ["Some interesting fact"]}),
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Create sender
        await signal_server.push_message(sender=TEST_SENDER, content="hello")
        await signal_server.wait_for_message(timeout=10.0)

        # Create entity with positive interest
        entity = penny.db.entities.get_or_create(TEST_SENDER, "interval test entity")
        assert entity is not None and entity.id is not None
        penny.db.engagements.add(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.USER_SEARCH,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=1.0,
            entity_id=entity.id,
        )

        agent = EnrichAgent(
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

        # First execute: should succeed and record the timer
        result = await agent.execute()
        assert result is True
        assert agent._last_enrich_time is not None

        # Second execute: blocked by fixed interval (not enough time elapsed)
        result = await agent.execute()
        assert result is False

        # Simulate time passing by resetting the timer, and mark facts as
        # announced (as the notification agent would) so the entity stays eligible
        agent._last_enrich_time = None
        facts = penny.db.facts.get_for_entity(entity.id)
        penny.db.facts.mark_notified([f.id for f in facts if f.id is not None])

        # Third execute: timer reset, should succeed
        result = await agent.execute()
        assert result is True


@pytest.mark.asyncio
async def test_enrich_entity_rotation_cooldown(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Recently-enriched entities are skipped; the next eligible entity is picked instead.

    Entity A has higher interest than entity B.  After enriching entity A,
    its last_enriched_at is set and it enters the cooldown window.  On the
    next cycle only entity B is eligible, so entity B gets enriched.
    """
    config = make_config()

    def handler(request: dict, count: int) -> dict:
        return mock_ollama._make_text_response(
            request,
            json.dumps({"facts": ["A new fact discovered"]}),
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Create sender
        await signal_server.push_message(sender=TEST_SENDER, content="hello")
        await signal_server.wait_for_message(timeout=10.0)

        # Entity A: higher interest
        entity_a = penny.db.entities.get_or_create(TEST_SENDER, "entity alpha")
        assert entity_a is not None and entity_a.id is not None
        penny.db.engagements.add(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.USER_SEARCH,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=1.0,
            entity_id=entity_a.id,
        )

        # Entity B: lower interest but still eligible
        entity_b = penny.db.entities.get_or_create(TEST_SENDER, "entity beta")
        assert entity_b is not None and entity_b.id is not None
        penny.db.engagements.add(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.USER_SEARCH,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.5,
            entity_id=entity_b.id,
        )

        agent = EnrichAgent(
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

        # First cycle: entity A wins (higher interest, no cooldown yet)
        result = await agent.execute()
        assert result is True
        facts_a_after_first = penny.db.facts.get_for_entity(entity_a.id)
        assert len(facts_a_after_first) >= 1, "Entity A should have been enriched first"

        # Reset the fixed interval timer to allow immediate re-execution
        agent._last_enrich_time = None

        facts_b_before = penny.db.facts.get_for_entity(entity_b.id)

        # Second cycle: entity A is in cooldown + has unannounced facts,
        # entity B should be picked
        result = await agent.execute()
        assert result is True
        facts_b_after = penny.db.facts.get_for_entity(entity_b.id)
        assert len(facts_b_after) > len(facts_b_before), (
            "Entity B should have been enriched on the second cycle "
            "while entity A is in its cooldown window"
        )


@pytest.mark.asyncio
async def test_enrich_skips_entity_with_unannounced_facts(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Entities with unannounced facts are skipped to avoid piling on more."""
    config = make_config()

    async with running_penny(config) as penny:
        # Create sender
        await signal_server.push_message(sender=TEST_SENDER, content="hello")
        await signal_server.wait_for_message(timeout=10.0)

        # Create entity with positive interest and an unannounced fact
        entity = penny.db.entities.get_or_create(TEST_SENDER, "unannounced test entity")
        assert entity is not None and entity.id is not None
        penny.db.engagements.add(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.USER_SEARCH,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=1.0,
            entity_id=entity.id,
        )
        # Fact with notified_at=None (unannounced)
        penny.db.facts.add(entity_id=entity.id, content="Some unannounced fact")

        agent = EnrichAgent(
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

        # Should return False — entity has unannounced facts
        result = await agent.execute()
        assert result is False


@pytest.mark.asyncio
async def test_learn_enrichment_includes_tagline_in_extraction_prompt(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Tagline is included in the fact extraction prompt to disambiguate entities."""
    config = make_config()
    captured_prompts: list[str] = []

    def handler(request: dict, count: int) -> dict:
        messages = request.get("messages", [])
        last_content = messages[-1].get("content", "") if messages else ""
        captured_prompts.append(last_content)
        return mock_ollama._make_text_response(
            request,
            json.dumps({"facts": ["Some fact about the band"]}),
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Create sender
        await signal_server.push_message(sender=TEST_SENDER, content="hello")
        await signal_server.wait_for_message(timeout=10.0)

        # Create entity with a tagline to disambiguate
        entity = penny.db.entities.get_or_create(TEST_SENDER, "genesis")
        assert entity is not None and entity.id is not None
        penny.db.entities.update_tagline(entity.id, "british progressive rock band")
        penny.db.engagements.add(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.USER_SEARCH,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=1.0,
            entity_id=entity.id,
        )

        mock_ollama.requests.clear()
        captured_prompts.clear()

        agent = EnrichAgent(
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

        # The extraction prompt should include the tagline for disambiguation
        extraction_prompts = [p for p in captured_prompts if "Extract specific" in p]
        assert len(extraction_prompts) >= 1
        assert "genesis (british progressive rock band)" in extraction_prompts[0]

"""Integration tests for /learn command."""

import json

import pytest

from penny.constants import PennyConstants
from penny.tests.conftest import TEST_SENDER, wait_until


@pytest.mark.asyncio
async def test_learn_discovers_entities_from_search(
    signal_server, test_config, mock_ollama, _mock_search, running_penny
):
    """Search-based /learn acknowledges immediately, discovers entities in background."""

    def handler(request: dict, count: int) -> dict:
        messages = request.get("messages", [])
        last_content = messages[-1].get("content", "") if messages else ""

        if "Identify named entities" in last_content:
            return mock_ollama._make_text_response(
                request,
                json.dumps(
                    {
                        "known": [],
                        "new": [
                            {"name": "KEF LS50 Meta"},
                            {"name": "KEF R3 Meta"},
                        ],
                    }
                ),
            )
        return mock_ollama._make_text_response(request, "ok")

    mock_ollama.set_response_handler(handler)

    async with running_penny(test_config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/learn kef speakers")
        response = await signal_server.wait_for_message(timeout=10.0)

        # Immediate response acknowledges the topic (no entity list)
        assert "Okay" in response["message"]
        assert "kef speakers" in response["message"]

        # Background discovery creates entities asynchronously
        await wait_until(lambda: len(penny.db.get_user_entities(TEST_SENDER)) >= 2)

        entities = penny.db.get_user_entities(TEST_SENDER)
        entity_names = [e.name for e in entities]
        assert "kef ls50 meta" in entity_names
        assert "kef r3 meta" in entity_names

        # Verify LEARN_COMMAND engagements for each entity
        engagements = penny.db.get_user_engagements(TEST_SENDER)
        learn_engagements = [
            e
            for e in engagements
            if e.engagement_type == PennyConstants.EngagementType.LEARN_COMMAND
        ]
        assert len(learn_engagements) == 2
        for eng in learn_engagements:
            assert eng.valence == PennyConstants.EngagementValence.POSITIVE
            assert eng.strength == PennyConstants.ENGAGEMENT_STRENGTH_LEARN_COMMAND


@pytest.mark.asyncio
async def test_learn_includes_known_entities(
    signal_server, test_config, mock_ollama, _mock_search, running_penny
):
    """Search that finds existing entities includes them with new ones."""

    def handler(request: dict, count: int) -> dict:
        messages = request.get("messages", [])
        last_content = messages[-1].get("content", "") if messages else ""

        if "Identify named entities" in last_content:
            return mock_ollama._make_text_response(
                request,
                json.dumps(
                    {
                        "known": ["espresso machines"],
                        "new": [{"name": "Breville Barista Express"}],
                    }
                ),
            )
        return mock_ollama._make_text_response(request, "ok")

    mock_ollama.set_response_handler(handler)

    async with running_penny(test_config) as penny:
        # Pre-create an entity
        penny.db.get_or_create_entity(TEST_SENDER, "espresso machines")

        await signal_server.push_message(sender=TEST_SENDER, content="/learn espresso")
        response = await signal_server.wait_for_message(timeout=10.0)

        # Immediate response acknowledges the topic
        assert "espresso" in response["message"].lower()

        # Background discovery finds both known and new entities
        await wait_until(lambda: len(penny.db.get_user_engagements(TEST_SENDER)) >= 2)

        engagements = penny.db.get_user_engagements(TEST_SENDER)
        learn_engagements = [
            e
            for e in engagements
            if e.engagement_type == PennyConstants.EngagementType.LEARN_COMMAND
        ]
        assert len(learn_engagements) == 2


@pytest.mark.asyncio
async def test_learn_no_search_tool_acknowledges(
    signal_server, mock_ollama, _mock_search, make_config, running_penny
):
    """Without Perplexity API key, acknowledges topic but no background discovery."""
    config = make_config(perplexity_api_key=None)

    async with running_penny(config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/learn kef ls50")
        response = await signal_server.wait_for_message(timeout=10.0)

        assert "kef ls50" in response["message"]

        # No entities created (no search tool to discover them)
        entities = penny.db.get_user_entities(TEST_SENDER)
        assert len(entities) == 0


@pytest.mark.asyncio
async def test_learn_no_args_lists_tracked(signal_server, test_config, mock_ollama, running_penny):
    """Test /learn with no args lists entities with positive interest."""
    async with running_penny(test_config) as penny:
        entity = penny.db.get_or_create_entity(TEST_SENDER, "quantum computing")
        assert entity is not None and entity.id is not None
        penny.db.add_engagement(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.LEARN_COMMAND,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=PennyConstants.ENGAGEMENT_STRENGTH_LEARN_COMMAND,
            entity_id=entity.id,
        )
        penny.db.add_fact(entity_id=entity.id, content="Uses qubits instead of classical bits")

        await signal_server.push_message(sender=TEST_SENDER, content="/learn")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "Here's what I'm tracking:" in response["message"]
        assert "quantum computing" in response["message"]
        assert "1 fact" in response["message"]


@pytest.mark.asyncio
async def test_learn_no_args_empty(signal_server, test_config, mock_ollama, running_penny):
    """Test /learn with no args when no entities exist."""
    async with running_penny(test_config) as _penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/learn")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "Nothing being actively researched" in response["message"]

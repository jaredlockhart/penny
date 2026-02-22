"""Integration tests for /memory command."""

import pytest

from penny.constants import PennyConstants
from penny.tests.conftest import TEST_SENDER


@pytest.mark.asyncio
async def test_memory_list_empty(signal_server, test_config, mock_ollama, running_penny):
    """Test /memory with no stored entities."""
    async with running_penny(test_config) as _penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/memory")
        response = await signal_server.wait_for_message(timeout=5.0)
        assert "You don't have any stored memories yet" in response["message"]


@pytest.mark.asyncio
async def test_memory_list_ranked_by_interest(
    signal_server, test_config, mock_ollama, running_penny
):
    """Test /memory lists entities ranked by interest score with scores displayed."""
    async with running_penny(test_config) as penny:
        # Create entities with different engagement levels
        entity1 = penny.db.get_or_create_entity(TEST_SENDER, "nvidia jetson")
        penny.db.add_fact(entity1.id, "Edge AI compute module")

        entity2 = penny.db.get_or_create_entity(TEST_SENDER, "kef ls50 meta")
        penny.db.add_fact(entity2.id, "Costs $1,599 per pair")
        penny.db.add_fact(entity2.id, "Uses MAT driver")

        # Strong engagement for entity2 (learn_command = 1.0)
        penny.db.add_engagement(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.LEARN_COMMAND,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=1.0,
            entity_id=entity2.id,
        )

        # Weaker engagement for entity1 (message_mention = 0.2)
        penny.db.add_engagement(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.MESSAGE_MENTION,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.2,
            entity_id=entity1.id,
        )

        await signal_server.push_message(sender=TEST_SENDER, content="/memory")
        response = await signal_server.wait_for_message(timeout=5.0)

        msg = response["message"]
        assert "Your Memory" in msg
        # Entity2 (score ~1.0) should be ranked above entity1 (score ~0.2)
        assert "1. **kef ls50 meta** (2 facts, interest: +1.00)" in msg
        assert "2. **nvidia jetson** (1 fact, interest: +0.20)" in msg

        # Show entity details — #1 is kef (highest interest)
        await signal_server.push_message(sender=TEST_SENDER, content="/memory 1")
        response2 = await signal_server.wait_for_message(timeout=5.0)
        assert "kef ls50 meta" in response2["message"]
        assert "Costs $1,599 per pair" in response2["message"]
        assert "Uses MAT driver" in response2["message"]


@pytest.mark.asyncio
async def test_memory_shows_negative_scores(signal_server, test_config, mock_ollama, running_penny):
    """Test /memory shows entities with negative interest scores."""
    async with running_penny(test_config) as penny:
        entity = penny.db.get_or_create_entity(TEST_SENDER, "sports")
        penny.db.add_fact(entity.id, "Various athletic activities")

        penny.db.add_engagement(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.EMOJI_REACTION,
            valence=PennyConstants.EngagementValence.NEGATIVE,
            strength=0.8,
            entity_id=entity.id,
        )

        await signal_server.push_message(sender=TEST_SENDER, content="/memory")
        response = await signal_server.wait_for_message(timeout=5.0)

        msg = response["message"]
        assert "sports" in msg
        assert "-0.80" in msg


@pytest.mark.asyncio
async def test_memory_show_not_found(signal_server, test_config, mock_ollama, running_penny):
    """Test /memory <number> with invalid number."""
    async with running_penny(test_config) as _penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/memory 99")
        response = await signal_server.wait_for_message(timeout=5.0)
        assert "doesn't match any memory" in response["message"]


@pytest.mark.asyncio
async def test_memory_delete(signal_server, test_config, mock_ollama, running_penny):
    """Test /memory <number> delete removes entity."""
    async with running_penny(test_config) as penny:
        # Seed entity
        entity = penny.db.get_or_create_entity(TEST_SENDER, "wharfedale linton")
        penny.db.add_fact(entity.id, "Classic heritage speaker")
        penny.db.add_fact(entity.id, "3-way design")

        # Delete it
        await signal_server.push_message(sender=TEST_SENDER, content="/memory 1 delete")
        response = await signal_server.wait_for_message(timeout=5.0)
        assert "Deleted 'wharfedale linton'" in response["message"]
        assert "2 fact(s)" in response["message"]

        # Verify it's gone
        await signal_server.push_message(sender=TEST_SENDER, content="/memory")
        response2 = await signal_server.wait_for_message(timeout=5.0)
        assert "You don't have any stored memories yet" in response2["message"]

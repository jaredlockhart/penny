"""Integration tests for /interests command."""

import pytest

from penny.constants import PennyConstants
from penny.tests.conftest import TEST_SENDER


@pytest.mark.asyncio
async def test_interests_empty(signal_server, test_config, mock_ollama, running_penny):
    """Test /interests with no entities or engagements."""
    async with running_penny(test_config) as _penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/interests")
        response = await signal_server.wait_for_message(timeout=5.0)
        assert "No interest data yet" in response["message"]


@pytest.mark.asyncio
async def test_interests_ranked_by_score(signal_server, test_config, mock_ollama, running_penny):
    """Test /interests shows entities ranked by interest score."""
    async with running_penny(test_config) as penny:
        # Create entities with different engagement levels
        entity1 = penny.db.get_or_create_entity(TEST_SENDER, "kef ls50 meta")
        penny.db.add_fact(entity1.id, "Costs $1,599 per pair")
        penny.db.add_fact(entity1.id, "Uses MAT driver")

        entity2 = penny.db.get_or_create_entity(TEST_SENDER, "espresso machines")
        penny.db.add_fact(entity2.id, "Brews coffee under pressure")

        # Add strong engagement for entity1 (learn_command = 1.0)
        penny.db.add_engagement(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.LEARN_COMMAND,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=PennyConstants.ENGAGEMENT_STRENGTH_LEARN_COMMAND,
            entity_id=entity1.id,
        )

        # Add weaker engagement for entity2 (message_mention = 0.2)
        penny.db.add_engagement(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.MESSAGE_MENTION,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=PennyConstants.ENGAGEMENT_STRENGTH_MESSAGE_MENTION,
            entity_id=entity2.id,
        )

        await signal_server.push_message(sender=TEST_SENDER, content="/interests")
        response = await signal_server.wait_for_message(timeout=5.0)

        msg = response["message"]
        assert "Here's what I think you're interested in:" in msg
        # Entity1 (score ~1.0) should be ranked above entity2 (score ~0.2)
        assert "1. **kef ls50 meta**" in msg
        assert "2 facts" in msg
        assert "2. **espresso machines**" in msg
        assert "1 fact" in msg


@pytest.mark.asyncio
async def test_interests_shows_negative_scores(
    signal_server, test_config, mock_ollama, running_penny
):
    """Test /interests shows entities with negative interest."""
    async with running_penny(test_config) as penny:
        entity = penny.db.get_or_create_entity(TEST_SENDER, "sports")

        penny.db.add_engagement(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.EMOJI_REACTION,
            valence=PennyConstants.EngagementValence.NEGATIVE,
            strength=PennyConstants.ENGAGEMENT_STRENGTH_EMOJI_REACTION_PROACTIVE_NEGATIVE,
            entity_id=entity.id,
        )

        await signal_server.push_message(sender=TEST_SENDER, content="/interests")
        response = await signal_server.wait_for_message(timeout=5.0)

        msg = response["message"]
        assert "**sports**" in msg
        assert "-0.80" in msg


@pytest.mark.asyncio
async def test_interests_skips_entities_without_engagements(
    signal_server, test_config, mock_ollama, running_penny
):
    """Test /interests skips entities that have no engagements."""
    async with running_penny(test_config) as penny:
        # Create entity with engagement
        entity1 = penny.db.get_or_create_entity(TEST_SENDER, "kef ls50 meta")
        penny.db.add_engagement(
            user=TEST_SENDER,
            engagement_type=PennyConstants.EngagementType.LEARN_COMMAND,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=PennyConstants.ENGAGEMENT_STRENGTH_LEARN_COMMAND,
            entity_id=entity1.id,
        )

        # Create entity without engagement
        penny.db.get_or_create_entity(TEST_SENDER, "random entity")

        await signal_server.push_message(sender=TEST_SENDER, content="/interests")
        response = await signal_server.wait_for_message(timeout=5.0)

        msg = response["message"]
        assert "kef ls50 meta" in msg
        assert "random entity" not in msg

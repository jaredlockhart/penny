"""Integration tests for /like, /dislike, /unlike, /undislike commands."""

import pytest

from penny.constants import PennyConstants
from penny.tests.conftest import TEST_SENDER


@pytest.mark.asyncio
async def test_like_list_empty(signal_server, test_config, mock_ollama, running_penny):
    """Test /like with no stored likes."""
    async with running_penny(test_config) as _penny:
        # Send /like
        await signal_server.push_message(sender=TEST_SENDER, content="/like")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should show empty message
        assert "You don't have any likes stored yet" in response["message"]


@pytest.mark.asyncio
async def test_like_add_and_list(signal_server, test_config, mock_ollama, running_penny):
    """Test /like <topic> adds a like, records engagement, and /like lists it."""
    async with running_penny(test_config) as penny:
        # Add a like
        await signal_server.push_message(sender=TEST_SENDER, content="/like cats")
        response1 = await signal_server.wait_for_message(timeout=5.0)
        assert "I added cats to your likes" in response1["message"]

        # Verify engagement was recorded
        engagements = penny.db.get_user_engagements(TEST_SENDER)
        like_engagements = [
            e
            for e in engagements
            if e.engagement_type == PennyConstants.EngagementType.LIKE_COMMAND
        ]
        assert len(like_engagements) == 1
        assert like_engagements[0].valence == PennyConstants.EngagementValence.POSITIVE
        assert like_engagements[0].strength == PennyConstants.ENGAGEMENT_STRENGTH_LIKE_COMMAND
        assert like_engagements[0].preference_id is not None

        # List likes
        await signal_server.push_message(sender=TEST_SENDER, content="/like")
        response2 = await signal_server.wait_for_message(timeout=5.0)
        assert "Here are your stored likes:" in response2["message"]
        assert "1. cats" in response2["message"]


@pytest.mark.asyncio
async def test_like_add_multiple(signal_server, test_config, mock_ollama, running_penny):
    """Test adding multiple likes."""
    async with running_penny(test_config) as _penny:
        # Add first like
        await signal_server.push_message(sender=TEST_SENDER, content="/like space")
        response1 = await signal_server.wait_for_message(timeout=5.0)
        assert "I added space to your likes" in response1["message"]

        # Add second like
        await signal_server.push_message(sender=TEST_SENDER, content="/like video games")
        response2 = await signal_server.wait_for_message(timeout=5.0)
        assert "I added video games to your likes" in response2["message"]

        # List likes
        await signal_server.push_message(sender=TEST_SENDER, content="/like")
        response3 = await signal_server.wait_for_message(timeout=5.0)
        assert "1. space" in response3["message"]
        assert "2. video games" in response3["message"]


@pytest.mark.asyncio
async def test_like_conflict_with_dislike(signal_server, test_config, mock_ollama, running_penny):
    """Test adding a like that conflicts with an existing dislike."""
    async with running_penny(test_config) as penny:
        # Add dislike first
        await signal_server.push_message(sender=TEST_SENDER, content="/dislike bananas")
        response1 = await signal_server.wait_for_message(timeout=5.0)
        assert "I added bananas to your dislikes" in response1["message"]

        # Add same topic as like (should remove from dislikes)
        await signal_server.push_message(sender=TEST_SENDER, content="/like bananas")
        response2 = await signal_server.wait_for_message(timeout=5.0)
        assert (
            "I added bananas to your likes and removed it from your dislikes"
            in response2["message"]
        )

        # Verify both dislike and like engagements exist (historical signals preserved)
        engagements = penny.db.get_user_engagements(TEST_SENDER)
        dislike_engagements = [
            e
            for e in engagements
            if e.engagement_type == PennyConstants.EngagementType.DISLIKE_COMMAND
        ]
        like_engagements = [
            e
            for e in engagements
            if e.engagement_type == PennyConstants.EngagementType.LIKE_COMMAND
        ]
        assert len(dislike_engagements) == 1
        assert len(like_engagements) == 1

        # Verify it's in likes
        await signal_server.push_message(sender=TEST_SENDER, content="/like")
        response3 = await signal_server.wait_for_message(timeout=5.0)
        assert "bananas" in response3["message"]

        # Verify it's not in dislikes
        await signal_server.push_message(sender=TEST_SENDER, content="/dislike")
        response4 = await signal_server.wait_for_message(timeout=5.0)
        assert "You don't have any dislikes stored yet" in response4["message"]


@pytest.mark.asyncio
async def test_dislike_list_empty(signal_server, test_config, mock_ollama, running_penny):
    """Test /dislike with no stored dislikes."""
    async with running_penny(test_config) as _penny:
        # Send /dislike
        await signal_server.push_message(sender=TEST_SENDER, content="/dislike")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should show empty message
        assert "You don't have any dislikes stored yet" in response["message"]


@pytest.mark.asyncio
async def test_dislike_add_and_list(signal_server, test_config, mock_ollama, running_penny):
    """Test /dislike <topic> adds a dislike, records engagement, and /dislike lists it."""
    async with running_penny(test_config) as penny:
        # Add a dislike
        await signal_server.push_message(sender=TEST_SENDER, content="/dislike ai music")
        response1 = await signal_server.wait_for_message(timeout=5.0)
        assert "I added ai music to your dislikes" in response1["message"]

        # Verify engagement was recorded
        engagements = penny.db.get_user_engagements(TEST_SENDER)
        dislike_engagements = [
            e
            for e in engagements
            if e.engagement_type == PennyConstants.EngagementType.DISLIKE_COMMAND
        ]
        assert len(dislike_engagements) == 1
        assert dislike_engagements[0].valence == PennyConstants.EngagementValence.NEGATIVE
        assert dislike_engagements[0].strength == PennyConstants.ENGAGEMENT_STRENGTH_DISLIKE_COMMAND
        assert dislike_engagements[0].preference_id is not None

        # List dislikes
        await signal_server.push_message(sender=TEST_SENDER, content="/dislike")
        response2 = await signal_server.wait_for_message(timeout=5.0)
        assert "Here are your stored dislikes:" in response2["message"]
        assert "1. ai music" in response2["message"]


@pytest.mark.asyncio
async def test_unlike_removes_like(signal_server, test_config, mock_ollama, running_penny):
    """Test /unlike <topic> removes a like."""
    async with running_penny(test_config) as _penny:
        # Add a like
        await signal_server.push_message(sender=TEST_SENDER, content="/like guitars")
        response1 = await signal_server.wait_for_message(timeout=5.0)
        assert "I added guitars to your likes" in response1["message"]

        # Remove it
        await signal_server.push_message(sender=TEST_SENDER, content="/unlike guitars")
        response2 = await signal_server.wait_for_message(timeout=5.0)
        assert "I removed guitars from your likes" in response2["message"]

        # Verify it's gone
        await signal_server.push_message(sender=TEST_SENDER, content="/like")
        response3 = await signal_server.wait_for_message(timeout=5.0)
        assert "You don't have any likes stored yet" in response3["message"]


@pytest.mark.asyncio
async def test_unlike_not_found(signal_server, test_config, mock_ollama, running_penny):
    """Test /unlike with a topic that doesn't exist."""
    async with running_penny(test_config) as _penny:
        # Try to unlike something that wasn't liked
        await signal_server.push_message(sender=TEST_SENDER, content="/unlike space")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should show not found message
        assert "space wasn't in your likes" in response["message"]


@pytest.mark.asyncio
async def test_undislike_removes_dislike(signal_server, test_config, mock_ollama, running_penny):
    """Test /undislike <topic> removes a dislike."""
    async with running_penny(test_config) as _penny:
        # Add a dislike
        await signal_server.push_message(sender=TEST_SENDER, content="/dislike sports")
        response1 = await signal_server.wait_for_message(timeout=5.0)
        assert "I added sports to your dislikes" in response1["message"]

        # Remove it
        await signal_server.push_message(sender=TEST_SENDER, content="/undislike sports")
        response2 = await signal_server.wait_for_message(timeout=5.0)
        assert "I removed sports from your dislikes" in response2["message"]

        # Verify it's gone
        await signal_server.push_message(sender=TEST_SENDER, content="/dislike")
        response3 = await signal_server.wait_for_message(timeout=5.0)
        assert "You don't have any dislikes stored yet" in response3["message"]


@pytest.mark.asyncio
async def test_unlike_no_args(signal_server, test_config, mock_ollama, running_penny):
    """Test /unlike with no arguments shows error."""
    async with running_penny(test_config) as _penny:
        # Send /unlike with no args
        await signal_server.push_message(sender=TEST_SENDER, content="/unlike")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should show usage message
        assert "Please specify what to remove" in response["message"]


@pytest.mark.asyncio
async def test_undislike_no_args(signal_server, test_config, mock_ollama, running_penny):
    """Test /undislike with no arguments shows error."""
    async with running_penny(test_config) as _penny:
        # Send /undislike with no args
        await signal_server.push_message(sender=TEST_SENDER, content="/undislike")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should show usage message
        assert "Please specify what to remove" in response["message"]


@pytest.mark.asyncio
async def test_unlike_by_number(signal_server, test_config, mock_ollama, running_penny):
    """Test /unlike <number> removes a like by list position."""
    async with running_penny(test_config) as _penny:
        # Add multiple likes
        await signal_server.push_message(sender=TEST_SENDER, content="/like space")
        await signal_server.wait_for_message(timeout=5.0)
        await signal_server.push_message(sender=TEST_SENDER, content="/like video games")
        await signal_server.wait_for_message(timeout=5.0)
        await signal_server.push_message(sender=TEST_SENDER, content="/like cats")
        await signal_server.wait_for_message(timeout=5.0)

        # Remove by number (2 = video games)
        await signal_server.push_message(sender=TEST_SENDER, content="/unlike 2")
        response = await signal_server.wait_for_message(timeout=5.0)
        assert "I removed video games from your likes" in response["message"]

        # Verify video games is gone but others remain
        await signal_server.push_message(sender=TEST_SENDER, content="/like")
        response2 = await signal_server.wait_for_message(timeout=5.0)
        assert "1. space" in response2["message"]
        assert "2. cats" in response2["message"]
        assert "video games" not in response2["message"]


@pytest.mark.asyncio
async def test_unlike_by_number_out_of_range(
    signal_server, test_config, mock_ollama, running_penny
):
    """Test /unlike with invalid number shows error."""
    async with running_penny(test_config) as _penny:
        # Add one like
        await signal_server.push_message(sender=TEST_SENDER, content="/like space")
        await signal_server.wait_for_message(timeout=5.0)

        # Try to unlike item 5 (out of range)
        await signal_server.push_message(sender=TEST_SENDER, content="/unlike 5")
        response = await signal_server.wait_for_message(timeout=5.0)
        assert "doesn't match any of your likes" in response["message"]


@pytest.mark.asyncio
async def test_undislike_by_number(signal_server, test_config, mock_ollama, running_penny):
    """Test /undislike <number> removes a dislike by list position."""
    async with running_penny(test_config) as _penny:
        # Add multiple dislikes
        await signal_server.push_message(sender=TEST_SENDER, content="/dislike sports")
        await signal_server.wait_for_message(timeout=5.0)
        await signal_server.push_message(sender=TEST_SENDER, content="/dislike bananas")
        await signal_server.wait_for_message(timeout=5.0)
        await signal_server.push_message(sender=TEST_SENDER, content="/dislike ai music")
        await signal_server.wait_for_message(timeout=5.0)

        # Remove by number (1 = sports)
        await signal_server.push_message(sender=TEST_SENDER, content="/undislike 1")
        response = await signal_server.wait_for_message(timeout=5.0)
        assert "I removed sports from your dislikes" in response["message"]

        # Verify sports is gone but others remain
        await signal_server.push_message(sender=TEST_SENDER, content="/dislike")
        response2 = await signal_server.wait_for_message(timeout=5.0)
        assert "1. bananas" in response2["message"]
        assert "2. ai music" in response2["message"]
        assert "sports" not in response2["message"]


@pytest.mark.asyncio
async def test_like_duplicate_no_extra_engagement(
    signal_server, test_config, mock_ollama, running_penny
):
    """Test /like with duplicate topic does not record an extra engagement."""
    async with running_penny(test_config) as penny:
        # Add a like
        await signal_server.push_message(sender=TEST_SENDER, content="/like cats")
        await signal_server.wait_for_message(timeout=5.0)

        # Try to add the same like again
        await signal_server.push_message(sender=TEST_SENDER, content="/like cats")
        response = await signal_server.wait_for_message(timeout=5.0)
        assert "cats is already in your likes" in response["message"]

        # Should still only have one engagement
        engagements = penny.db.get_user_engagements(TEST_SENDER)
        like_engagements = [
            e
            for e in engagements
            if e.engagement_type == PennyConstants.EngagementType.LIKE_COMMAND
        ]
        assert len(like_engagements) == 1

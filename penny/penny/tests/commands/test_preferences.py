"""Integration tests for /like, /dislike, /unlike, /undislike commands."""

import pytest

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
    """Test /like <topic> adds a like and /like lists it."""
    async with running_penny(test_config) as _penny:
        # Add a like
        await signal_server.push_message(sender=TEST_SENDER, content="/like cats")
        response1 = await signal_server.wait_for_message(timeout=5.0)
        assert "I added cats to your likes" in response1["message"]

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
    async with running_penny(test_config) as _penny:
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
    """Test /dislike <topic> adds a dislike and /dislike lists it."""
    async with running_penny(test_config) as _penny:
        # Add a dislike
        await signal_server.push_message(sender=TEST_SENDER, content="/dislike ai music")
        response1 = await signal_server.wait_for_message(timeout=5.0)
        assert "I added ai music to your dislikes" in response1["message"]

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

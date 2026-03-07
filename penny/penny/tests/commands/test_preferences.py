"""Integration tests for /like, /unlike, /dislike, /undislike commands."""

from datetime import UTC, datetime

import pytest

from penny.constants import PennyConstants
from penny.tests.conftest import TEST_SENDER


def _add_preference(penny, content: str, valence: str) -> None:
    """Helper to seed a preference directly into the DB."""
    now = datetime.now(UTC)
    penny.db.preferences.add(
        user=TEST_SENDER,
        content=content,
        valence=valence,
        source_period_start=now,
        source_period_end=now,
    )


@pytest.mark.asyncio
async def test_like_list_empty(signal_server, test_config, mock_ollama, running_penny):
    """Test /like with no likes shows empty message."""
    async with running_penny(test_config) as _penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/like")
        response = await signal_server.wait_for_message(timeout=5.0)
        assert "You don't have any likes yet" in response["message"]


@pytest.mark.asyncio
async def test_like_list_shows_positives(signal_server, test_config, mock_ollama, running_penny):
    """Test /like lists positive preferences with numbers."""
    async with running_penny(test_config) as penny:
        _add_preference(penny, "dark roast coffee", PennyConstants.PreferenceValence.POSITIVE)
        _add_preference(penny, "hiking", PennyConstants.PreferenceValence.POSITIVE)
        # Negative pref should not appear
        _add_preference(penny, "cold weather", PennyConstants.PreferenceValence.NEGATIVE)

        await signal_server.push_message(sender=TEST_SENDER, content="/like")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "Your Likes" in response["message"]
        assert "dark roast coffee" in response["message"]
        assert "hiking" in response["message"]
        assert "cold weather" not in response["message"]


@pytest.mark.asyncio
async def test_unlike_deletes_positive(signal_server, test_config, mock_ollama, running_penny):
    """Test /unlike <N> removes a positive preference."""
    async with running_penny(test_config) as penny:
        _add_preference(penny, "dark roast coffee", PennyConstants.PreferenceValence.POSITIVE)
        _add_preference(penny, "hiking", PennyConstants.PreferenceValence.POSITIVE)

        await signal_server.push_message(sender=TEST_SENDER, content="/unlike 1")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "Removed" in response["message"]
        assert "Remaining" in response["message"]

        # Verify DB state
        remaining = penny.db.preferences.get_for_user_by_valence(
            TEST_SENDER, PennyConstants.PreferenceValence.POSITIVE
        )
        assert len(remaining) == 1


@pytest.mark.asyncio
async def test_dislike_list_shows_negatives(signal_server, test_config, mock_ollama, running_penny):
    """Test /dislike lists negative preferences."""
    async with running_penny(test_config) as penny:
        _add_preference(penny, "cold weather", PennyConstants.PreferenceValence.NEGATIVE)
        _add_preference(penny, "dark roast coffee", PennyConstants.PreferenceValence.POSITIVE)

        await signal_server.push_message(sender=TEST_SENDER, content="/dislike")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "Your Dislikes" in response["message"]
        assert "cold weather" in response["message"]
        assert "dark roast coffee" not in response["message"]


@pytest.mark.asyncio
async def test_undislike_deletes_last_shows_no_remaining(
    signal_server, test_config, mock_ollama, running_penny
):
    """Test /undislike <N> on the last dislike shows 'no more' message."""
    async with running_penny(test_config) as penny:
        _add_preference(penny, "cold weather", PennyConstants.PreferenceValence.NEGATIVE)

        await signal_server.push_message(sender=TEST_SENDER, content="/undislike 1")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "Removed 'cold weather' from your dislikes" in response["message"]
        assert "No more dislikes" in response["message"]


@pytest.mark.asyncio
async def test_preference_invalid_number(signal_server, test_config, mock_ollama, running_penny):
    """Test preference command with invalid number shows error."""
    async with running_penny(test_config) as penny:
        _add_preference(penny, "hiking", PennyConstants.PreferenceValence.POSITIVE)

        await signal_server.push_message(sender=TEST_SENDER, content="/like 99")
        response = await signal_server.wait_for_message(timeout=5.0)
        assert "No preference with number 99" in response["message"]

        await signal_server.push_message(sender=TEST_SENDER, content="/like abc")
        response = await signal_server.wait_for_message(timeout=5.0)
        assert "Invalid preference number: abc" in response["message"]

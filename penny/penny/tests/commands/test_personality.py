"""Integration tests for /personality command."""

import pytest

from penny.tests.conftest import TEST_SENDER


@pytest.mark.asyncio
async def test_personality_no_args_shows_default(
    signal_server, test_config, mock_ollama, running_penny
):
    """Test /personality with no args shows default when no custom personality set."""
    async with running_penny(test_config) as _penny:
        # Send /personality
        await signal_server.push_message(sender=TEST_SENDER, content="/personality")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should show default message
        assert "No custom personality set" in response["message"]
        assert "using the default" in response["message"]


@pytest.mark.asyncio
async def test_personality_set_and_view(signal_server, test_config, mock_ollama, running_penny):
    """Test setting and viewing custom personality."""

    # Mock the personality transform to return transformed versions of the responses
    # Request 1: transform confirmation message
    # Request 2: transform view response
    def response_handler(request: dict, count: int) -> dict:
        if count == 1:
            return mock_ollama._make_text_response(request, "Ok, personality updated ✅")
        else:
            return mock_ollama._make_text_response(
                request, "Current personality: you are a pirate who speaks in nautical metaphors"
            )

    mock_ollama.set_response_handler(response_handler)

    async with running_penny(test_config) as _penny:
        # Set personality
        await signal_server.push_message(
            sender=TEST_SENDER,
            content="/personality you are a pirate who speaks in nautical metaphors",
        )

        # Wait for confirmation
        response = await signal_server.wait_for_message(timeout=5.0)
        assert "personality updated" in response["message"]

        # View personality
        await signal_server.push_message(sender=TEST_SENDER, content="/personality")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)
        assert "Current personality:" in response["message"]
        assert "pirate who speaks in nautical metaphors" in response["message"]


@pytest.mark.asyncio
async def test_personality_reset(signal_server, test_config, mock_ollama, running_penny):
    """Test resetting personality to default."""

    # Mock the personality transform responses
    # Request 1: set confirmation
    # Request 2: reset confirmation
    def response_handler(request: dict, count: int) -> dict:
        if count == 1:
            return mock_ollama._make_text_response(request, "Ok, personality updated ✅")
        else:
            return mock_ollama._make_text_response(request, "Ok, personality reset to default ✅")

    mock_ollama.set_response_handler(response_handler)

    async with running_penny(test_config) as _penny:
        # Set personality
        await signal_server.push_message(
            sender=TEST_SENDER, content="/personality you are extremely concise"
        )
        await signal_server.wait_for_message(timeout=5.0)

        # Reset personality
        await signal_server.push_message(sender=TEST_SENDER, content="/personality reset")

        # Wait for confirmation
        response = await signal_server.wait_for_message(timeout=5.0)
        assert "personality reset to default" in response["message"]

        # Verify it's reset (no personality, so no transform happens)
        await signal_server.push_message(sender=TEST_SENDER, content="/personality")
        response = await signal_server.wait_for_message(timeout=5.0)
        assert "No custom personality set" in response["message"]


@pytest.mark.asyncio
async def test_personality_reset_when_not_set(
    signal_server, test_config, mock_ollama, running_penny
):
    """Test resetting personality when none was set."""
    async with running_penny(test_config) as _penny:
        # Try to reset without setting first
        await signal_server.push_message(sender=TEST_SENDER, content="/personality reset")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)
        assert "No custom personality to reset" in response["message"]

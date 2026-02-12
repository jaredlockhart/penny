"""Integration tests for /style command and AdaptiveStyleAgent."""

import pytest

from penny.tests.conftest import TEST_SENDER, wait_until


@pytest.mark.asyncio
async def test_style_no_profile_yet(signal_server, test_config, mock_ollama, running_penny):
    """Test /style when user has no profile yet."""
    async with running_penny(test_config) as _penny:
        # Send /style with no messages
        await signal_server.push_message(sender=TEST_SENDER, content="/style")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should say not enough messages
        assert "No style profile yet" in response["message"]
        assert "at least 20 messages" in response["message"]


@pytest.mark.asyncio
async def test_style_view_existing_profile(signal_server, test_config, mock_ollama, running_penny):
    """Test /style displays existing style profile."""
    async with running_penny(test_config) as penny:
        # Create a style profile directly in database
        penny.db.upsert_user_style_profile(
            user_id=TEST_SENDER,
            style_prompt="User writes in short lowercase sentences with minimal punctuation.",
            message_count=50,
        )

        # Send /style
        await signal_server.push_message(sender=TEST_SENDER, content="/style")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should display the profile
        assert "**Your Speaking Style**" in response["message"]
        assert "User writes in short lowercase sentences" in response["message"]
        assert "Status: enabled" in response["message"]
        assert "Based on 50 messages" in response["message"]


@pytest.mark.asyncio
async def test_style_reset(signal_server, test_config, mock_ollama, running_penny):
    """Test /style reset regenerates profile from current messages."""

    # Mock the style analysis response
    def handler(request: dict, count: int) -> dict:
        return mock_ollama._make_text_response(
            request,
            '{"description": "User writes in proper sentences with good grammar and punctuation."}',
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(test_config) as penny:
        # Add enough messages to database
        for i in range(25):
            penny.db.log_message(
                direction="incoming",
                sender=TEST_SENDER,
                content=f"This is test message number {i}. It has proper grammar and punctuation.",
            )

        # Send /style reset
        await signal_server.push_message(sender=TEST_SENDER, content="/style reset")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should confirm regeneration
        assert "Style profile reset and regenerated" in response["message"]

        # Verify profile was created in database
        profile = penny.db.get_user_style_profile(TEST_SENDER)
        assert profile is not None
        assert "proper sentences" in profile.style_prompt
        assert profile.enabled is True


@pytest.mark.asyncio
async def test_style_reset_not_enough_messages(
    signal_server, test_config, mock_ollama, running_penny
):
    """Test /style reset with insufficient messages."""
    async with running_penny(test_config) as penny:
        # Add only a few messages
        for i in range(5):
            penny.db.log_message(
                direction="incoming",
                sender=TEST_SENDER,
                content=f"Message {i}",
            )

        # Send /style reset
        await signal_server.push_message(sender=TEST_SENDER, content="/style reset")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should say not enough messages
        assert "at least 20 messages" in response["message"]


@pytest.mark.asyncio
async def test_style_off(signal_server, test_config, mock_ollama, running_penny):
    """Test /style off disables adaptive style."""
    async with running_penny(test_config) as penny:
        # Create a style profile
        penny.db.upsert_user_style_profile(
            user_id=TEST_SENDER,
            style_prompt="User writes casually.",
            message_count=30,
        )

        # Send /style off
        await signal_server.push_message(sender=TEST_SENDER, content="/style off")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should confirm disabled
        assert "Adaptive style disabled" in response["message"]

        # Verify profile was disabled
        profile = penny.db.get_user_style_profile(TEST_SENDER)
        assert profile is not None
        assert profile.enabled is False


@pytest.mark.asyncio
async def test_style_on(signal_server, test_config, mock_ollama, running_penny):
    """Test /style on re-enables adaptive style."""
    async with running_penny(test_config) as penny:
        # Create a disabled style profile
        penny.db.upsert_user_style_profile(
            user_id=TEST_SENDER,
            style_prompt="User writes casually.",
            message_count=30,
        )
        penny.db.update_style_profile_enabled(TEST_SENDER, False)

        # Send /style on
        await signal_server.push_message(sender=TEST_SENDER, content="/style on")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should confirm enabled
        assert "Adaptive style enabled" in response["message"]

        # Verify profile was enabled
        profile = penny.db.get_user_style_profile(TEST_SENDER)
        assert profile is not None
        assert profile.enabled is True


@pytest.mark.asyncio
async def test_style_on_no_profile(signal_server, test_config, mock_ollama, running_penny):
    """Test /style on with no existing profile."""
    async with running_penny(test_config) as _penny:
        # Send /style on with no profile
        await signal_server.push_message(sender=TEST_SENDER, content="/style on")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should say no profile yet
        assert "No style profile yet" in response["message"]


@pytest.mark.asyncio
async def test_style_unknown_subcommand(signal_server, test_config, mock_ollama, running_penny):
    """Test /style with unknown subcommand."""
    async with running_penny(test_config) as _penny:
        # Send /style foo
        await signal_server.push_message(sender=TEST_SENDER, content="/style foo")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should show error
        assert "Unknown subcommand: foo" in response["message"]


@pytest.mark.asyncio
async def test_style_injection_in_message_agent(
    signal_server, test_config, mock_ollama, running_penny
):
    """Test that style profile is injected into MessageAgent system prompt."""
    # Mock Ollama to capture the messages it receives
    captured_messages = []

    def capture_messages(request: dict, count: int) -> dict:
        captured_messages.append(request.get("messages", []))
        return mock_ollama._make_text_response(request, "ok cool")

    mock_ollama.set_response_handler(capture_messages)

    async with running_penny(test_config) as penny:
        # Create a style profile
        penny.db.upsert_user_style_profile(
            user_id=TEST_SENDER,
            style_prompt="User writes in ALL CAPS WITH LOTS OF EXCLAMATION MARKS!!!",
            message_count=50,
        )

        # Send a message
        await signal_server.push_message(sender=TEST_SENDER, content="hey whats up")

        # Wait for response
        await signal_server.wait_for_message(timeout=5.0)

        # Check that style profile was injected into the messages
        assert len(captured_messages) > 0
        messages = captured_messages[0]

        # Find the system message with style profile
        style_found = False
        for msg in messages:
            if msg["role"] == "system" and "User Communication Style" in msg["content"]:
                assert "ALL CAPS" in msg["content"]
                style_found = True
                break

        assert style_found, "Style profile was not injected into system messages"


@pytest.mark.asyncio
async def test_style_agent_background_analysis(
    signal_server, test_config, mock_ollama, running_penny
):
    """Test that AdaptiveStyleAgent runs in background and creates profiles."""

    # Mock the style analysis response
    def handler(request: dict, count: int) -> dict:
        return mock_ollama._make_text_response(
            request,
            '{"description": "User writes in short lowercase messages with minimal punctuation."}',
        )

    mock_ollama.set_response_handler(handler)

    # Reduce the style agent interval to 1 second for testing
    async with running_penny(test_config) as penny:
        # Override the style agent's schedule interval
        for schedule in penny.scheduler._schedules:
            if schedule.agent.name == "style":
                schedule._interval = 1.0
                schedule._last_run = None

        # Add enough messages for the user
        for i in range(25):
            penny.db.log_message(
                direction="incoming",
                sender=TEST_SENDER,
                content=f"test message {i}",
            )

        # Wait for the style agent to run
        await wait_until(
            lambda: penny.db.get_user_style_profile(TEST_SENDER) is not None,
            timeout=10.0,
        )

        # Verify profile was created
        profile = penny.db.get_user_style_profile(TEST_SENDER)
        assert profile is not None
        assert "short lowercase messages" in profile.style_prompt
        assert profile.enabled is True
        assert profile.message_count > 0

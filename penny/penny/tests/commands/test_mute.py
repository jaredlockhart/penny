"""Integration tests for /mute and /unmute commands."""

import pytest

from penny.agents.notification import NotificationAgent
from penny.tests.conftest import TEST_SENDER


def _create_notification_agent(penny, config):
    """Create a NotificationAgent wired to penny's DB and channel."""
    agent = NotificationAgent(
        system_prompt="",
        background_model_client=penny.background_model_client,
        foreground_model_client=penny.foreground_model_client,
        tools=[],
        db=penny.db,
        max_steps=1,
        tool_timeout=config.tool_timeout,
        config=config,
    )
    agent.set_channel(penny.channel)
    return agent


@pytest.mark.asyncio
async def test_mute_command(signal_server, test_config, mock_ollama, running_penny):
    """Test /mute sets mute state and returns acknowledgment."""
    async with running_penny(test_config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/mute")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "Notifications muted" in response["message"]
        assert "Use /unmute" in response["message"]
        assert penny.db.users.is_muted(TEST_SENDER) is True


@pytest.mark.asyncio
async def test_unmute_command(signal_server, test_config, mock_ollama, running_penny):
    """Test /unmute clears mute state and returns acknowledgment."""
    async with running_penny(test_config) as penny:
        # Mute first
        penny.db.users.set_muted(TEST_SENDER)

        await signal_server.push_message(sender=TEST_SENDER, content="/unmute")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "Notifications unmuted" in response["message"]
        assert penny.db.users.is_muted(TEST_SENDER) is False


@pytest.mark.asyncio
async def test_mute_already_muted(signal_server, test_config, mock_ollama, running_penny):
    """Test /mute when already muted returns 'already muted' message."""
    async with running_penny(test_config) as penny:
        penny.db.users.set_muted(TEST_SENDER)

        await signal_server.push_message(sender=TEST_SENDER, content="/mute")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "already muted" in response["message"]


@pytest.mark.asyncio
async def test_unmute_already_unmuted(signal_server, test_config, mock_ollama, running_penny):
    """Test /unmute when not muted returns 'aren't muted' message."""
    async with running_penny(test_config) as _penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/unmute")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "aren't muted" in response["message"]


@pytest.mark.asyncio
async def test_notification_skipped_when_muted(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Muted users receive no proactive notifications."""
    config = make_config()
    mock_ollama.set_default_flow(search_query="test", final_response="ok")

    async with running_penny(config) as penny:
        msg_id = penny.db.messages.log_message(
            direction="incoming", sender=TEST_SENDER, content="hello"
        )
        penny.db.messages.mark_processed([msg_id])

        # Create entity with un-notified facts
        entity = penny.db.entities.get_or_create(TEST_SENDER, "test entity")
        assert entity is not None and entity.id is not None
        penny.db.facts.add(entity.id, "Interesting fact")

        # Mute the user
        penny.db.users.set_muted(TEST_SENDER)

        agent = _create_notification_agent(penny, config)
        signal_server.outgoing_messages.clear()
        result = await agent.execute()

        assert result is False
        assert len(signal_server.outgoing_messages) == 0

        # Unmute and verify notification can now be sent
        penny.db.users.set_unmuted(TEST_SENDER)

        def handler(request: dict, count: int) -> dict:
            return mock_ollama._make_text_response(
                request,
                "Here's an interesting discovery — some really great new findings about this!",
            )

        mock_ollama.set_response_handler(handler)

        signal_server.outgoing_messages.clear()
        result = await agent.execute()

        assert result is True
        assert len(signal_server.outgoing_messages) == 1

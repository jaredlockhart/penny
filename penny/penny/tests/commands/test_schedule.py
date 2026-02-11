"""Integration tests for /schedule command."""

import pytest

from penny.database.models import UserInfo
from penny.tests.conftest import TEST_SENDER


@pytest.mark.asyncio
async def test_schedule_list_empty(signal_server, test_config, mock_ollama, running_penny):
    """Test /schedule with no schedules shows empty message."""
    async with running_penny(test_config) as penny:
        # Create user profile so we have timezone
        with penny.db.get_session() as session:
            user_info = UserInfo(
                sender=TEST_SENDER,
                name="Test User",
                location="Seattle",
                timezone="America/Los_Angeles",
                date_of_birth="1990-01-01",
            )
            session.add(user_info)
            session.commit()

        # Send /schedule
        await signal_server.push_message(sender=TEST_SENDER, content="/schedule")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should show empty message
        assert "You don't have any scheduled tasks yet" in response["message"]


@pytest.mark.asyncio
async def test_schedule_create_requires_timezone(
    signal_server, test_config, mock_ollama, running_penny
):
    """Test /schedule creation requires user timezone to be set."""
    async with running_penny(test_config) as _penny:
        # Try to create schedule without user profile
        await signal_server.push_message(
            sender=TEST_SENDER, content="/schedule daily 9am what's the news?"
        )

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should prompt for timezone
        assert "I need to know your timezone first" in response["message"]
        assert "Send me your location" in response["message"]


@pytest.mark.asyncio
async def test_schedule_create_and_list(signal_server, test_config, mock_ollama, running_penny):
    """Test creating a schedule and listing it."""
    # Configure mock Ollama to return schedule parse result
    mock_ollama.set_response_handler(
        lambda messages, tools: {
            "message": {
                "role": "assistant",
                "content": (
                    '{"timing_description": "daily 9am", '
                    '"prompt_text": "what\'s the news?", '
                    '"cron_expression": "0 9 * * *"}'
                ),
            }
        }
    )

    async with running_penny(test_config) as penny:
        # Create user profile with timezone
        with penny.db.get_session() as session:
            user_info = UserInfo(
                sender=TEST_SENDER,
                name="Test User",
                location="Seattle",
                timezone="America/Los_Angeles",
                date_of_birth="1990-01-01",
            )
            session.add(user_info)
            session.commit()

        # Create schedule
        await signal_server.push_message(
            sender=TEST_SENDER, content="/schedule daily 9am what's the news?"
        )

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should confirm creation
        assert "Added daily 9am: what's the news?" in response["message"]

        # List schedules
        await signal_server.push_message(sender=TEST_SENDER, content="/schedule")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should list the schedule
        assert "1. daily 9am 'what's the news?'" in response["message"]


@pytest.mark.asyncio
async def test_schedule_delete(signal_server, test_config, mock_ollama, running_penny):
    """Test deleting a schedule."""
    # Configure mock Ollama to return schedule parse result
    mock_ollama.set_response_handler(
        lambda messages, tools: {
            "message": {
                "role": "assistant",
                "content": (
                    '{"timing_description": "hourly", '
                    '"prompt_text": "sports scores", '
                    '"cron_expression": "0 * * * *"}'
                ),
            }
        }
    )

    async with running_penny(test_config) as penny:
        # Create user profile with timezone
        with penny.db.get_session() as session:
            user_info = UserInfo(
                sender=TEST_SENDER,
                name="Test User",
                location="Seattle",
                timezone="America/Los_Angeles",
                date_of_birth="1990-01-01",
            )
            session.add(user_info)
            session.commit()

        # Create schedule
        await signal_server.push_message(
            sender=TEST_SENDER, content="/schedule hourly sports scores"
        )

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)
        assert "Added hourly: sports scores" in response["message"]

        # Delete schedule
        await signal_server.push_message(sender=TEST_SENDER, content="/schedule delete 1")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should confirm deletion
        assert "Deleted 'hourly sports scores'" in response["message"]
        assert "No more scheduled tasks" in response["message"]


@pytest.mark.asyncio
async def test_schedule_delete_invalid_index(
    signal_server, test_config, mock_ollama, running_penny
):
    """Test deleting with invalid index shows error."""
    async with running_penny(test_config) as penny:
        # Create user profile
        with penny.db.get_session() as session:
            user_info = UserInfo(
                sender=TEST_SENDER,
                name="Test User",
                location="Seattle",
                timezone="America/Los_Angeles",
                date_of_birth="1990-01-01",
            )
            session.add(user_info)
            session.commit()

        # Try to delete non-existent schedule
        await signal_server.push_message(sender=TEST_SENDER, content="/schedule delete 99")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should show error
        assert "No schedule with number 99" in response["message"]

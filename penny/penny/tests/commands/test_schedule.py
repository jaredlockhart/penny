"""Integration tests for /schedule command."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from penny.database.models import UserInfo
from penny.tests.conftest import TEST_SENDER


def _is_schedule_due(cron_expression: str, now: datetime) -> bool:
    """Helper that mirrors the fixed ScheduleExecutor firing logic."""
    from croniter import croniter

    cron = croniter(cron_expression, now - timedelta(seconds=60))
    next_occurrence = cron.get_next(datetime)
    return next_occurrence <= now


def test_schedule_fires_at_exact_cron_time():
    """Schedule must fire when checked at the exact scheduled second.

    Regression test for the bug where croniter.get_prev(now) returned
    yesterday's occurrence when 'now' exactly equalled the cron time,
    causing the schedule to silently miss its tick.
    """
    tz = ZoneInfo("America/Los_Angeles")
    # Exactly at 9:30:00 — this is the problematic case
    now = datetime(2026, 2, 24, 9, 30, 0, tzinfo=tz)
    assert _is_schedule_due("30 9 * * *", now), "Schedule should fire at the exact cron second"


def test_schedule_fires_within_60_second_window():
    """Schedule should fire for any check within the 60-second window."""
    tz = ZoneInfo("America/Los_Angeles")
    for offset_seconds in [0, 1, 30, 59]:
        now = datetime(2026, 2, 24, 9, 30, offset_seconds, tzinfo=tz)
        assert _is_schedule_due("30 9 * * *", now), (
            f"Schedule should fire at +{offset_seconds}s past cron time"
        )


def test_schedule_does_not_fire_before_cron_time():
    """Schedule must not fire before the cron time."""
    tz = ZoneInfo("America/Los_Angeles")
    for offset_seconds in [1, 30, 59]:
        now = datetime(2026, 2, 24, 9, 29, 60 - offset_seconds, tzinfo=tz)
        assert not _is_schedule_due("30 9 * * *", now), (
            f"Schedule should NOT fire {offset_seconds}s before cron time"
        )


def test_schedule_does_not_fire_after_window():
    """Schedule must not fire more than 60 seconds after the cron time."""
    tz = ZoneInfo("America/Los_Angeles")
    now = datetime(2026, 2, 24, 9, 31, 0, tzinfo=tz)  # 60 seconds after 9:30
    assert not _is_schedule_due("30 9 * * *", now), (
        "Schedule should NOT fire 60 seconds after cron time"
    )


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
        assert "1. **daily 9am**: what's the news?" in response["message"]


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

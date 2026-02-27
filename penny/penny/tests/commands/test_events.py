"""Integration tests for /events command."""

from datetime import UTC, datetime

import pytest

from penny.constants import PennyConstants
from penny.tests.conftest import TEST_SENDER


@pytest.mark.asyncio
async def test_events_lists_recent(signal_server, mock_ollama, make_config, running_penny):
    """/events shows recent events as a numbered list."""
    config = make_config(news_api_key="test-key")

    async with running_penny(config) as penny:
        penny.db.events.add(
            user=TEST_SENDER,
            headline="GPT-5 Released",
            summary="OpenAI releases GPT-5 with improved reasoning.",
            occurred_at=datetime(2026, 2, 20, tzinfo=UTC),
            source_type=PennyConstants.EventSourceType.NEWS_API,
            source_url="https://example.com/gpt5",
        )

        await signal_server.push_message(sender=TEST_SENDER, content="/events")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "Recent Events" in response["message"]
        assert "GPT-5 Released" in response["message"]


@pytest.mark.asyncio
async def test_events_detail(signal_server, mock_ollama, make_config, running_penny):
    """/events <N> shows full event detail including linked entities."""
    config = make_config(news_api_key="test-key")

    async with running_penny(config) as penny:
        event = penny.db.events.add(
            user=TEST_SENDER,
            headline="SpaceX Starship Launch",
            summary="SpaceX successfully launches Starship to orbit.",
            occurred_at=datetime(2026, 2, 25, 14, 30, tzinfo=UTC),
            source_type=PennyConstants.EventSourceType.NEWS_API,
            source_url="https://example.com/starship",
        )
        assert event is not None and event.id is not None

        entity = penny.db.entities.get_or_create(TEST_SENDER, "SpaceX")
        assert entity is not None and entity.id is not None
        penny.db.events.link_entity(event.id, entity.id)

        await signal_server.push_message(sender=TEST_SENDER, content="/events 1")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "SpaceX Starship Launch" in response["message"]
        assert "successfully launches" in response["message"]
        assert "https://example.com/starship" in response["message"]
        assert "SpaceX" in response["message"]


@pytest.mark.asyncio
async def test_events_empty(signal_server, mock_ollama, make_config, running_penny):
    """/events with no events shows empty message."""
    config = make_config(news_api_key="test-key")

    async with running_penny(config) as _penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/events")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "No recent events" in response["message"]


@pytest.mark.asyncio
async def test_events_invalid_number(signal_server, mock_ollama, make_config, running_penny):
    """/events with out-of-range number shows error."""
    config = make_config(news_api_key="test-key")

    async with running_penny(config) as penny:
        penny.db.events.add(
            user=TEST_SENDER,
            headline="Some Event",
            summary="Something happened.",
            occurred_at=datetime(2026, 2, 20, tzinfo=UTC),
            source_type=PennyConstants.EventSourceType.NEWS_API,
        )

        await signal_server.push_message(sender=TEST_SENDER, content="/events 99")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "doesn't match" in response["message"]


@pytest.mark.asyncio
async def test_events_not_registered_without_api_key(
    signal_server, mock_ollama, test_config, running_penny
):
    """/events is not available when NEWS_API_KEY is not configured."""
    async with running_penny(test_config) as _penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/events")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "Unknown command" in response["message"]

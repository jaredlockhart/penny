"""Integration tests for /follow and /unfollow commands."""

import json
from datetime import UTC, datetime

import pytest

from penny.constants import PennyConstants
from penny.tests.conftest import TEST_SENDER


def _get_prompt_text(request: dict) -> str:
    """Extract the user message content from a mock Ollama request."""
    messages = request.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _make_follow_handler(mock_ollama):
    """Create a handler that returns parse result then query terms on successive calls."""

    def handler(request: dict, count: int) -> dict:
        prompt = _get_prompt_text(request)
        if "Parse this follow command" in prompt:
            return mock_ollama._make_text_response(
                request,
                json.dumps(
                    {
                        "timing_description": "daily",
                        "topic_text": "AI safety",
                        "cron_expression": "0 9 * * *",
                    }
                ),
            )
        # Query terms generation
        return mock_ollama._make_text_response(
            request,
            json.dumps({"query_terms": ["AI safety", "AI alignment"]}),
        )

    return handler


@pytest.mark.asyncio
async def test_follow_creates_prompt(
    signal_server, mock_ollama, make_config, test_user_info, running_penny
):
    """/follow <topic> parses timing via LLM and creates a FollowPrompt."""
    mock_ollama.set_response_handler(_make_follow_handler(mock_ollama))
    config = make_config(news_api_key="test-key")

    async with running_penny(config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/follow AI safety")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "keep track of" in response["message"]
        assert "AI safety" in response["message"]
        assert "daily" in response["message"]

        follows = penny.db.follow_prompts.get_active(TEST_SENDER)
        assert len(follows) == 1
        assert follows[0].prompt_text == "AI safety"
        assert follows[0].cron_expression == "0 9 * * *"
        assert follows[0].timing_description == "daily"
        assert follows[0].user_timezone == "America/Los_Angeles"
        assert json.loads(follows[0].query_terms) == ["AI safety", "AI alignment"]


@pytest.mark.asyncio
async def test_follow_with_timing(
    signal_server, mock_ollama, make_config, test_user_info, running_penny
):
    """/follow daily 9:30am <topic> creates a prompt with cron-based timing."""

    def handler(request: dict, count: int) -> dict:
        prompt = _get_prompt_text(request)
        if "Parse this follow command" in prompt:
            return mock_ollama._make_text_response(
                request,
                json.dumps(
                    {
                        "timing_description": "daily 9:30am",
                        "topic_text": "AI safety",
                        "cron_expression": "30 9 * * *",
                    }
                ),
            )
        return mock_ollama._make_text_response(
            request,
            json.dumps({"query_terms": ["AI safety", "AI alignment"]}),
        )

    mock_ollama.set_response_handler(handler)
    config = make_config(news_api_key="test-key")

    async with running_penny(config) as penny:
        await signal_server.push_message(
            sender=TEST_SENDER, content="/follow daily 9:30am AI safety"
        )
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "keep track of" in response["message"]
        assert "daily 9:30am" in response["message"]

        follows = penny.db.follow_prompts.get_active(TEST_SENDER)
        assert len(follows) == 1
        assert follows[0].prompt_text == "AI safety"
        assert follows[0].cron_expression == "30 9 * * *"
        assert follows[0].timing_description == "daily 9:30am"


@pytest.mark.asyncio
async def test_follow_requires_timezone(signal_server, mock_ollama, make_config, running_penny):
    """/follow without a user timezone prompts to set one."""
    config = make_config(news_api_key="test-key")

    async with running_penny(config) as _penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/follow AI safety")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "timezone" in response["message"].lower()


@pytest.mark.asyncio
async def test_follow_no_args_lists_active(signal_server, mock_ollama, make_config, running_penny):
    """/follow with no args shows active subscriptions with timing."""
    config = make_config(news_api_key="test-key")

    async with running_penny(config) as penny:
        penny.db.follow_prompts.create(
            user=TEST_SENDER,
            prompt_text="spacex launches",
            query_terms='["spacex", "rocket launch"]',
            cron_expression="0 9 * * *",
            timing_description="daily",
        )

        await signal_server.push_message(sender=TEST_SENDER, content="/follow")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "Following" in response["message"]
        assert "spacex launches" in response["message"]
        assert "daily" in response["message"]


@pytest.mark.asyncio
async def test_follow_no_args_empty(signal_server, mock_ollama, make_config, running_penny):
    """/follow with no args and no follows shows empty message."""
    config = make_config(news_api_key="test-key")

    async with running_penny(config) as _penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/follow")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "not following anything" in response["message"]


@pytest.mark.asyncio
async def test_unfollow_cancels(signal_server, mock_ollama, make_config, running_penny):
    """/unfollow <N> cancels the Nth follow."""
    config = make_config(news_api_key="test-key")

    async with running_penny(config) as penny:
        fp = penny.db.follow_prompts.create(
            user=TEST_SENDER,
            prompt_text="quantum computing",
            query_terms='["quantum computing"]',
            cron_expression="0 9 * * *",
            timing_description="daily",
        )
        assert fp is not None and fp.id is not None

        # Create events linked to this follow prompt
        for i in range(3):
            event = penny.db.events.add(
                user=TEST_SENDER,
                headline=f"Quantum news {i}",
                summary=f"Article {i}",
                occurred_at=datetime.now(UTC),
                source_type=PennyConstants.EventSourceType.NEWS_API,
                source_url=f"https://example.com/q{i}",
                external_id=f"https://example.com/q{i}",
                follow_prompt_id=fp.id,
            )
            assert event is not None and event.id is not None

        await signal_server.push_message(sender=TEST_SENDER, content="/unfollow 1")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "Stopped following" in response["message"]
        assert "quantum computing" in response["message"]

        # Verify cancelled in DB
        active = penny.db.follow_prompts.get_active(TEST_SENDER)
        assert len(active) == 0

        # Verify related events deleted
        events = penny.db.events.get_for_user(TEST_SENDER)
        assert len(events) == 0


@pytest.mark.asyncio
async def test_unfollow_invalid_number(signal_server, mock_ollama, make_config, running_penny):
    """/unfollow with out-of-range number shows error."""
    config = make_config(news_api_key="test-key")

    async with running_penny(config) as penny:
        penny.db.follow_prompts.create(
            user=TEST_SENDER,
            prompt_text="quantum computing",
            query_terms='["quantum computing"]',
            cron_expression="0 9 * * *",
            timing_description="daily",
        )

        await signal_server.push_message(sender=TEST_SENDER, content="/unfollow 5")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "doesn't match" in response["message"]


@pytest.mark.asyncio
async def test_unfollow_no_args_lists(signal_server, mock_ollama, make_config, running_penny):
    """/unfollow with no args shows numbered list."""
    config = make_config(news_api_key="test-key")

    async with running_penny(config) as penny:
        penny.db.follow_prompts.create(
            user=TEST_SENDER,
            prompt_text="quantum computing",
            query_terms='["quantum computing"]',
            cron_expression="0 9 * * *",
            timing_description="daily",
        )

        await signal_server.push_message(sender=TEST_SENDER, content="/unfollow")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "Following" in response["message"]
        assert "quantum computing" in response["message"]


@pytest.mark.asyncio
async def test_follow_without_api_key_shows_config_error(
    signal_server, mock_ollama, test_config, running_penny
):
    """/follow without NEWS_API_KEY tells user to configure it."""
    async with running_penny(test_config) as _penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/follow AI")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "NEWS_API_KEY" in response["message"]

"""Integration tests for /follow and /unfollow commands."""

import json

import pytest

from penny.tests.conftest import TEST_SENDER


@pytest.mark.asyncio
async def test_follow_creates_prompt(signal_server, mock_ollama, make_config, running_penny):
    """/follow <topic> generates query terms and creates a FollowPrompt."""

    def handler(request: dict, count: int) -> dict:
        return mock_ollama._make_text_response(
            request,
            json.dumps({"query_terms": ["AI safety", "AI alignment"]}),
        )

    mock_ollama.set_response_handler(handler)
    config = make_config(news_api_key="test-key")

    async with running_penny(config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/follow AI safety")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "keep track of" in response["message"]
        assert "AI safety" in response["message"]

        follows = penny.db.follow_prompts.get_active(TEST_SENDER)
        assert len(follows) == 1
        assert follows[0].prompt_text == "AI safety"
        assert json.loads(follows[0].query_terms) == ["AI safety", "AI alignment"]


@pytest.mark.asyncio
async def test_follow_no_args_lists_active(signal_server, mock_ollama, make_config, running_penny):
    """/follow with no args shows active subscriptions."""
    config = make_config(news_api_key="test-key")

    async with running_penny(config) as penny:
        penny.db.follow_prompts.create(
            user=TEST_SENDER,
            prompt_text="spacex launches",
            query_terms='["spacex", "rocket launch"]',
        )

        await signal_server.push_message(sender=TEST_SENDER, content="/follow")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "Following" in response["message"]
        assert "spacex launches" in response["message"]


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
        penny.db.follow_prompts.create(
            user=TEST_SENDER,
            prompt_text="quantum computing",
            query_terms='["quantum computing"]',
        )

        await signal_server.push_message(sender=TEST_SENDER, content="/unfollow 1")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "Stopped following" in response["message"]
        assert "quantum computing" in response["message"]

        # Verify cancelled in DB
        active = penny.db.follow_prompts.get_active(TEST_SENDER)
        assert len(active) == 0


@pytest.mark.asyncio
async def test_unfollow_invalid_number(signal_server, mock_ollama, make_config, running_penny):
    """/unfollow with out-of-range number shows error."""
    config = make_config(news_api_key="test-key")

    async with running_penny(config) as penny:
        penny.db.follow_prompts.create(
            user=TEST_SENDER,
            prompt_text="quantum computing",
            query_terms='["quantum computing"]',
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
        )

        await signal_server.push_message(sender=TEST_SENDER, content="/unfollow")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "Following" in response["message"]
        assert "quantum computing" in response["message"]


@pytest.mark.asyncio
async def test_follow_not_registered_without_api_key(
    signal_server, mock_ollama, test_config, running_penny
):
    """/follow is not available when NEWS_API_KEY is not configured."""
    async with running_penny(test_config) as _penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/follow AI")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "Unknown command" in response["message"]

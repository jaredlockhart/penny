"""Integration tests for EventAgent."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from penny.agents.event import EventAgent, _normalize_headline
from penny.constants import PennyConstants
from penny.tests.conftest import TEST_SENDER
from penny.tools.news import NewsArticle


def _make_article(
    title: str = "SpaceX launches Starship",
    description: str = "SpaceX successfully launched its Starship rocket today.",
    url: str = "https://example.com/spacex-starship",
) -> NewsArticle:
    """Create a test NewsArticle."""
    return NewsArticle(
        title=title,
        description=description,
        url=url,
        published_at=datetime.now(UTC),
        source_name="Test News",
    )


def _create_event_agent(penny, config, news_tool=None, embedding_model_client=None):
    """Create an EventAgent wired to penny's DB with a mock news tool."""
    return EventAgent(
        news_tool=news_tool,
        system_prompt="",
        background_model_client=penny.background_model_client,
        foreground_model_client=penny.foreground_model_client,
        embedding_model_client=embedding_model_client,
        tools=[],
        db=penny.db,
        max_steps=1,
        tool_timeout=config.tool_timeout,
        config=config,
    )


def _make_mock_news_tool(articles: list[NewsArticle] | None = None):
    """Create a mock NewsTool that returns the given articles."""
    tool = AsyncMock()
    tool.search = AsyncMock(return_value=articles or [])
    return tool


@pytest.mark.asyncio
async def test_event_agent_skips_without_news_tool(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """EventAgent returns False when no news tool is configured."""
    config = make_config()

    async with running_penny(config) as penny:
        penny.db.follow_prompts.create(
            user=TEST_SENDER,
            prompt_text="AI safety",
            query_terms='["AI safety"]',
        )

        agent = _create_event_agent(penny, config, news_tool=None)
        result = await agent.execute()
        assert result is False


@pytest.mark.asyncio
async def test_event_agent_skips_without_follow_prompts(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """EventAgent returns False when no follow prompts exist."""
    config = make_config()
    news_tool = _make_mock_news_tool()

    async with running_penny(config) as penny:
        agent = _create_event_agent(penny, config, news_tool=news_tool)
        result = await agent.execute()
        assert result is False


@pytest.mark.asyncio
async def test_event_agent_creates_events_and_links_entities(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Full cycle: poll → create events → extract + link entities."""
    config = make_config()
    articles = [
        _make_article(
            title="SpaceX launches Starship",
            description="SpaceX successfully launched its Starship rocket today.",
            url="https://example.com/spacex-1",
        ),
        _make_article(
            title="NASA announces Artemis update",
            description="NASA provided an update on the Artemis program.",
            url="https://example.com/nasa-1",
        ),
    ]
    news_tool = _make_mock_news_tool(articles)

    def handler(request: dict, count: int) -> dict:
        return mock_ollama._make_text_response(
            request,
            json.dumps({"entities": ["SpaceX", "NASA"]}),
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        penny.db.follow_prompts.create(
            user=TEST_SENDER,
            prompt_text="space launches",
            query_terms='["spacex", "rocket launch"]',
        )

        agent = _create_event_agent(penny, config, news_tool=news_tool)
        result = await agent.execute()

        assert result is True

        # Two events created
        events = penny.db.events.get_recent(TEST_SENDER, days=7)
        assert len(events) == 2

        # Entities created (names are lowercased by entity store)
        all_entities = penny.db.entities.get_for_user(TEST_SENDER)
        entity_names = {e.name for e in all_entities}
        assert "spacex" in entity_names
        assert "nasa" in entity_names

        # Each event linked to both entities
        for event in events:
            assert event.id is not None
            linked = penny.db.events.get_entities_for_event(event.id)
            assert len(linked) == 2

        # last_polled_at updated
        follows = penny.db.follow_prompts.get_active(TEST_SENDER)
        assert follows[0].last_polled_at is not None


@pytest.mark.asyncio
async def test_event_agent_dedup_by_url(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Articles with duplicate URLs are filtered out."""
    config = make_config()
    article = _make_article(url="https://example.com/existing")
    news_tool = _make_mock_news_tool([article])

    mock_ollama.set_response_handler(
        lambda req, count: mock_ollama._make_text_response(req, json.dumps({"entities": []}))
    )

    async with running_penny(config) as penny:
        penny.db.follow_prompts.create(
            user=TEST_SENDER,
            prompt_text="tech news",
            query_terms='["tech"]',
        )

        # Pre-create an event with the same URL
        penny.db.events.add(
            user=TEST_SENDER,
            headline="Existing article",
            summary="Already in DB",
            occurred_at=datetime.now(UTC),
            source_type=PennyConstants.EventSourceType.NEWS_API,
            source_url="https://example.com/existing",
            external_id="https://example.com/existing",
        )

        agent = _create_event_agent(penny, config, news_tool=news_tool)
        result = await agent.execute()

        # No new events created (duplicate filtered)
        assert result is False
        events = penny.db.events.get_recent(TEST_SENDER, days=7)
        assert len(events) == 1  # only the pre-existing one


@pytest.mark.asyncio
async def test_event_agent_dedup_by_headline(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Articles with matching normalized headlines are filtered out."""
    config = make_config()
    # Same headline, different URL — should still dedup
    article = _make_article(
        title="SpaceX Launches Starship!",
        url="https://example.com/new-url",
    )
    news_tool = _make_mock_news_tool([article])

    mock_ollama.set_response_handler(
        lambda req, count: mock_ollama._make_text_response(req, json.dumps({"entities": []}))
    )

    async with running_penny(config) as penny:
        penny.db.follow_prompts.create(
            user=TEST_SENDER,
            prompt_text="space news",
            query_terms='["spacex"]',
        )

        # Pre-create event with same headline (different punctuation/case)
        penny.db.events.add(
            user=TEST_SENDER,
            headline="spacex launches starship",
            summary="Existing",
            occurred_at=datetime.now(UTC),
            source_type=PennyConstants.EventSourceType.NEWS_API,
            source_url="https://example.com/old-url",
            external_id="https://example.com/old-url",
        )

        agent = _create_event_agent(penny, config, news_tool=news_tool)
        result = await agent.execute()

        assert result is False
        events = penny.db.events.get_recent(TEST_SENDER, days=7)
        assert len(events) == 1


@pytest.mark.asyncio
async def test_event_agent_filters_irrelevant_articles(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Articles irrelevant to the followed topic are filtered by embedding similarity."""
    config = make_config()
    articles = [
        _make_article(
            title="SpaceX launches Starship",
            url="https://example.com/spacex-relevant",
        ),
        _make_article(
            title="Netflix buys Warner Brothers",
            url="https://example.com/netflix-irrelevant",
        ),
    ]
    news_tool = _make_mock_news_tool(articles)

    # Mock embedding client: returns vectors where the topic and relevant article
    # are similar (same direction) but the irrelevant article is orthogonal.
    topic_vec = [1.0, 0.0, 0.0]
    relevant_vec = [0.9, 0.1, 0.0]  # cosine ~0.99 with topic
    irrelevant_vec = [0.0, 0.0, 1.0]  # cosine 0.0 with topic

    embed_responses = {
        "space launches": [topic_vec],
        "SpaceX launches Starship": [relevant_vec],
        "Netflix buys Warner Brothers": [irrelevant_vec],
    }
    embedding_client = AsyncMock()
    embedding_client.embed = AsyncMock(side_effect=lambda text: embed_responses[text])

    mock_ollama.set_response_handler(
        lambda req, count: mock_ollama._make_text_response(req, json.dumps({"entities": []}))
    )

    async with running_penny(config) as penny:
        penny.db.follow_prompts.create(
            user=TEST_SENDER,
            prompt_text="space launches",
            query_terms='["spacex", "rocket launch"]',
        )

        agent = _create_event_agent(
            penny, config, news_tool=news_tool, embedding_model_client=embedding_client
        )
        result = await agent.execute()

        assert result is True
        events = penny.db.events.get_recent(TEST_SENDER, days=7)
        assert len(events) == 1
        assert events[0].headline == "SpaceX launches Starship"


@pytest.mark.asyncio
async def test_event_agent_dedup_by_tcr(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Articles with high token containment ratio to existing events are filtered out."""
    config = make_config()
    # Same story, reworded — high TCR but different normalized headline
    article = _make_article(
        title="SpaceX Successfully Launches Starship Rocket",
        url="https://example.com/different-url",
    )
    news_tool = _make_mock_news_tool([article])

    mock_ollama.set_response_handler(
        lambda req, count: mock_ollama._make_text_response(req, json.dumps({"entities": []}))
    )

    async with running_penny(config) as penny:
        penny.db.follow_prompts.create(
            user=TEST_SENDER,
            prompt_text="space news",
            query_terms='["spacex"]',
        )

        # Pre-create event — shares most tokens with the new article
        penny.db.events.add(
            user=TEST_SENDER,
            headline="SpaceX Launches Starship",
            summary="Existing",
            occurred_at=datetime.now(UTC),
            source_type=PennyConstants.EventSourceType.NEWS_API,
            source_url="https://example.com/old-url",
            external_id="https://example.com/old-url",
        )

        agent = _create_event_agent(penny, config, news_tool=news_tool)
        result = await agent.execute()

        assert result is False
        events = penny.db.events.get_recent(TEST_SENDER, days=7)
        assert len(events) == 1


def test_normalize_headline():
    """Headline normalization strips punctuation and normalizes case."""
    assert _normalize_headline("SpaceX Launches Starship!") == "spacex launches starship"
    assert _normalize_headline("  Hello, World!  ") == "hello world"
    assert _normalize_headline("AI: The Future?") == "ai the future"
    assert _normalize_headline("café résumé") == "cafe resume"

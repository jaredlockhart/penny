"""Integration tests for EventAgent."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from penny.agents.event import EventAgent, _normalize_headline
from penny.constants import PennyConstants
from penny.database.models import FollowPrompt
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
async def test_event_agent_creates_events(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Full cycle: poll → create events (no entity extraction)."""
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

    async with running_penny(config) as penny:
        penny.db.follow_prompts.create(
            user=TEST_SENDER,
            prompt_text="space launches",
            query_terms='["spacex", "rocket launch"]',
        )

        agent = _create_event_agent(penny, config, news_tool=news_tool)
        result = await agent.execute()

        assert result is True

        # Two events created, each linked to the follow prompt
        events = penny.db.events.get_recent(TEST_SENDER, days=7)
        assert len(events) == 2
        follows = penny.db.follow_prompts.get_active(TEST_SENDER)
        assert follows[0].id is not None
        for event in events:
            assert event.follow_prompt_id == follows[0].id

        # No entities created from news articles
        all_entities = penny.db.entities.get_for_user(TEST_SENDER)
        assert len(all_entities) == 0

        # last_polled_at updated
        follows = penny.db.follow_prompts.get_active(TEST_SENDER)
        assert follows[0].last_polled_at is not None

        # Regression: second execute reads last_polled_at back from SQLite as
        # a naive datetime — must not raise TypeError on aware/naive subtraction
        result2 = await agent.execute()
        assert result2 is False  # poll interval not elapsed


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
async def test_event_agent_tag_fallback_rescues_broad_topic(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Tag extraction rescues articles whose title doesn't embed close to a broad topic.

    When the title embedding scores below the threshold, the agent extracts
    topic tags via the LLM and checks those against the topic instead.
    """
    config = make_config()
    articles = [
        _make_article(
            title="Regeneron Renews Sponsorship of Science Talent Search",
            url="https://example.com/regeneron",
        ),
    ]
    news_tool = _make_mock_news_tool(articles)

    # Title embedding is orthogonal to topic (fails 0.40 threshold),
    # but the extracted tags embedding is similar (passes).
    topic_vec = [1.0, 0.0, 0.0]
    title_vec = [0.1, 0.0, 0.9]  # cosine ~0.11 with topic — fails
    tags_vec = [0.9, 0.1, 0.0]  # cosine ~0.99 with topic — passes

    embed_responses = {
        "science": [topic_vec],
        "Regeneron Renews Sponsorship of Science Talent Search": [title_vec],
        "science, education, regeneron": [tags_vec],
    }
    embedding_client = AsyncMock()
    embedding_client.embed = AsyncMock(side_effect=lambda text: embed_responses[text])

    def handler(request: dict, count: int) -> dict:
        content = request.get("messages", [{}])[-1].get("content", "")
        if "Extract 2-4 one-word topic tags" in content:
            return mock_ollama._make_text_response(request, '["science", "education", "regeneron"]')
        return mock_ollama._make_text_response(request, json.dumps({"entities": []}))

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        penny.db.follow_prompts.create(
            user=TEST_SENDER,
            prompt_text="science",
            query_terms='["science news", "scientific research"]',
        )

        agent = _create_event_agent(
            penny, config, news_tool=news_tool, embedding_model_client=embedding_client
        )
        result = await agent.execute()

        assert result is True
        events = penny.db.events.get_recent(TEST_SENDER, days=7)
        assert len(events) == 1
        assert "Regeneron" in events[0].headline


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


@pytest.mark.asyncio
async def test_event_agent_polls_hourly_skips_daily(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Cron-aware polling: hourly follow is due, daily follow is skipped."""
    config = make_config()
    articles = [_make_article(url="https://example.com/hourly-article")]
    news_tool = _make_mock_news_tool(articles)

    mock_ollama.set_response_handler(
        lambda req, count: mock_ollama._make_text_response(req, json.dumps({"entities": []}))
    )

    async with running_penny(config) as penny:
        # Daily follow polled 2 hours ago — not due (cron period ~24h)
        daily = penny.db.follow_prompts.create(
            user=TEST_SENDER,
            prompt_text="daily topic",
            query_terms='["daily"]',
            cron_expression="0 9 * * *",
            timing_description="daily",
        )
        assert daily is not None and daily.id is not None
        penny.db.follow_prompts.update_last_polled(daily.id)
        # Backdate last_polled_at to 2 hours ago
        with penny.db.get_session() as session:
            row = session.get(FollowPrompt, daily.id)
            assert row is not None
            row.last_polled_at = datetime.now(UTC) - timedelta(hours=2)
            session.add(row)
            session.commit()

        # Hourly follow polled 2 hours ago — due (cron period ~1h)
        hourly = penny.db.follow_prompts.create(
            user=TEST_SENDER,
            prompt_text="hourly topic",
            query_terms='["hourly"]',
            cron_expression="0 * * * *",
            timing_description="hourly",
        )
        assert hourly is not None and hourly.id is not None
        penny.db.follow_prompts.update_last_polled(hourly.id)
        with penny.db.get_session() as session:
            row = session.get(FollowPrompt, hourly.id)
            assert row is not None
            row.last_polled_at = datetime.now(UTC) - timedelta(hours=2)
            session.add(row)
            session.commit()

        agent = _create_event_agent(penny, config, news_tool=news_tool)
        result = await agent.execute()

        assert result is True

        # Only the hourly follow was polled — check its last_polled_at updated
        updated_hourly = penny.db.follow_prompts.get(hourly.id)
        updated_daily = penny.db.follow_prompts.get(daily.id)
        assert updated_hourly is not None and updated_daily is not None
        assert updated_hourly.last_polled_at is not None
        assert updated_daily.last_polled_at is not None
        # Hourly was re-polled (recent), daily was not (still 2h ago)
        # SQLite returns naive datetimes — add UTC for comparison
        hourly_polled = updated_hourly.last_polled_at.replace(tzinfo=UTC)
        daily_polled = updated_daily.last_polled_at.replace(tzinfo=UTC)
        hourly_elapsed = (datetime.now(UTC) - hourly_polled).total_seconds()
        daily_elapsed = (datetime.now(UTC) - daily_polled).total_seconds()
        assert hourly_elapsed < 10  # just polled
        assert daily_elapsed > 3600  # still ~2h ago (more than 1h hourly period)


@pytest.mark.asyncio
async def test_event_agent_caps_by_relevance(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Articles are ranked by relevance score and capped at EVENT_MAX_PER_POLL."""
    config = make_config()
    # Create 4 articles — cap at 2 so only top 2 by relevance survive
    articles = [
        _make_article(title="Low relevance article", url="https://example.com/low"),
        _make_article(title="High relevance article", url="https://example.com/high"),
        _make_article(title="Medium relevance article", url="https://example.com/med"),
        _make_article(title="Top relevance article", url="https://example.com/top"),
    ]
    news_tool = _make_mock_news_tool(articles)

    # Topic and article embeddings — all above relevance threshold (0.40) but
    # with different scores so ranking produces a clear top 2
    topic_vec = [1.0, 0.0, 0.0]
    embed_map = {
        "space launches": [topic_vec],
        "Low relevance article": [[0.5, 0.0, 0.866]],  # cosine ~0.50
        "High relevance article": [[0.95, 0.05, 0.0]],  # cosine ~0.998
        "Medium relevance article": [[0.7, 0.0, 0.714]],  # cosine ~0.70
        "Top relevance article": [[1.0, 0.0, 0.0]],  # cosine 1.0
    }
    embedding_client = AsyncMock()
    embedding_client.embed = AsyncMock(side_effect=lambda text: embed_map[text])

    async with running_penny(config) as penny:
        # Override EVENT_MAX_PER_POLL to 2 for this test
        penny.config.runtime.EVENT_MAX_PER_POLL = 2

        penny.db.follow_prompts.create(
            user=TEST_SENDER,
            prompt_text="space launches",
            query_terms='["spacex"]',
        )

        agent = _create_event_agent(
            penny, config, news_tool=news_tool, embedding_model_client=embedding_client
        )
        agent.config = penny.config  # Use the overridden config
        result = await agent.execute()

        assert result is True
        events = penny.db.events.get_recent(TEST_SENDER, days=7)
        assert len(events) == 2

        headlines = {e.headline for e in events}
        assert "Top relevance article" in headlines
        assert "High relevance article" in headlines
        assert "Low relevance article" not in headlines
        assert "Medium relevance article" not in headlines


def test_normalize_headline():
    """Headline normalization strips punctuation and normalizes case."""
    assert _normalize_headline("SpaceX Launches Starship!") == "spacex launches starship"
    assert _normalize_headline("  Hello, World!  ") == "hello world"
    assert _normalize_headline("AI: The Future?") == "ai the future"
    assert _normalize_headline("café résumé") == "cafe resume"

"""Integration tests for EventAgent."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from penny.agents.event import _normalize_headline
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


def _make_mock_news_tool(
    articles: list[NewsArticle] | None = None,
    rate_limited: bool = False,
):
    """Create a mock NewsTool that returns the given articles."""
    from unittest.mock import MagicMock

    tool = AsyncMock()
    tool.search = AsyncMock(return_value=articles or [])
    tool.consume_rate_limit_notification = MagicMock(return_value=rate_limited)
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

        result = await penny.event_agent.execute()
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
        penny.event_agent._news_tool = news_tool
        result = await penny.event_agent.execute()
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

        penny.event_agent._news_tool = news_tool
        result = await penny.event_agent.execute()

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
        result2 = await penny.event_agent.execute()
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

        # Pre-create an event with the same URL (marked notified so polling proceeds)
        existing = penny.db.events.add(
            user=TEST_SENDER,
            headline="Existing article",
            summary="Already in DB",
            occurred_at=datetime.now(UTC),
            source_type=PennyConstants.EventSourceType.NEWS_API,
            source_url="https://example.com/existing",
            external_id="https://example.com/existing",
        )
        assert existing is not None and existing.id is not None
        penny.db.events.mark_notified([existing.id])

        penny.event_agent._news_tool = news_tool
        result = await penny.event_agent.execute()

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

        # Pre-create event with same headline (marked notified so polling proceeds)
        existing = penny.db.events.add(
            user=TEST_SENDER,
            headline="spacex launches starship",
            summary="Existing",
            occurred_at=datetime.now(UTC),
            source_type=PennyConstants.EventSourceType.NEWS_API,
            source_url="https://example.com/old-url",
            external_id="https://example.com/old-url",
        )
        assert existing is not None and existing.id is not None
        penny.db.events.mark_notified([existing.id])

        penny.event_agent._news_tool = news_tool
        result = await penny.event_agent.execute()

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

        penny.event_agent._news_tool = news_tool
        penny.event_agent._embedding_model_client = embedding_client
        result = await penny.event_agent.execute()

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

        penny.event_agent._news_tool = news_tool
        penny.event_agent._embedding_model_client = embedding_client
        result = await penny.event_agent.execute()

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

        # Pre-create event — shares most tokens (marked notified so polling proceeds)
        existing = penny.db.events.add(
            user=TEST_SENDER,
            headline="SpaceX Launches Starship",
            summary="Existing",
            occurred_at=datetime.now(UTC),
            source_type=PennyConstants.EventSourceType.NEWS_API,
            source_url="https://example.com/old-url",
            external_id="https://example.com/old-url",
        )
        assert existing is not None and existing.id is not None
        penny.db.events.mark_notified([existing.id])

        penny.event_agent._news_tool = news_tool
        result = await penny.event_agent.execute()

        assert result is False
        events = penny.db.events.get_recent(TEST_SENDER, days=7)
        assert len(events) == 1


@pytest.mark.asyncio
async def test_event_agent_skips_prompt_with_unannounced_events(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Follow prompt with unannounced events is skipped; announced prompt is polled."""
    config = make_config()
    articles = [_make_article(url="https://example.com/new-article")]
    news_tool = _make_mock_news_tool(articles)

    async with running_penny(config) as penny:
        # First prompt: has an unannounced event — should be skipped
        blocked = penny.db.follow_prompts.create(
            user=TEST_SENDER,
            prompt_text="blocked topic",
            query_terms='["blocked"]',
        )
        assert blocked is not None and blocked.id is not None
        penny.db.events.add(
            user=TEST_SENDER,
            headline="Unannounced article",
            summary="Waiting for notification",
            occurred_at=datetime.now(UTC),
            source_type=PennyConstants.EventSourceType.NEWS_API,
            source_url="https://example.com/unannounced",
            external_id="https://example.com/unannounced",
            follow_prompt_id=blocked.id,
        )

        # Second prompt: no unannounced events — should be polled
        ready = penny.db.follow_prompts.create(
            user=TEST_SENDER,
            prompt_text="ready topic",
            query_terms='["ready"]',
        )
        assert ready is not None and ready.id is not None

        penny.event_agent._news_tool = news_tool
        result = await penny.event_agent.execute()

        assert result is True

        # Only the ready prompt was polled
        updated_ready = penny.db.follow_prompts.get(ready.id)
        updated_blocked = penny.db.follow_prompts.get(blocked.id)
        assert updated_ready is not None and updated_blocked is not None
        assert updated_ready.last_polled_at is not None
        assert updated_blocked.last_polled_at is None  # never polled


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

        penny.event_agent._news_tool = news_tool
        penny.event_agent._embedding_model_client = embedding_client
        result = await penny.event_agent.execute()

        assert result is True
        events = penny.db.events.get_recent(TEST_SENDER, days=7)
        assert len(events) == 2

        headlines = {e.headline for e in events}
        assert "Top relevance article" in headlines
        assert "High relevance article" in headlines
        assert "Low relevance article" not in headlines
        assert "Medium relevance article" not in headlines


@pytest.mark.asyncio
async def test_event_agent_notifies_user_on_rate_limit(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """EventAgent sends a message to the user when the NewsAPI rate limit is first hit."""
    config = make_config()
    # Mock news tool that signals a new rate limit event
    news_tool = _make_mock_news_tool(articles=[], rate_limited=True)

    async with running_penny(config) as penny:
        penny.db.follow_prompts.create(
            user=TEST_SENDER,
            prompt_text="AI safety",
            query_terms='["AI safety"]',
        )

        penny.event_agent._news_tool = news_tool
        await penny.event_agent.execute()

        # User should receive a notification about the rate limit
        from penny.tests.conftest import wait_until

        await wait_until(lambda: len(signal_server.outgoing_messages) > 0)
        assert any(
            "rate limit" in msg["message"].lower() for msg in signal_server.outgoing_messages
        )


def test_normalize_headline():
    """Headline normalization strips punctuation and normalizes case."""
    assert _normalize_headline("SpaceX Launches Starship!") == "spacex launches starship"
    assert _normalize_headline("  Hello, World!  ") == "hello world"
    assert _normalize_headline("AI: The Future?") == "ai the future"
    assert _normalize_headline("café résumé") == "cafe resume"

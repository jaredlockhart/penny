"""Unit tests for NewsTool caching and rate limit handling."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from newsapi.newsapi_exception import NewsAPIException

from penny.tools.news import CACHE_TTL_SECONDS, NEWS_API_RATE_LIMITED_CODE, NewsTool

_ARTICLE_TITLE = "SpaceX Launches Starship"
_ARTICLE_URL = "https://example.com/spacex"
_PUBLISHED_AT = "2026-03-01T10:00:00Z"

_FAKE_RESPONSE = {
    "status": "ok",
    "totalResults": 1,
    "articles": [
        {
            "title": _ARTICLE_TITLE,
            "description": "A test article.",
            "url": _ARTICLE_URL,
            "publishedAt": _PUBLISHED_AT,
            "source": {"name": "Test News"},
            "urlToImage": None,
        }
    ],
}


def _make_news_tool(api_response: dict | None = None) -> tuple[NewsTool, MagicMock]:
    """Create a NewsTool wired to a mock NewsApiClient.

    Returns (tool, mock_client) so callers can inspect call counts.
    """
    mock_client = MagicMock()
    mock_client.get_everything.return_value = api_response or _FAKE_RESPONSE

    tool = NewsTool.__new__(NewsTool)
    tool._client = mock_client
    tool._cache = {}
    return tool, mock_client


class TestNewsToolCaching:
    """NewsTool caches successful API responses."""

    @pytest.mark.asyncio
    async def test_cache_hit_skips_api_call(self):
        """Second search with same query returns cached result without an API call."""
        tool, mock_client = _make_news_tool()

        results1 = await tool.search(["spacex", "rocket"])
        results2 = await tool.search(["spacex", "rocket"])

        assert len(results1) == 1
        assert results1 == results2
        # API should only have been called once
        assert mock_client.get_everything.call_count == 1

    @pytest.mark.asyncio
    async def test_different_queries_each_hit_api(self):
        """Different query terms produce different cache keys — both call the API."""
        tool, mock_client = _make_news_tool()

        await tool.search(["spacex"])
        await tool.search(["nasa"])

        assert mock_client.get_everything.call_count == 2

    @pytest.mark.asyncio
    async def test_expired_cache_calls_api_again(self):
        """Cache entries older than CACHE_TTL_SECONDS are considered stale."""
        tool, mock_client = _make_news_tool()

        # Pre-populate an expired cache entry
        query = "spacex OR rocket"
        from_date = datetime(2026, 3, 1, tzinfo=UTC)
        cache_key = tool._make_cache_key(query, from_date)
        stale_time = datetime.now(UTC) - timedelta(seconds=CACHE_TTL_SECONDS + 1)
        tool._cache[cache_key] = (stale_time, [])

        await tool.search(["spacex", "rocket"], from_date=from_date)

        assert mock_client.get_everything.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_miss_after_from_date_changes(self):
        """Different from_date (different day) produces a different cache key."""
        tool, mock_client = _make_news_tool()

        day1 = datetime(2026, 3, 1, tzinfo=UTC)
        day2 = datetime(2026, 3, 2, tzinfo=UTC)

        await tool.search(["spacex"], from_date=day1)
        await tool.search(["spacex"], from_date=day2)

        assert mock_client.get_everything.call_count == 2

    @pytest.mark.asyncio
    async def test_successful_results_are_cached(self):
        """Successful API response is stored so subsequent calls return it."""
        tool, mock_client = _make_news_tool()

        await tool.search(["spacex"])

        # Verify cache is populated
        assert len(tool._cache) == 1
        _, cached_articles = next(iter(tool._cache.values()))
        assert len(cached_articles) == 1
        assert cached_articles[0].title == _ARTICLE_TITLE


class TestNewsToolMakeCacheKey:
    """Cache key construction normalizes from_date to day granularity."""

    def test_same_day_same_key(self):
        tool, _ = _make_news_tool()
        key1 = tool._make_cache_key("spacex", datetime(2026, 3, 1, 8, 0, tzinfo=UTC))
        key2 = tool._make_cache_key("spacex", datetime(2026, 3, 1, 22, 30, tzinfo=UTC))
        assert key1 == key2

    def test_different_day_different_key(self):
        tool, _ = _make_news_tool()
        key1 = tool._make_cache_key("spacex", datetime(2026, 3, 1, tzinfo=UTC))
        key2 = tool._make_cache_key("spacex", datetime(2026, 3, 2, tzinfo=UTC))
        assert key1 != key2

    def test_none_from_date(self):
        tool, _ = _make_news_tool()
        key = tool._make_cache_key("spacex", None)
        assert "none" in key


class TestNewsToolRateLimitHandling:
    """Rate limit errors are logged at WARNING, not ERROR."""

    @pytest.mark.asyncio
    async def test_rate_limit_logs_warning_not_error(self, caplog):
        """NewsAPIException with rateLimited code is logged at WARNING level."""
        rate_limit_msg = (
            f'{{"status": "error", "code": "{NEWS_API_RATE_LIMITED_CODE}", '
            '"message": "Too many requests"}}'
        )
        tool, mock_client = _make_news_tool()
        mock_client.get_everything.side_effect = NewsAPIException(rate_limit_msg)

        with caplog.at_level(logging.WARNING, logger="penny.tools.news"):
            results = await tool.search(["spacex"])

        assert results == []
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(NEWS_API_RATE_LIMITED_CODE in r.message for r in warnings), (
            f"Expected WARNING with '{NEWS_API_RATE_LIMITED_CODE}' in message, got: "
            f"{[r.message for r in caplog.records]}"
        )
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert not errors, f"Expected no ERROR logs, got: {[r.message for r in errors]}"

    @pytest.mark.asyncio
    async def test_other_api_error_logs_error(self, caplog):
        """Non-rate-limit NewsAPIException is still logged at ERROR level."""
        tool, mock_client = _make_news_tool()
        mock_client.get_everything.side_effect = NewsAPIException(
            '{"status": "error", "code": "apiKeyInvalid", "message": "Invalid API key"}'
        )

        with caplog.at_level(logging.DEBUG, logger="penny.tools.news"):
            results = await tool.search(["spacex"])

        assert results == []
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert errors, "Expected at least one ERROR log for non-rate-limit API error"

    @pytest.mark.asyncio
    async def test_rate_limit_returns_empty_list(self):
        """Rate limit response returns empty list (graceful degradation)."""
        rate_limit_msg = f'{{"code": "{NEWS_API_RATE_LIMITED_CODE}"}}'
        tool, mock_client = _make_news_tool()
        mock_client.get_everything.side_effect = NewsAPIException(rate_limit_msg)

        results = await tool.search(["spacex"])

        assert results == []

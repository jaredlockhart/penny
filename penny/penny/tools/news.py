"""NewsAPI.org client for structured news article retrieval."""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from functools import partial

from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 6 * 3600  # 6 hours — conservative given 100 req/day developer quota
RATE_LIMIT_BACKOFF_SECONDS = 12 * 3600  # 12 hours — matches NewsAPI quota reset window
NEWS_API_RATE_LIMITED_CODE = "rateLimited"


class NewsArticle(BaseModel):
    """A structured news article from NewsAPI.org."""

    title: str
    description: str
    url: str
    published_at: datetime
    source_name: str
    url_to_image: str | None = None


class NewsTool:
    """Queries NewsAPI.org for structured news articles. Used by EventAgent directly.

    Caches successful responses for CACHE_TTL_SECONDS to conserve the developer
    quota (100 requests/24h). Rate limit errors are logged at WARNING, not ERROR,
    since they are an expected API state rather than an application failure.

    When a rate limit error is received, all subsequent requests are skipped for
    RATE_LIMIT_BACKOFF_SECONDS (12 hours) to avoid hammering the API while the
    quota is exhausted.
    """

    def __init__(self, api_key: str):
        self._client = NewsApiClient(api_key=api_key)
        self._cache: dict[str, tuple[datetime, list[NewsArticle]]] = {}
        self._rate_limited_until: datetime | None = None
        self._rate_limit_notification_pending: bool = False

    async def search(
        self,
        query_terms: list[str],
        from_date: datetime | None = None,
    ) -> list[NewsArticle]:
        """Search for news articles matching query terms.

        Returns cached results if the same query was fetched within CACHE_TTL_SECONDS.
        Returns empty list immediately if currently rate limited.

        Args:
            query_terms: Search terms to query (joined with OR).
            from_date: Oldest article date. Defaults to None (API default).

        Returns:
            List of NewsArticle results, or empty list on failure/rate-limit.
        """
        if self._is_rate_limited():
            return []

        query = " OR ".join(query_terms)
        cache_key = self._make_cache_key(query, from_date)

        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        articles = await self._fetch_articles(query, from_date)
        self._set_cached(cache_key, articles)
        return articles

    def consume_rate_limit_notification(self) -> bool:
        """Return True once when a rate limit was just triggered.

        Call this after search() to detect when the rate limit is first hit.
        Returns True the first time after a new rate limit event, False thereafter.
        Clears the pending flag on read so only one notification fires per window.
        """
        if self._rate_limit_notification_pending:
            self._rate_limit_notification_pending = False
            return True
        return False

    def _is_rate_limited(self) -> bool:
        """Return True if we are in a rate limit backoff window."""
        if self._rate_limited_until is None:
            return False
        if datetime.now(UTC) < self._rate_limited_until:
            logger.debug(
                "NewsTool: skipping request — rate limited until %s",
                self._rate_limited_until.isoformat(),
            )
            return True
        self._rate_limited_until = None
        return False

    def _make_cache_key(self, query: str, from_date: datetime | None) -> str:
        """Build a cache key from query and from_date (normalized to day)."""
        date_str = from_date.strftime("%Y-%m-%d") if from_date else "none"
        return f"{query}:{date_str}"

    def _get_cached(self, cache_key: str) -> list[NewsArticle] | None:
        """Return cached articles if they exist and haven't expired, else None."""
        entry = self._cache.get(cache_key)
        if entry is None:
            return None
        cached_at, articles = entry
        age_seconds = (datetime.now(UTC) - cached_at).total_seconds()
        if age_seconds >= CACHE_TTL_SECONDS:
            return None
        logger.debug("NewsTool: cache hit for query (age %.0fs)", age_seconds)
        return articles

    def _set_cached(self, cache_key: str, articles: list[NewsArticle]) -> None:
        """Store articles in cache with current timestamp."""
        self._cache[cache_key] = (datetime.now(UTC), articles)

    async def _fetch_articles(self, query: str, from_date: datetime | None) -> list[NewsArticle]:
        """Execute the API call in a thread and parse results."""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                partial(self._call_api, query, from_date),
            )
            return self._parse_articles(response)
        except NewsAPIException as e:
            if NEWS_API_RATE_LIMITED_CODE in str(e):
                self._rate_limited_until = datetime.now(UTC) + timedelta(
                    seconds=RATE_LIMIT_BACKOFF_SECONDS
                )
                self._rate_limit_notification_pending = True
                logger.warning(
                    "NewsAPI rate limit reached — backing off until %s",
                    self._rate_limited_until.isoformat(),
                )
            else:
                logger.error("NewsAPI error: %s", e)
            return []
        except Exception as e:
            logger.error("Unexpected error fetching news: %s", e)
            return []

    def _call_api(self, query: str, from_date: datetime | None) -> dict:
        """Synchronous API call (runs in executor)."""
        kwargs: dict = {
            "q": query,
            "language": "en",
            "sort_by": "publishedAt",
            "page_size": 20,
        }
        if from_date:
            kwargs["from_param"] = from_date.strftime("%Y-%m-%dT%H:%M:%S")
        return self._client.get_everything(**kwargs)

    def _parse_articles(self, response: dict) -> list[NewsArticle]:
        """Parse API response into NewsArticle models."""
        articles: list[NewsArticle] = []
        for raw in response.get("articles", []):
            article = self._parse_single_article(raw)
            if article:
                articles.append(article)
        return articles

    def _parse_single_article(self, raw: dict) -> NewsArticle | None:
        """Parse a single article dict, returning None if essential fields are missing."""
        title = raw.get("title")
        url = raw.get("url")
        published = raw.get("publishedAt")
        if not title or not url or not published:
            return None
        return NewsArticle(
            title=title,
            description=raw.get("description") or "",
            url=url,
            published_at=datetime.fromisoformat(published.replace("Z", "+00:00")),
            source_name=(raw.get("source") or {}).get("name", ""),
            url_to_image=raw.get("urlToImage"),
        )

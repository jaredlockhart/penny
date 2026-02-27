"""NewsAPI.org client for structured news article retrieval."""

import asyncio
import logging
from datetime import datetime
from functools import partial

from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class NewsArticle(BaseModel):
    """A structured news article from NewsAPI.org."""

    title: str
    description: str
    url: str
    published_at: datetime
    source_name: str
    url_to_image: str | None = None


class NewsTool:
    """Queries NewsAPI.org for structured news articles. Used by EventAgent directly."""

    def __init__(self, api_key: str):
        self._client = NewsApiClient(api_key=api_key)

    async def search(
        self,
        query_terms: list[str],
        from_date: datetime | None = None,
    ) -> list[NewsArticle]:
        """Search for news articles matching query terms.

        Args:
            query_terms: Search terms to query (joined with OR).
            from_date: Oldest article date. Defaults to None (API default).

        Returns:
            List of NewsArticle results, or empty list on failure.
        """
        query = " OR ".join(query_terms)
        return await self._fetch_articles(query, from_date)

    async def _fetch_articles(self, query: str, from_date: datetime | None) -> list[NewsArticle]:
        """Execute the API call in a thread and parse results."""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                partial(self._call_api, query, from_date),
            )
            return self._parse_articles(response)
        except NewsAPIException as e:
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

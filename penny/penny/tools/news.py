"""TheNewsAPI.com client for structured news article retrieval."""

import logging
from datetime import datetime

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

THE_NEWS_API_BASE_URL = "https://api.thenewsapi.com/v1/news/all"
THE_NEWS_API_PAGE_SIZE = 20


class _RawArticle(BaseModel):
    """Raw article shape from TheNewsAPI.com response."""

    title: str | None = None
    description: str | None = None
    snippet: str | None = None
    url: str | None = None
    image_url: str | None = None
    published_at: str | None = None
    source: str | None = None


class _ApiResponse(BaseModel):
    """TheNewsAPI.com response envelope."""

    data: list[_RawArticle] = Field(default_factory=list)


class NewsArticle(BaseModel):
    """A structured news article from TheNewsAPI.com."""

    title: str
    description: str
    url: str
    published_at: datetime
    source_name: str
    url_to_image: str | None = None


class NewsTool:
    """Queries TheNewsAPI.com for structured news articles. Used by EventAgent directly."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._http = httpx.AsyncClient(timeout=30.0)

    async def search(
        self,
        query_terms: list[str],
        from_date: datetime | None = None,
    ) -> list[NewsArticle]:
        """Search for news articles matching query terms.

        Args:
            query_terms: Search terms to query (joined with |).
            from_date: Oldest article date. Defaults to None (API default).

        Returns:
            List of NewsArticle results, or empty list on failure.
        """
        query = " | ".join(query_terms)
        return await self._fetch_articles(query, from_date)

    async def _fetch_articles(self, query: str, from_date: datetime | None) -> list[NewsArticle]:
        """Execute the API call and parse results."""
        try:
            response = await self._call_api(query, from_date)
            return self._parse_articles(response)
        except httpx.HTTPStatusError as e:
            logger.error("TheNewsAPI HTTP error %d: %s", e.response.status_code, e.response.text)
            return []
        except Exception as e:
            logger.error("Unexpected error fetching news: %s", e, exc_info=True)
            return []

    async def _call_api(self, query: str, from_date: datetime | None) -> _ApiResponse:
        """Call TheNewsAPI.com search endpoint."""
        params: dict[str, str | int] = {
            "api_token": self._api_key,
            "search": query,
            "language": "en",
            "sort": "published_at",
            "limit": THE_NEWS_API_PAGE_SIZE,
        }
        if from_date:
            params["published_after"] = from_date.strftime("%Y-%m-%dT%H:%M:%S")
        resp = await self._http.get(THE_NEWS_API_BASE_URL, params=params)
        resp.raise_for_status()
        return _ApiResponse.model_validate(resp.json())

    def _parse_articles(self, response: _ApiResponse) -> list[NewsArticle]:
        """Convert raw API articles into NewsArticle models."""
        articles: list[NewsArticle] = []
        for raw in response.data:
            article = self._to_news_article(raw)
            if article:
                articles.append(article)
        return articles

    def _to_news_article(self, raw: _RawArticle) -> NewsArticle | None:
        """Convert a raw API article, returning None if essential fields are missing."""
        if not raw.title or not raw.url or not raw.published_at:
            return None
        return NewsArticle(
            title=raw.title,
            description=raw.description or raw.snippet or "",
            url=raw.url,
            published_at=datetime.fromisoformat(raw.published_at.replace("Z", "+00:00")),
            source_name=raw.source or "",
            url_to_image=raw.image_url,
        )

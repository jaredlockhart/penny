"""Fetch news tool — search for recent news articles."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from penny.tools.base import Tool

if TYPE_CHECKING:
    from penny.tools.news import NewsTool

logger = logging.getLogger(__name__)


class FetchNewsTool(Tool):
    """Search for recent news articles via TheNewsAPI."""

    name = "fetch_news"
    description = (
        "Search for recent news articles. Requires a topic string — always specify the subject "
        "to search for (e.g. 'artificial intelligence', 'climate change'). "
        "Returns headlines, summaries, and URLs."
    )
    parameters = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": (
                    "The subject to search news for. Required — always provide a specific topic "
                    "(e.g. 'artificial intelligence', 'climate change', 'stock market')."
                ),
            }
        },
        "required": ["topic"],
    }

    def __init__(self, news_tool: NewsTool):
        self._news_tool = news_tool

    async def execute(self, **kwargs: Any) -> str:
        """Search for news and format results."""
        topic: str = kwargs["topic"]
        logger.info("[inner_monologue] fetch_news: %s", topic)
        articles = await self._news_tool.search(query_terms=[topic])
        if not articles:
            return f"No recent news found for '{topic}'."
        lines = []
        for article in articles[:10]:
            ts = article.published_at.strftime("%Y-%m-%d")
            line = f"[{ts}] {article.title}"
            if article.description:
                line += f"\n  {article.description[:150]}"
            if article.url:
                line += f"\n  {article.url}"
            lines.append(line)
        return "\n\n".join(lines)

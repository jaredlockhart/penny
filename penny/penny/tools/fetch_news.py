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
        "Search for recent news articles on a specific topic. "
        "Always infer the topic from the user's message context — for example, if they mention "
        "'Tesla' or 'climate change', use that as the topic. "
        "If the user's request is too vague to identify a clear topic (e.g. 'any news?'), "
        "ask them what topic they are interested in before calling this tool."
    )
    parameters = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": (
                    "The specific topic or keyword to search for "
                    "(e.g. 'climate change', 'SpaceX', 'US election'). "
                    "Must be a non-empty word or phrase derived from the user's message. "
                    "Do not call this tool without a clear topic from the conversation."
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

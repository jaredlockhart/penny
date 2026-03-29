"""MultiTool — dispatches heterogeneous parallel lookups via a single tool call.

Works around single-tool-call-per-turn limitations in models like gpt-oss:20b.
The model packs everything into a single queries array; the server detects URLs
and routes them to browse_url while plain text goes to search.
"""

from __future__ import annotations

import asyncio
import logging
import re
import urllib.parse
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from penny.tools.base import Tool
from penny.tools.models import MultiToolArgs, SearchResult

if TYPE_CHECKING:
    from penny.tools.browse_url import BrowseUrlTool
    from penny.tools.fetch_news import FetchNewsTool
    from penny.tools.search import SearchTool

logger = logging.getLogger(__name__)

_URL_PATTERN = re.compile(r"^https?://")


class MultiTool(Tool):
    """Single tool call that fans out queries to search or browse_url.

    The model emits one tool call with a queries array:
      {"queries": ["topic", "https://example.com", "another topic"]}
    URLs are detected and routed to browse_url; plain text goes to search.
    """

    name = "fetch"

    def __init__(
        self,
        search_tool: SearchTool | None = None,
        news_tool: FetchNewsTool | None = None,
        max_calls: int = 5,
    ):
        n = max_calls
        items = "query and/or URL" if n == 1 else "queries and/or URLs"
        self.description = f"Look things up. Pass up to {n} {items}."
        self.parameters = {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Think out loud about what you're looking up and why.",
                },
                "queries": {
                    "type": "array",
                    "description": f"Search queries and/or URLs to look up (max {n})",
                    "items": {"type": "string"},
                    "maxItems": n,
                },
            },
            "required": ["queries"],
        }
        self._search_tool = search_tool
        self._news_tool = news_tool
        self._max_calls = max_calls
        self._browse_url_provider: Callable[[], BrowseUrlTool | None] | None = None

    def set_browse_url_provider(self, provider: Callable[[], BrowseUrlTool | None]) -> None:
        """Set a provider that returns the current BrowseUrlTool (or None if disconnected)."""
        self._browse_url_provider = provider

    @property
    def redact_terms(self) -> list[str]:
        """Proxy redact_terms to the inner search tool."""
        return self._search_tool.redact_terms if self._search_tool else []

    @redact_terms.setter
    def redact_terms(self, terms: list[str]) -> None:
        if self._search_tool:
            self._search_tool.redact_terms = terms

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        """Format lookups into a readable status string."""
        parts: list[str] = []
        for q in arguments.get("queries", []):
            if _URL_PATTERN.match(q):
                short = q.replace("https://", "").replace("http://", "")
                parts.append(f"Reading {short[:50]}")
            else:
                parts.append(f'Searching "{q}"')
        return "<br>".join(parts) if parts else "Looking up..."

    async def execute(self, **kwargs: Any) -> SearchResult:
        """Dispatch all lookups in parallel — URLs to browse, text to search."""
        args = MultiToolArgs(**kwargs)

        cap = self._max_calls
        tasks: list[tuple[str, str, Any]] = []
        has_browser = bool(self._browse_url_provider and self._browse_url_provider())
        for q in args.queries[:cap]:
            if _URL_PATTERN.match(q):
                tasks.append(("browse_url", q, self._dispatch_browse(q)))
            elif has_browser:
                kagi_url = f"https://kagi.com/search?q={urllib.parse.quote(q)}"
                tasks.append(("search", q, self._dispatch_browse(kagi_url)))
            else:
                tasks.append(("search", q, self._dispatch_search(q)))

        results = await asyncio.gather(*[coro for _, _, coro in tasks], return_exceptions=True)

        sections: list[str] = []
        all_urls: list[str] = []
        for (kind, value, _), result in zip(tasks, results, strict=True):
            label = f"{kind}: {value}"
            if isinstance(result, Exception):
                logger.warning("MultiTool sub-call failed (%s): %s", label, result)
                sections.append(f"## {label}\nError: {result}")
            elif isinstance(result, SearchResult):
                all_urls.extend(result.urls)
                sections.append(f"## {label}\n{result.text}")
            else:
                sections.append(f"## {label}\n{result}")

        return SearchResult(text="\n\n---\n\n".join(sections), urls=all_urls)

    async def _dispatch_search(self, query: str) -> Any:
        """Run a single search query."""
        if not self._search_tool:
            return "Search not available."
        return await self._search_tool.execute(query=query)

    async def _dispatch_browse(self, url: str) -> Any:
        """Read a single URL."""
        browse_tool = self._browse_url_provider() if self._browse_url_provider else None
        if not browse_tool:
            return f"No browser connected — cannot read {url}."
        return await browse_tool.execute(url=url)

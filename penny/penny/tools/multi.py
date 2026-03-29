"""MultiTool — dispatches heterogeneous parallel lookups via a single tool call.

Works around single-tool-call-per-turn limitations in models like gpt-oss:20b.
The model packs all lookups into one tools call; the server fans them out in parallel.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from penny.tools.base import Tool
from penny.tools.models import InnerCall, MultiToolArgs, SearchResult

if TYPE_CHECKING:
    from penny.tools.browse_url import BrowseUrlTool
    from penny.tools.fetch_news import FetchNewsTool
    from penny.tools.search import SearchTool

logger = logging.getLogger(__name__)


class MultiTool(Tool):
    """Single tool call that fans out to search, browse_url, and fetch_news in parallel.

    The model emits one tool call with a tool_calls array of single-key objects:
      [{"search": "query"}, {"browse_url": "https://..."}, {"fetch_news": "topic"}]
    Each item is dispatched to its sub-tool concurrently.
    """

    name = "tools"
    description = "Run search, browse_url, and fetch_news lookups in parallel via a calls array."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Think out loud about what you're looking up and why.",
            },
            "calls": {
                "type": "array",
                "description": "Up to 5 lookups to run in parallel.",
                "items": {
                    "type": "object",
                    "description": (
                        '{"search": "query"} or '
                        '{"browse_url": "https://..."} or '
                        '{"fetch_news": "topic"}'
                    ),
                    "properties": {
                        "search": {"type": "string"},
                        "browse_url": {"type": "string"},
                        "fetch_news": {"type": "string"},
                    },
                },
            },
        },
        "required": ["calls"],
    }

    def __init__(
        self,
        search_tool: SearchTool | None = None,
        news_tool: FetchNewsTool | None = None,
    ):
        self._search_tool = search_tool
        self._news_tool = news_tool
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
        """Format inner calls into a readable status string."""
        calls = arguments.get("calls", [])
        parts = []
        for c in calls:
            if "search" in c:
                parts.append(f'Searching "{c["search"]}"')
            elif "browse_url" in c:
                url = c["browse_url"].replace("https://", "").replace("http://", "")
                parts.append(f"Reading {url[:50]}")
            elif "fetch_news" in c:
                parts.append(f"News: {c['fetch_news']}")
        return " + ".join(parts) if parts else "Looking up..."

    async def execute(self, **kwargs: Any) -> SearchResult:
        """Dispatch all inner calls in parallel and return labeled combined results."""
        args = MultiToolArgs(**kwargs)
        calls = args.calls[:5]  # enforce config cap

        results = await asyncio.gather(
            *[self._dispatch(call) for call in calls], return_exceptions=True
        )

        sections: list[str] = []
        all_urls: list[str] = []
        for call, result in zip(calls, results, strict=True):
            label = f"{call.tool_name}: {call.value}"
            if isinstance(result, Exception):
                logger.warning("MultiTool sub-call failed (%s): %s", label, result)
                sections.append(f"## {label}\nError: {result}")
            elif isinstance(result, SearchResult):
                all_urls.extend(result.urls)
                sections.append(f"## {label}\n{result.text}")
            else:
                sections.append(f"## {label}\n{result}")

        return SearchResult(text="\n\n---\n\n".join(sections), urls=all_urls)

    async def _dispatch(self, call: InnerCall) -> Any:
        """Route one inner call to the right sub-tool."""
        if call.search is not None:
            if not self._search_tool:
                return "Search not available."
            return await self._search_tool.execute(query=call.search)
        if call.browse_url is not None:
            browse_tool = self._browse_url_provider() if self._browse_url_provider else None
            if not browse_tool:
                return f"No browser connected — cannot read {call.browse_url}."
            return await browse_tool.execute(url=call.browse_url)
        if call.fetch_news is not None:
            if not self._news_tool:
                return "News not available."
            return await self._news_tool.execute(topic=call.fetch_news)
        return "No tool specified."

"""MultiTool — dispatches heterogeneous parallel lookups via a single tool call.

Works around single-tool-call-per-turn limitations in models like gpt-oss:20b.
The model packs everything into a single queries array; the server detects URLs
and routes them to browse_url while plain text goes to Kagi search.
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

logger = logging.getLogger(__name__)

_URL_PATTERN = re.compile(r"^https?://")


class MultiTool(Tool):
    """Single tool call that fans out queries to browse_url.

    The model emits one tool call with a queries array:
      {"queries": ["topic", "https://example.com", "another topic"]}
    URLs are routed to browse_url directly; plain text is converted
    to a Kagi search URL and browsed.
    """

    name = "fetch"

    def __init__(self, max_calls: int):
        self._max_calls = max_calls
        self._browse_url_provider: Callable[[], BrowseUrlTool | None] | None = None

    @property
    def description(self) -> str:  # type: ignore[override]
        """Dynamic description reflecting current max_calls."""
        n = self._max_calls
        items = "query and/or URL" if n == 1 else "queries and/or URLs"
        return f"Look things up. Pass up to {n} {items}."

    @property
    def parameters(self) -> dict[str, Any]:  # type: ignore[override]
        """Dynamic parameters reflecting current max_calls."""
        n = self._max_calls
        return {
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

    def set_browse_url_provider(self, provider: Callable[[], BrowseUrlTool | None]) -> None:
        """Set a provider that returns the current BrowseUrlTool (or None if disconnected)."""
        self._browse_url_provider = provider

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
        """Dispatch all lookups in parallel — URLs and text queries to browse_url."""
        args = MultiToolArgs(**kwargs)

        cap = self._max_calls
        tasks: list[tuple[str, str, Any]] = []
        for q in args.queries[:cap]:
            if _URL_PATTERN.match(q):
                tasks.append(("browse_url", q, self._dispatch_browse(q)))
            else:
                kagi_url = f"https://kagi.com/search?q={urllib.parse.quote(q)}"
                tasks.append(("search", q, self._dispatch_browse(kagi_url)))

        results = await asyncio.gather(*[coro for _, _, coro in tasks], return_exceptions=True)

        sections: list[str] = []
        all_urls: list[str] = []
        first_image: str | None = None
        for (kind, value, _), result in zip(tasks, results, strict=True):
            label = f"{kind}: {value}"
            if isinstance(result, Exception):
                logger.warning("MultiTool sub-call failed (%s): %s", label, result)
                sections.append(f"## {label}\nError: {result}")
            elif isinstance(result, SearchResult):
                all_urls.extend(result.urls)
                sections.append(f"## {label}\n{result.text}")
                if not first_image and result.image_base64:
                    first_image = result.image_base64
            else:
                sections.append(f"## {label}\n{result}")

        return SearchResult(
            text="\n\n---\n\n".join(sections),
            urls=all_urls,
            image_base64=first_image,
        )

    async def _dispatch_browse(self, url: str) -> Any:
        """Read a single URL."""
        browse_tool = self._browse_url_provider() if self._browse_url_provider else None
        if not browse_tool:
            return f"No browser connected — cannot read {url}."
        return await browse_tool.execute(url=url)

"""BrowseTool — searches and reads web pages via the browser extension.

The model packs everything into a single queries array; the tool detects URLs
and reads them directly, while plain text is converted to Kagi search URLs.
Queries are dispatched in parallel.
"""

from __future__ import annotations

import asyncio
import logging
import re
import urllib.parse
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from penny.constants import PennyConstants
from penny.tools.base import Tool
from penny.tools.content_cleaning import clean_browser_content
from penny.tools.models import BrowseArgs, SearchResult

if TYPE_CHECKING:
    from penny.channels.permission_manager import PermissionManager

logger = logging.getLogger(__name__)

_URL_PATTERN = re.compile(r"^https?://")

# Type alias for the browser request function
RequestFn = Callable[[str, dict], Awaitable[tuple[str, str | None]]]


class BrowseTool(Tool):
    """Search the web and read pages via the browser extension.

    The model emits one tool call with a queries array:
      {"queries": ["topic", "https://example.com", "another topic"]}
    URLs are read directly; plain text is converted to a Kagi search URL.
    All queries are dispatched in parallel.
    """

    name = "browse"

    def __init__(self, max_calls: int, search_url: str = "https://kagi.com/search?q="):
        self._max_calls = max_calls
        self._search_url = search_url
        self._browse_provider: Callable[[], tuple[RequestFn, PermissionManager] | None] | None = (
            None
        )

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

    def set_browse_provider(
        self,
        provider: Callable[[], tuple[RequestFn, PermissionManager] | None],
    ) -> None:
        """Set a provider that returns (request_fn, permission_manager) or None."""
        self._browse_provider = provider

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
        """Dispatch all lookups in parallel via the browser extension."""
        args = BrowseArgs(**kwargs)

        cap = self._max_calls
        tasks: list[tuple[str, str, Any]] = []
        for q in args.queries[:cap]:
            if _URL_PATTERN.match(q):
                tasks.append(("browse", q, self._read_page(q)))
            else:
                search_url = self._search_url + urllib.parse.quote(q)
                tasks.append(("search", q, self._read_page(search_url)))

        results = await asyncio.gather(*[coro for _, _, coro in tasks], return_exceptions=True)

        sections: list[str] = []
        all_urls: list[str] = []
        first_image: str | None = None
        for (kind, value, _), result in zip(tasks, results, strict=True):
            label = f"{kind}: {value}"
            if isinstance(result, Exception):
                logger.warning("Browse sub-call failed (%s): %s", label, result)
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

    async def _read_page(self, url: str) -> SearchResult | str:
        """Read a single URL via the browser extension, retrying on disconnect."""
        for attempt in range(1 + PennyConstants.BROWSE_RETRIES):
            connection = self._browse_provider() if self._browse_provider else None
            if not connection:
                if attempt < PennyConstants.BROWSE_RETRIES:
                    logger.info(
                        "No browser connection, retrying in %.0fs (%s)",
                        PennyConstants.BROWSE_RETRY_DELAY,
                        url,
                    )
                    await asyncio.sleep(PennyConstants.BROWSE_RETRY_DELAY)
                    continue
                return f"No browser connected — cannot read {url}."

            request_fn, permission_manager = connection
            await permission_manager.check_domain(url)

            try:
                text, image_url = await request_fn("browse_url", {"url": url})
            except ConnectionError:
                if attempt < PennyConstants.BROWSE_RETRIES:
                    logger.info(
                        "Browser disconnected, retrying in %.0fs (%s)",
                        PennyConstants.BROWSE_RETRY_DELAY,
                        url,
                    )
                    await asyncio.sleep(PennyConstants.BROWSE_RETRY_DELAY)
                    continue
                raise

            if not text.strip():
                return SearchResult(text=f"Page at {url} returned no content.")

            text = clean_browser_content(text)
            return SearchResult(text=text, image_base64=image_url)

        return f"No browser connected — cannot read {url}."

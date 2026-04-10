"""BrowseTool — searches and reads web pages via the browser extension.

The model packs everything into a single queries array; the tool detects URLs
and reads them directly, while plain text is converted to search URLs.
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
from penny.prompts import Prompt
from penny.tools.base import Tool
from penny.tools.content_cleaning import clean_browser_content
from penny.tools.models import BrowseArgs, SearchResult

if TYPE_CHECKING:
    from penny.channels.permission_manager import PermissionManager

logger = logging.getLogger(__name__)

_URL_PATTERN = re.compile(r"^https?://")

_LINK_RE = re.compile(r"^\s*\[([^\]]*)\]\(https?://(?:[^)\\]|\\.)*\)\s*$")

# Type alias for the browser request function
RequestFn = Callable[[str, dict], Awaitable[tuple[str, str | None]]]


def _trim_search_result(text: str, context_lines: int = 2) -> str:
    """Trim search result page to lines near standalone markdown links.

    Pipeline: filter to solo-link lines (drops knowledge panel prose),
    cap at MAX_SEARCH_LINKS, then keep context lines around each.
    """
    lines = text.split("\n")

    link_lines: list[int] = []
    for i, line in enumerate(lines):
        if _LINK_RE.match(line):
            link_lines.append(i)

    if not link_lines:
        return text

    keep: set[int] = set()
    for line_number in link_lines[: PennyConstants.MAX_SEARCH_LINKS]:
        for offset in range(-context_lines, context_lines + 1):
            idx = line_number + offset
            if 0 <= idx < len(lines):
                keep.add(idx)

    trimmed = "\n".join(lines[i] for i in sorted(keep))
    return f"{Prompt.SEARCH_RESULT_HEADER}\n\n{trimmed}"


class BrowseTool(Tool):
    """Search the web and read pages via the browser extension.

    The model emits one tool call with a queries array:
      {"queries": ["topic", "https://example.com", "another topic"]}
    URLs are read directly; plain text is converted to a search URL.
    All queries are dispatched in parallel.
    """

    name = "browse"

    def __init__(self, max_calls: int, search_url: str = "https://duckduckgo.com/?q="):
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

    @classmethod
    def to_progress_emoji(cls, arguments: dict) -> str:
        """Pick 📖 if any query is a URL (reading), 🔍 otherwise (searching)."""
        for q in arguments.get("queries", []):
            if _URL_PATTERN.match(q):
                return "\U0001f4d6"  # 📖
        return "\U0001f50d"  # 🔍

    async def execute(self, **kwargs: Any) -> SearchResult:
        """Dispatch all lookups in parallel via the browser extension."""
        args = BrowseArgs(**kwargs)

        cap = self._max_calls
        tasks: list[tuple[str, str, Any]] = []
        for q in args.queries[:cap]:
            if _URL_PATTERN.match(q):
                tasks.append((PennyConstants.BROWSE_PAGE_HEADER, q, self._read_page(q)))
            else:
                search_url = self._search_url + urllib.parse.quote(q)
                tasks.append((PennyConstants.BROWSE_SEARCH_HEADER, q, self._read_page(search_url)))

        results = await asyncio.gather(*[coro for _, _, coro in tasks], return_exceptions=True)

        sections: list[str] = []
        all_urls: list[str] = []
        first_image: str | None = None
        for (header, value, _), result in zip(tasks, results, strict=True):
            if isinstance(result, Exception):
                logger.warning("Browse sub-call failed (%s%s): %s", header, value, result)
                error_label = f"{PennyConstants.BROWSE_ERROR_HEADER}{value}"
                sections.append(f"{error_label}\nCould not read page: {result}")
                continue
            label = f"{header}{value}"
            text = result.text
            if header == PennyConstants.BROWSE_SEARCH_HEADER:
                text = _trim_search_result(text)
            all_urls.extend(result.urls)
            sections.append(f"{label}\n{text}")
            if not first_image and result.image_base64:
                first_image = result.image_base64

        return SearchResult(
            text=PennyConstants.SECTION_SEPARATOR.join(sections),
            urls=all_urls,
            image_base64=first_image,
        )

    async def _read_page(self, url: str) -> SearchResult:
        """Read a single URL via the browser extension, retrying with backoff on disconnect.

        Raises ConnectionError if no browser is reachable after all retries, and
        propagates any RuntimeError raised by the browser extension itself (which
        signals a structured failure: extraction failed, page never became ready,
        host permission denied, etc.).
        """
        for attempt in range(1 + PennyConstants.BROWSE_RETRIES):
            delay = PennyConstants.BROWSE_RETRY_DELAY * (2**attempt)
            connection = self._browse_provider() if self._browse_provider else None
            if not connection:
                if attempt < PennyConstants.BROWSE_RETRIES:
                    logger.info(
                        "No browser connection, retrying in %.0fs (%s)",
                        delay,
                        url,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise ConnectionError("no browser connected")

            request_fn, permission_manager = connection
            await permission_manager.check_domain(url)

            try:
                text, image_url = await request_fn("browse_url", {"url": url})
            except ConnectionError:
                if attempt < PennyConstants.BROWSE_RETRIES:
                    logger.info(
                        "Browser disconnected, retrying in %.0fs (%s)",
                        delay,
                        url,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

            text = clean_browser_content(text)
            return SearchResult(text=text, image_base64=image_url)

        raise ConnectionError("no browser connected")

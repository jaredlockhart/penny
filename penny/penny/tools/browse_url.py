"""Browse URL tool — fetches a web page via the browser extension."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from penny.tools.base import Tool
from penny.tools.models import BrowseUrlArgs, SearchResult

if TYPE_CHECKING:
    from penny.channels.permission_manager import PermissionManager

logger = logging.getLogger(__name__)


class BrowseUrlTool(Tool):
    """Open a web page in the browser and return its content.

    Checks domain permission before browsing. Content is sanitized and
    summarized by the BrowserChannel before reaching the agent context.
    """

    name = "browse_url"
    description = (
        "Open a web page in the user's browser and return a summary of its content. "
        "Uses the browser's full rendering engine and user session. "
        "The URL must be on the user's allowed domain list."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to browse",
            },
        },
        "required": ["url"],
    }

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        """Format URL into a readable status string."""
        url = arguments.get("url", "")
        return f"Reading {url}" if url else "Reading page"

    def __init__(
        self,
        request_fn: Callable[[str, dict], Awaitable[tuple[str, str | None]]],
        permission_manager: PermissionManager | None = None,
    ):
        self._request_fn = request_fn
        self._permission_manager = permission_manager

    async def execute(self, **kwargs: Any) -> SearchResult:
        """Check domain permission, then fetch the page via the browser."""
        args = BrowseUrlArgs(**kwargs)
        logger.info("browse_url: requesting %s", args.url)

        if self._permission_manager:
            await self._permission_manager.check_domain(args.url)

        text, image_url = await self._request_fn("browse_url", {"url": args.url})
        if not text.strip():
            return SearchResult(text=f"Page at {args.url} returned no content.")

        return SearchResult(text=text, image_base64=image_url)

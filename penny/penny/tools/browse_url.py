"""Browse URL tool — fetches a web page via the browser extension."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from penny.tools.base import Tool
from penny.tools.models import BrowseUrlArgs

logger = logging.getLogger(__name__)


class BrowseUrlTool(Tool):
    """Open a web page in the browser and return its content.

    Content is already sanitized and summarized by the BrowserChannel
    before it reaches this tool — no raw web content enters the agent context.
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

    def __init__(self, request_fn: Callable[[str, dict], Awaitable[str]]):
        self._request_fn = request_fn

    async def execute(self, **kwargs: Any) -> str:
        """Fetch the page via the browser. Content arrives pre-summarized."""
        args = BrowseUrlArgs(**kwargs)
        logger.info("browse_url: requesting %s", args.url)

        result = await self._request_fn("browse_url", {"url": args.url})
        if not result.strip():
            return f"Page at {args.url} returned no content."

        return result

"""Browse URL tool — fetches a web page via the browser extension."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from penny.tools.base import Tool
from penny.tools.models import BrowseUrlArgs

if TYPE_CHECKING:
    from penny.ollama.client import OllamaClient

logger = logging.getLogger(__name__)

SUMMARIZE_SYSTEM_PROMPT = (
    "Summarize the following web page content. "
    "Include key facts, names, dates, prices, and details. "
    "Output only the summary — no commentary about the task."
)

MAX_RAW_CHARS = 50_000


class BrowseUrlTool(Tool):
    """Open a web page in the browser and return a summary of its content."""

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

    def __init__(
        self,
        request_fn: Callable[[str, dict], Awaitable[str]],
        model_client: OllamaClient,
    ):
        self._request_fn = request_fn
        self._model_client = model_client

    async def execute(self, **kwargs: Any) -> str:
        """Fetch the page via the browser, then summarize in a sandboxed model call."""
        args = BrowseUrlArgs(**kwargs)
        logger.info("browse_url: requesting %s", args.url)

        raw_text = await self._request_fn("browse_url", {"url": args.url})
        if not raw_text.strip():
            return f"Page at {args.url} returned no content."

        summary = await self._sandboxed_summarize(raw_text, args.url)
        return summary

    async def _sandboxed_summarize(self, raw_text: str, url: str) -> str:
        """Summarize raw page content in a constrained model call.

        No tools, no user context, no preferences — just summarization.
        Raw untrusted content never enters the agent's main context.
        """
        truncated = raw_text[:MAX_RAW_CHARS]
        response = await self._model_client.chat(
            messages=[
                {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
                {"role": "user", "content": f"URL: {url}\n\n{truncated}"},
            ],
        )
        content = response.message.content if response.message else ""
        logger.info(
            "browse_url: summarized %s (%d chars → %d chars)", url, len(truncated), len(content)
        )
        return content.strip()

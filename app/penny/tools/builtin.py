"""Built-in tools."""

import asyncio
import base64
import logging
import time
from datetime import UTC, datetime
from functools import partial
from typing import Any

import httpx
from duckduckgo_search import DDGS
from perplexity import Perplexity
from perplexity.types.output_item import MessageOutputItem, SearchResultsOutputItem

from penny.constants import (
    IMAGE_DOWNLOAD_TIMEOUT,
    IMAGE_MAX_RESULTS,
    NO_RESULTS_TEXT,
    PERPLEXITY_PRESET,
)
from penny.tools.base import Tool
from penny.tools.models import SearchResult

logger = logging.getLogger(__name__)


class SearchTool(Tool):
    """Combined search tool: Perplexity for text, DuckDuckGo for images, run in parallel."""

    name = "search"
    description = (
        "Search the web for information and a relevant image. "
        "Use this for every message to research your answer. "
        "Returns search results text and attaches a relevant image."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            }
        },
        "required": ["query"],
    }

    def __init__(self, perplexity_api_key: str, db=None):
        self.perplexity = Perplexity(api_key=perplexity_api_key)
        self.db = db

    async def execute(self, **kwargs) -> Any:
        """Run Perplexity text search and DuckDuckGo image search in parallel."""
        query: str = kwargs["query"]
        text_result, image_result = await asyncio.gather(
            self._search_text(query),
            self._search_image(query),
            return_exceptions=True,
        )

        # Handle text result
        urls: list[str] = []
        if isinstance(text_result, Exception):
            text = f"Error performing search: {text_result}"
        else:
            text, urls = text_result

        # Handle image result
        if isinstance(image_result, Exception) or image_result is None:
            return SearchResult(text=text, urls=urls)

        return SearchResult(text=text, image_base64=image_result, urls=urls)

    async def _search_text(self, query: str) -> tuple[str, list[str]]:
        """Search via Perplexity. Returns (text, urls)."""
        start = time.time()

        today = datetime.now(UTC).strftime("%B %d, %Y")
        dated_query = f"[Today is {today}] {query}"

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            partial(
                self.perplexity.responses.create,
                preset=PERPLEXITY_PRESET,
                input=dated_query,
            ),
        )

        duration_ms = int((time.time() - start) * 1000)
        result = response.output_text if response.output_text else NO_RESULTS_TEXT

        # Extract citation URLs from response
        urls: list[str] = []
        for output in response.output:
            if isinstance(output, SearchResultsOutputItem):
                for r in output.results:
                    if r.url and r.url not in urls:
                        urls.append(r.url)
            elif isinstance(output, MessageOutputItem):
                for part in output.content:
                    for ann in part.annotations or []:
                        if ann.url and ann.url not in urls:
                            urls.append(ann.url)

        if self.db:
            self.db.log_search(query=query, response=result, duration_ms=duration_ms)

        return result, urls

    async def _search_image(self, query: str) -> str | None:
        """Search for an image via DuckDuckGo and return base64 data."""
        try:
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(
                None, partial(DDGS().images, query, max_results=IMAGE_MAX_RESULTS)
            )

            if not results:
                return None

            async with httpx.AsyncClient(
                timeout=IMAGE_DOWNLOAD_TIMEOUT, follow_redirects=True
            ) as client:
                for result in results:
                    image_url = result.get("image", "")
                    if not image_url:
                        continue
                    try:
                        resp = await client.get(image_url)
                        resp.raise_for_status()
                        content_type = resp.headers.get("content-type", "")
                        if "image" not in content_type:
                            continue
                        image_b64 = base64.b64encode(resp.content).decode()
                        mime = content_type.split(";")[0].strip()
                        return f"data:{mime};base64,{image_b64}"
                    except httpx.HTTPError:
                        logger.debug("Failed to download image: %s", image_url)
                        continue

            return None
        except Exception as e:
            logger.warning("Image search failed: %s", e)
            return None

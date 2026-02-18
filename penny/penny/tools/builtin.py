"""Built-in tools."""

import asyncio
import logging
import re
import time
from datetime import UTC, datetime
from functools import partial
from typing import Any

from perplexity import Perplexity
from perplexity.types.output_item import MessageOutputItem, SearchResultsOutputItem

from penny.constants import PennyConstants
from penny.responses import PennyResponse
from penny.tools.base import Tool
from penny.tools.image_search import search_image
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

    def __init__(self, perplexity_api_key: str, db=None, skip_images: bool = False):
        self.perplexity = Perplexity(api_key=perplexity_api_key)
        self.db = db
        self.redact_terms: list[str] = []
        self.skip_images = skip_images

    @staticmethod
    def _clean_text(raw_text: str) -> str:
        """Strip markdown formatting and citations from Perplexity results."""
        text = raw_text
        # Remove citations like [web:1], [page:2], [conversation_history:0]
        text = re.sub(r"\[[\w:]+(?::\d+)?\]", "", text)
        # Remove markdown headings
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Remove markdown bold/italic
        text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
        # Remove markdown bullet points
        text = re.sub(r"^[-*]\s+", "", text, flags=re.MULTILINE)
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    async def execute(self, **kwargs) -> Any:
        """Run Perplexity text search and optionally DuckDuckGo image search in parallel."""
        query: str = kwargs["query"]
        redacted_query = self._redact_query(query)

        if self.skip_images:
            text_result = await self._search_text(redacted_query)
            if isinstance(text_result, Exception):
                return SearchResult(text=PennyResponse.SEARCH_ERROR.format(error=text_result))
            text, urls = text_result
            return SearchResult(text=text, urls=urls)

        text_result, image_result = await asyncio.gather(
            self._search_text(redacted_query),
            self._search_image(redacted_query),
            return_exceptions=True,
        )

        # Handle text result
        urls: list[str] = []
        if isinstance(text_result, Exception):
            text = PennyResponse.SEARCH_ERROR.format(error=text_result)
        else:
            text, urls = text_result

        # Handle image result
        if isinstance(image_result, Exception) or image_result is None:
            return SearchResult(text=text, urls=urls)

        return SearchResult(text=text, image_base64=image_result, urls=urls)

    def _redact_query(self, query: str) -> str:
        """Remove redact_terms from query (case-insensitive, whole-word)."""
        redacted = query
        for term in self.redact_terms:
            if not term:
                continue
            redacted = re.sub(rf"\b{re.escape(term)}\b", "", redacted, flags=re.IGNORECASE)
        # Collapse extra whitespace left by redaction
        return re.sub(r"\s{2,}", " ", redacted).strip()

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
                preset=PennyConstants.PERPLEXITY_PRESET,
                input=dated_query,
            ),
        )

        duration_ms = int((time.time() - start) * 1000)
        raw_text = response.output_text if response.output_text else PennyResponse.NO_RESULTS_TEXT
        result = self._clean_text(raw_text)

        # Extract the most-cited URL from response annotations
        url_counts: dict[str, int] = {}
        for output in response.output:
            if isinstance(output, SearchResultsOutputItem):
                for r in output.results:
                    if r.url:
                        url_counts.setdefault(r.url, 0)
            elif isinstance(output, MessageOutputItem):
                for part in output.content:
                    for ann in part.annotations or []:
                        if ann.url:
                            url_counts[ann.url] = url_counts.get(ann.url, 0) + 1

        urls: list[str] = []
        if url_counts:
            filtered = {
                u: c
                for u, c in url_counts.items()
                if not any(domain in u for domain in PennyConstants.URL_BLOCKLIST_DOMAINS)
            }
            urls = sorted(filtered, key=filtered.get, reverse=True)[:5]  # type: ignore[arg-type]

        if self.db:
            self.db.log_search(query=query, response=result, duration_ms=duration_ms)

        return result, urls

    async def _search_image(self, query: str) -> str | None:
        """Search for an image via DuckDuckGo and return base64 data."""
        return await search_image(query)

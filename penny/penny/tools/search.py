"""Search tool — Perplexity text search."""

import asyncio
import logging
import re
import time
from functools import partial
from typing import Any

import perplexity
from perplexity import Perplexity
from perplexity.types.output_item import MessageOutputItem, SearchResultsOutputItem

from penny.constants import PennyConstants
from penny.responses import PennyResponse
from penny.tools.base import Tool
from penny.tools.models import SearchArgs, SearchResult

logger = logging.getLogger(__name__)


class SearchTool(Tool):
    """Search tool: runs a single Perplexity text search."""

    name = "search"

    def __init__(
        self,
        perplexity_api_key: str,
        db=None,
        *,
        default_trigger: str = PennyConstants.SearchTrigger.USER_MESSAGE,
    ):
        self.perplexity = Perplexity(api_key=perplexity_api_key)
        self.db = db
        self.redact_terms: list[str] = []
        self.default_trigger = default_trigger
        self.description = "Search the web for current information."
        self.parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
            },
            "required": ["query"],
        }

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        """Format search query into a readable status string."""
        try:
            return f'Searching for "{SearchArgs(**arguments).query}"'
        except Exception:
            return "Searching"

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
        """Run a single text search.

        Accepts optional kwargs beyond the tool schema (not exposed to the model):
            trigger: SearchTrigger value for log_search (default: user_message)
        """
        args = SearchArgs(**kwargs)
        trigger: str = kwargs.get("trigger", self.default_trigger)
        redacted = self._redact_query(args.query)
        text_result = await self._search_text(redacted, trigger)
        if isinstance(text_result, Exception):
            return SearchResult(text=PennyResponse.SEARCH_ERROR.format(error=text_result))
        text, urls = text_result
        return SearchResult(text=text, urls=urls)

    def _redact_query(self, query: str) -> str:
        """Remove redact_terms from query (case-insensitive, whole-word)."""
        redacted = query
        for term in self.redact_terms:
            if not term:
                continue
            redacted = re.sub(rf"\b{re.escape(term)}\b", "", redacted, flags=re.IGNORECASE)
        # Collapse extra whitespace left by redaction
        return re.sub(r"\s{2,}", " ", redacted).strip()

    async def _search_text(
        self,
        query: str,
        trigger: str = PennyConstants.SearchTrigger.USER_MESSAGE,
    ) -> tuple[str, list[str]]:
        """Search via Perplexity — summary method. Returns (text, urls)."""
        start = time.time()
        try:
            response = await self._call_perplexity(query)
        except perplexity.AuthenticationError as e:
            if self._is_quota_error(e):
                logger.error("Perplexity quota exceeded: %s", e)
                return PennyResponse.SEARCH_QUOTA_EXCEEDED, []
            raise
        duration_ms = int((time.time() - start) * 1000)
        raw_text = response.output_text if response.output_text else PennyResponse.NO_RESULTS_TEXT
        result = self._clean_text(raw_text)
        urls = self._extract_urls(response)
        self._log_search(query, result, duration_ms, trigger)
        return result, urls

    @staticmethod
    def _is_quota_error(e: perplexity.AuthenticationError) -> bool:
        """Return True if the AuthenticationError is a quota-exceeded error."""
        if not isinstance(e.body, dict):
            return False
        error = e.body.get("error")  # ty: ignore[invalid-argument-type]
        if not isinstance(error, dict):
            return False
        return error.get("type") == "insufficient_quota"  # ty: ignore[invalid-argument-type]

    async def _call_perplexity(self, query: str):
        """Call Perplexity API."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(
                self.perplexity.responses.create,
                preset=PennyConstants.PERPLEXITY_PRESET,
                input=f"{query}, with urls",
            ),
        )

    @staticmethod
    def _extract_urls(response) -> list[str]:
        """Extract the most-cited URLs from Perplexity response annotations."""
        url_counts: dict[str, int] = {}
        for output in response.output or []:
            if isinstance(output, SearchResultsOutputItem):
                for r in output.results or []:
                    if r.url:
                        url_counts.setdefault(r.url, 0)
            elif isinstance(output, MessageOutputItem):
                for part in output.content or []:
                    for ann in part.annotations or []:
                        if ann.url:
                            url_counts[ann.url] = url_counts.get(ann.url, 0) + 1
        if not url_counts:
            return []
        filtered = {
            u: c
            for u, c in url_counts.items()
            if not any(domain in u for domain in PennyConstants.URL_BLOCKLIST_DOMAINS)
        }
        return sorted(filtered, key=lambda u: filtered[u], reverse=True)[:5]

    def _log_search(
        self,
        query: str,
        result: str,
        duration_ms: int,
        trigger: str,
    ) -> None:
        """Log search to database if available."""
        if self.db:
            self.db.searches.log(
                query=query,
                response=result,
                duration_ms=duration_ms,
                trigger=trigger,
            )

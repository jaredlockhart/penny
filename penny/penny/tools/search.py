"""Search tool — Perplexity text search with optional Serper image search."""

import asyncio
import logging
import re
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

import perplexity as perplexity_sdk
from perplexity import Perplexity
from perplexity.types.output_item import MessageOutputItem, SearchResultsOutputItem

from penny.constants import PennyConstants
from penny.responses import PennyResponse
from penny.serper.client import search_image
from penny.tools.base import Tool
from penny.tools.models import SearchResult

logger = logging.getLogger(__name__)

QUOTA_STATE_FILENAME = "perplexity_quota_exceeded_at"


class SearchTool(Tool):
    """Combined search tool: Perplexity for text, Serper for images, run in parallel."""

    name = "search"
    _quota_exceeded_flag: ClassVar[bool] = False  # shared circuit breaker across all instances
    _quota_exceeded_at: ClassVar[datetime | None] = None  # when the breaker was tripped
    QUOTA_COOLDOWN_HOURS: ClassVar[int] = 24  # cooldown before retry
    description = (
        "Search the web for current information on a specific topic. "
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

    def __init__(
        self,
        perplexity_api_key: str,
        db=None,
        skip_images: bool = False,
        serper_api_key: str | None = None,
        *,
        image_max_results: int,
        image_download_timeout: float,
        default_trigger: str = PennyConstants.SearchTrigger.USER_MESSAGE,
        quota_state_file: Path | None = None,
    ):
        self.perplexity = Perplexity(api_key=perplexity_api_key)
        self.db = db
        # NOTE: do NOT set self._quota_exceeded here — that resets the shared ClassVar
        # every time a new SearchTool is created (e.g. /test command).
        self.redact_terms: list[str] = []
        self.skip_images = skip_images
        self.serper_api_key = serper_api_key
        self.image_max_results = image_max_results
        self.image_download_timeout = image_download_timeout
        self.default_trigger = default_trigger
        self.quota_state_file = quota_state_file
        self._restore_quota_state()

    def _restore_quota_state(self) -> None:
        """Restore circuit breaker from persistent file if within cooldown (survives restarts)."""
        if not self.quota_state_file or not self.quota_state_file.exists():
            return
        try:
            text = self.quota_state_file.read_text().strip()
            exceeded_at = datetime.fromisoformat(text)
            age_hours = (datetime.now(UTC) - exceeded_at).total_seconds() / 3600
            if age_hours < self.QUOTA_COOLDOWN_HOURS:
                SearchTool._quota_exceeded_flag = True
                SearchTool._quota_exceeded_at = exceeded_at
                logger.warning("Quota circuit breaker restored from file (%.1fh ago)", age_hours)
            else:
                self.quota_state_file.unlink(missing_ok=True)
                logger.info("Perplexity quota cooldown expired — search re-enabled")
        except Exception as e:
            logger.warning("Could not read quota state file %s: %s", self.quota_state_file, e)

    @property
    def _quota_exceeded(self) -> bool:
        """Class-level circuit breaker — shared across all SearchTool instances.

        Returns False once the cooldown period has elapsed, resetting the breaker.
        """
        if not SearchTool._quota_exceeded_flag:
            return False
        if SearchTool._quota_exceeded_at is None:
            return True  # flag set directly (e.g. tests) — treat as exceeded indefinitely
        age_hours = (datetime.now(UTC) - SearchTool._quota_exceeded_at).total_seconds() / 3600
        if age_hours >= self.QUOTA_COOLDOWN_HOURS:
            self._reset_quota_breaker()
            return False
        return True

    @_quota_exceeded.setter
    def _quota_exceeded(self, value: bool) -> None:
        if value:
            SearchTool._quota_exceeded_flag = True
            SearchTool._quota_exceeded_at = datetime.now(UTC)
            self._persist_quota_state()
        else:
            self._reset_quota_breaker()

    def _reset_quota_breaker(self) -> None:
        """Reset the circuit breaker after cooldown expires."""
        SearchTool._quota_exceeded_flag = False
        SearchTool._quota_exceeded_at = None
        if self.quota_state_file:
            self.quota_state_file.unlink(missing_ok=True)
        logger.info("Perplexity quota circuit breaker reset — search re-enabled")

    def _persist_quota_state(self) -> None:
        """Persist quota-exceeded timestamp to file for cross-restart durability."""
        if not self.quota_state_file or SearchTool._quota_exceeded_at is None:
            return
        try:
            self.quota_state_file.write_text(SearchTool._quota_exceeded_at.isoformat())
        except Exception as e:
            logger.warning("Could not write quota state file %s: %s", self.quota_state_file, e)

    @staticmethod
    def _is_quota_exceeded_error(e: perplexity_sdk.AuthenticationError) -> bool:
        """Return True only for insufficient_quota 401s (not invalid-key 401s)."""
        body = e.body
        if isinstance(body, dict):
            error = body.get("error")  # type: ignore[call-overload]
            if isinstance(error, dict):
                return error.get("type") == "insufficient_quota"
        return "insufficient_quota" in str(e).lower()

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
        """Run Perplexity text search and optionally Serper image search in parallel.

        Accepts optional kwargs beyond the tool schema (not exposed to the model):
            skip_images: Override instance default for this call
            trigger: SearchTrigger value for log_search (default: user_message)
        """
        query: str = kwargs["query"]
        skip_images: bool = kwargs.get("skip_images", self.skip_images)
        trigger: str = kwargs.get("trigger", self.default_trigger)
        redacted_query = self._redact_query(query)

        if skip_images:
            text_result = await self._search_text(redacted_query, trigger)
            if isinstance(text_result, Exception):
                return SearchResult(text=PennyResponse.SEARCH_ERROR.format(error=text_result))
            text, urls = text_result
            return SearchResult(text=text, urls=urls)

        text_result, image_result = await asyncio.gather(
            self._search_text(redacted_query, trigger),
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

    async def _search_text(
        self,
        query: str,
        trigger: str = PennyConstants.SearchTrigger.USER_MESSAGE,
    ) -> tuple[str, list[str]]:
        """Search via Perplexity — summary method. Returns (text, urls)."""
        if self._quota_exceeded:
            return PennyResponse.SEARCH_QUOTA_EXCEEDED, []
        start = time.time()
        try:
            response = await self._call_perplexity(query)
        except perplexity_sdk.AuthenticationError as e:
            if not self._is_quota_exceeded_error(e):
                raise
            self._quota_exceeded = True
            logger.warning("Perplexity quota exceeded — circuit breaker tripped: %s", e)
            return PennyResponse.SEARCH_QUOTA_EXCEEDED, []
        duration_ms = int((time.time() - start) * 1000)
        raw_text = response.output_text if response.output_text else PennyResponse.NO_RESULTS_TEXT
        result = self._clean_text(raw_text)
        urls = self._extract_urls(response)
        self._log_search(query, result, duration_ms, trigger)
        return result, urls

    async def _call_perplexity(self, query: str):
        """Call Perplexity API with dated query prefix."""
        today = datetime.now(UTC).strftime("%B %d, %Y")
        dated_query = f"[Today is {today}] {query}"
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(
                self.perplexity.responses.create,
                preset=PennyConstants.PERPLEXITY_PRESET,
                input=dated_query,
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
        return sorted(filtered, key=filtered.get, reverse=True)[:5]  # type: ignore[arg-type]

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

    async def _search_image(self, query: str) -> str | None:
        """Search for an image via Serper and return base64 data."""
        return await search_image(
            query,
            api_key=self.serper_api_key,
            max_results=self.image_max_results,
            timeout=self.image_download_timeout,
        )

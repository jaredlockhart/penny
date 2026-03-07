"""Search tool — Perplexity text search with optional Serper image search."""

import asyncio
import logging
import re
import time
from datetime import UTC, datetime
from functools import partial
from typing import Any, cast

from perplexity import AuthenticationError, Perplexity
from perplexity.types.output_item import MessageOutputItem, SearchResultsOutputItem

from penny.constants import PennyConstants
from penny.responses import PennyResponse
from penny.serper.client import search_image
from penny.tools.base import Tool
from penny.tools.models import SearchResult

logger = logging.getLogger(__name__)


class SearchTool(Tool):
    """Combined search tool: Perplexity for text, Serper for images, run in parallel."""

    name = "search"
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

    _QUOTA_RETRY_SECONDS: float = 3600.0  # retry Perplexity after 1 hour
    # RuntimeConfig keys used to persist quota state across restarts.
    _DB_QUOTA_KEY: str = "perplexity_quota_exceeded_at"
    # Separate key for the "ever exceeded" flag — persists across successful calls and
    # restarts so recurring quota exhaustions always log at WARNING, not ERROR.
    _DB_EVER_EXCEEDED_KEY: str = "perplexity_quota_ever_exceeded"

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
    ):
        self.perplexity = Perplexity(api_key=perplexity_api_key)
        self.db = db
        self.redact_terms: list[str] = []
        self.skip_images = skip_images
        self.serper_api_key = serper_api_key
        self.image_max_results = image_max_results
        self.image_download_timeout = image_download_timeout
        self.default_trigger = default_trigger
        # Timestamp (time.time()) when quota was first exceeded; None = not exceeded.
        # Resets automatically after _QUOTA_RETRY_SECONDS so search self-recovers when
        # quota is replenished without requiring a process restart.
        self._quota_exceeded_at: float | None = None
        # True once quota has been exceeded at least once. NOT cleared on success —
        # persisted via _DB_EVER_EXCEEDED_KEY so all future quota errors log at WARNING
        # rather than ERROR, preventing the monitor from filing repetitive bug reports.
        self._quota_ever_exceeded: bool = False
        self._load_quota_state()

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
        if self._is_quota_exceeded():
            return PennyResponse.SEARCH_QUOTA_EXCEEDED, []
        start = time.time()
        try:
            response = await self._call_perplexity(query)
        except AuthenticationError as e:
            return self._handle_auth_error(e), []
        self._clear_quota_state()
        duration_ms = int((time.time() - start) * 1000)
        raw_text = response.output_text if response.output_text else PennyResponse.NO_RESULTS_TEXT
        result = self._clean_text(raw_text)
        urls = self._extract_urls(response)
        self._log_search(query, result, duration_ms, trigger)
        return result, urls

    def _is_quota_exceeded(self) -> bool:
        """Return True if quota is exceeded and the retry window has not elapsed."""
        if self._quota_exceeded_at is None:
            return False
        if time.time() - self._quota_exceeded_at < self._QUOTA_RETRY_SECONDS:
            return True
        # Retry window elapsed — reset the in-memory circuit so we try Perplexity again.
        # We intentionally do NOT clear the DB entry here: if the process restarts before
        # the retry attempt, _load_quota_state() will restore _quota_ever_exceeded=True,
        # ensuring subsequent failures log at WARNING rather than ERROR. The DB timestamp
        # entry is only removed after a successful Perplexity call (in _search_text).
        logger.info("Perplexity quota retry window elapsed — resetting circuit breaker")
        self._quota_exceeded_at = None
        return False

    def _handle_auth_error(self, e: AuthenticationError) -> str:
        """Return user-friendly message and trip circuit-breaker on quota errors."""
        body = cast(Any, e.body)
        if isinstance(body, dict):
            error_info = body.get("error")
            if isinstance(error_info, dict) and error_info.get("type") == "insufficient_quota":
                if self._quota_ever_exceeded:
                    # Quota exceeded again — known recurring issue. Log at WARNING so
                    # the monitor doesn't file a new bug report each time.
                    logger.warning(
                        "Perplexity quota still exceeded after retry window — "
                        "disabling for another %.0f seconds",
                        self._QUOTA_RETRY_SECONDS,
                    )
                else:
                    logger.error(
                        "Perplexity quota exceeded — search disabled for %.0f seconds",
                        self._QUOTA_RETRY_SECONDS,
                    )
                self._quota_exceeded_at = time.time()
                self._quota_ever_exceeded = True
                self._persist_quota_exceeded()
                return PennyResponse.SEARCH_QUOTA_EXCEEDED
        logger.error("Perplexity authentication error: %s", e)
        return PennyResponse.SEARCH_AUTH_FAILED

    def _load_quota_state(self) -> None:
        """Load persisted quota state from DB on startup."""
        if self.db is None:
            return
        from sqlmodel import Session, select

        from penny.database.models import RuntimeConfig

        try:
            with Session(self.db.engine) as session:
                ts_row = session.exec(
                    select(RuntimeConfig).where(RuntimeConfig.key == self._DB_QUOTA_KEY)
                ).first()
                ever_row = session.exec(
                    select(RuntimeConfig).where(RuntimeConfig.key == self._DB_EVER_EXCEEDED_KEY)
                ).first()
                if ts_row:
                    self._quota_exceeded_at = float(ts_row.value)
                    self._quota_ever_exceeded = True
                    logger.info(
                        "Restored Perplexity quota circuit from DB (exceeded at %.0f)",
                        float(ts_row.value),
                    )
                elif ever_row:
                    # Active circuit already cleared (successful call), but flag persists.
                    self._quota_ever_exceeded = True
                    logger.info("Restored Perplexity quota ever-exceeded flag from DB")
        except Exception as ex:
            logger.warning("Failed to load quota state from DB: %s", ex)

    def _persist_quota_exceeded(self) -> None:
        """Persist quota-exceeded timestamp and ever-exceeded flag to DB."""
        if self.db is None or self._quota_exceeded_at is None:
            return
        from sqlmodel import Session, select

        from penny.database.models import RuntimeConfig

        try:
            with Session(self.db.engine) as session:
                # Persist active circuit timestamp.
                ts_existing = session.exec(
                    select(RuntimeConfig).where(RuntimeConfig.key == self._DB_QUOTA_KEY)
                ).first()
                if ts_existing:
                    ts_existing.value = str(self._quota_exceeded_at)
                    ts_existing.updated_at = datetime.now(UTC)
                    session.add(ts_existing)
                else:
                    session.add(
                        RuntimeConfig(
                            key=self._DB_QUOTA_KEY,
                            value=str(self._quota_exceeded_at),
                            description="Perplexity quota-exceeded timestamp (auto-managed)",
                            updated_at=datetime.now(UTC),
                        )
                    )
                # Persist ever-exceeded flag (separate key, never auto-deleted).
                ever_existing = session.exec(
                    select(RuntimeConfig).where(RuntimeConfig.key == self._DB_EVER_EXCEEDED_KEY)
                ).first()
                if not ever_existing:
                    session.add(
                        RuntimeConfig(
                            key=self._DB_EVER_EXCEEDED_KEY,
                            value="1",
                            description="Perplexity quota ever exceeded (auto-managed)",
                            updated_at=datetime.now(UTC),
                        )
                    )
                session.commit()
        except Exception as ex:
            logger.warning("Failed to persist quota state to DB: %s", ex)

    def _clear_quota_state(self) -> None:
        """Remove the active circuit timestamp from DB after a successful Perplexity call.

        The ever-exceeded flag (_DB_EVER_EXCEEDED_KEY) is intentionally kept so that
        _quota_ever_exceeded is restored as True on the next process restart, ensuring
        future quota errors log at WARNING rather than ERROR.
        """
        if self.db is None:
            return
        from sqlmodel import Session, select

        from penny.database.models import RuntimeConfig

        try:
            with Session(self.db.engine) as session:
                row = session.exec(
                    select(RuntimeConfig).where(RuntimeConfig.key == self._DB_QUOTA_KEY)
                ).first()
                if row:
                    session.delete(row)
                    session.commit()
        except Exception as ex:
            logger.warning("Failed to clear quota state from DB: %s", ex)

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

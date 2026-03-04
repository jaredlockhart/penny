"""Search store — logging and extraction tracking for search results."""

import logging

from sqlmodel import Session, select

from penny.database.models import SearchLog

logger = logging.getLogger(__name__)


class SearchStore:
    """Manages SearchLog records: creation, lookup, and extraction tracking."""

    def __init__(self, engine):
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    def log(
        self,
        query: str,
        response: str,
        duration_ms: int | None = None,
        trigger: str = "user_message",
    ) -> None:
        """Log a Perplexity search call."""
        try:
            with self._session() as session:
                log = SearchLog(
                    query=query,
                    response=response,
                    duration_ms=duration_ms,
                    trigger=trigger,
                )
                session.add(log)
                session.commit()
                logger.debug("Logged search query: %s", query[:50])
        except Exception as e:
            logger.error("Failed to log search: %s", e)

    def get(self, search_log_id: int) -> SearchLog | None:
        """Get a SearchLog by ID."""
        with self._session() as session:
            return session.get(SearchLog, search_log_id)

    def get_unprocessed(self, limit: int) -> list[SearchLog]:
        """Get unextracted SearchLog entries (oldest first)."""
        with self._session() as session:
            return list(
                session.exec(
                    select(SearchLog)
                    .where(SearchLog.extracted == False)  # noqa: E712
                    .order_by(
                        SearchLog.timestamp.asc(),  # type: ignore[unresolved-attribute]
                    )
                    .limit(limit)
                ).all()
            )

    def mark_extracted(self, search_log_id: int) -> None:
        """Mark a SearchLog entry as processed for entity extraction."""
        try:
            with self._session() as session:
                search_log = session.get(SearchLog, search_log_id)
                if search_log:
                    search_log.extracted = True
                    session.add(search_log)
                    session.commit()
        except Exception as e:
            logger.error("Failed to mark search %d as extracted: %s", search_log_id, e)

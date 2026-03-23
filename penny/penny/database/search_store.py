"""Search store — logging for search results."""

import logging

from sqlmodel import Session

from penny.database.models import SearchLog

logger = logging.getLogger(__name__)


class SearchStore:
    """Manages SearchLog records: creation and logging."""

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

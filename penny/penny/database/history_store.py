"""History store — conversation topic summaries for long-term context."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlmodel import Session, select

from penny.database.models import ConversationHistory

logger = logging.getLogger(__name__)


class HistoryStore:
    """Manages ConversationHistory records: daily topic summaries of past conversations."""

    def __init__(self, engine):
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    def add(
        self,
        user: str,
        period_start: datetime,
        period_end: datetime,
        duration: str,
        topics: str,
        embedding: bytes | None = None,
    ) -> ConversationHistory | None:
        """Insert a history entry. Returns the created record or None."""
        try:
            with self._session() as session:
                entry = ConversationHistory(
                    user=user,
                    period_start=period_start,
                    period_end=period_end,
                    duration=duration,
                    topics=topics,
                    embedding=embedding,
                    created_at=datetime.now(UTC),
                )
                session.add(entry)
                session.commit()
                session.refresh(entry)
                logger.debug("History entry added for %s (%s)", user, duration)
                return entry
        except Exception as e:
            logger.error("Failed to add history entry: %s", e)
            return None

    def get_latest(self, user: str, duration: str) -> ConversationHistory | None:
        """Get the most recent entry of a given duration for a user."""
        with self._session() as session:
            return session.exec(
                select(ConversationHistory)
                .where(
                    ConversationHistory.user == user,
                    ConversationHistory.duration == duration,
                )
                .order_by(ConversationHistory.period_start.desc())  # type: ignore[unresolved-attribute]
                .limit(1)
            ).first()

    def get_recent(self, user: str, duration: str, limit: int = 14) -> list[ConversationHistory]:
        """Get recent entries for context injection, oldest first."""
        with self._session() as session:
            entries = list(
                session.exec(
                    select(ConversationHistory)
                    .where(
                        ConversationHistory.user == user,
                        ConversationHistory.duration == duration,
                    )
                    .order_by(ConversationHistory.period_start.desc())  # type: ignore[unresolved-attribute]
                    .limit(limit)
                ).all()
            )
            entries.reverse()
            return entries

    def upsert(
        self,
        user: str,
        period_start: datetime,
        period_end: datetime,
        duration: str,
        topics: str,
        embedding: bytes | None = None,
    ) -> ConversationHistory | None:
        """Create or update a history entry for the given period."""
        try:
            with self._session() as session:
                existing = session.exec(
                    select(ConversationHistory).where(
                        ConversationHistory.user == user,
                        ConversationHistory.period_start == period_start,
                        ConversationHistory.duration == duration,
                    )
                ).first()
                if existing:
                    existing.topics = topics
                    existing.embedding = embedding
                    existing.created_at = datetime.now(UTC)
                    session.add(existing)
                    session.commit()
                    session.refresh(existing)
                    return existing
                return self.add(
                    user, period_start, period_end, duration, topics, embedding=embedding
                )
        except Exception as e:
            logger.error("Failed to upsert history entry: %s", e)
            return None

    def exists(self, user: str, period_start: datetime, duration: str) -> bool:
        """Check if an entry already exists for a given period."""
        with self._session() as session:
            result = session.exec(
                select(ConversationHistory)
                .where(
                    ConversationHistory.user == user,
                    ConversationHistory.period_start == period_start,
                    ConversationHistory.duration == duration,
                )
                .limit(1)
            ).first()
            return result is not None

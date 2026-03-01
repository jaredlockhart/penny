"""Event store — CRUD and queries for time-aware events."""

import logging
from datetime import UTC, datetime, timedelta

from sqlmodel import Session, select

from penny.database.models import Event

logger = logging.getLogger(__name__)


class EventStore:
    """Manages Event records: creation, dedup, entity linking, and notification tracking."""

    def __init__(self, engine):
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    def add(
        self,
        user: str,
        headline: str,
        summary: str,
        occurred_at: datetime,
        source_type: str,
        source_url: str | None = None,
        external_id: str | None = None,
        embedding: bytes | None = None,
        follow_prompt_id: int | None = None,
    ) -> Event | None:
        """Create a new event. Returns the created Event, or None on failure."""
        try:
            with self._session() as session:
                event = Event(
                    user=user,
                    headline=headline,
                    summary=summary,
                    occurred_at=occurred_at,
                    source_type=source_type,
                    source_url=source_url,
                    external_id=external_id,
                    embedding=embedding,
                    follow_prompt_id=follow_prompt_id,
                )
                session.add(event)
                session.commit()
                session.refresh(event)
                logger.debug("Created event for user %s: %s", user, headline[:50])
                return event
        except Exception as e:
            logger.error("Failed to create event: %s", e)
            return None

    def get(self, event_id: int) -> Event | None:
        """Get an event by ID."""
        with self._session() as session:
            return session.get(Event, event_id)

    def get_for_user(self, user: str, limit: int = 50) -> list[Event]:
        """Get events for a user, ordered by occurred_at descending."""
        with self._session() as session:
            return list(
                session.exec(
                    select(Event)
                    .where(Event.user == user)
                    .order_by(Event.occurred_at.desc())  # type: ignore[unresolved-attribute]
                    .limit(limit)
                ).all()
            )

    def get_by_external_id(self, external_id: str) -> Event | None:
        """Get an event by its external ID (for dedup)."""
        with self._session() as session:
            return session.exec(select(Event).where(Event.external_id == external_id)).first()

    def get_recent(self, user: str, days: int = 7) -> list[Event]:
        """Get recent events within a time window (for dedup)."""
        cutoff = datetime.now(UTC) - timedelta(days=days)
        with self._session() as session:
            return list(
                session.exec(
                    select(Event)
                    .where(Event.user == user, Event.discovered_at >= cutoff)
                    .order_by(Event.occurred_at.desc())  # type: ignore[unresolved-attribute]
                ).all()
            )

    def get_unnotified(self, user: str) -> list[Event]:
        """Get events that haven't been communicated to the user yet."""
        with self._session() as session:
            return list(
                session.exec(
                    select(Event)
                    .where(Event.user == user, Event.notified_at == None)  # noqa: E711
                    .order_by(Event.occurred_at.desc())  # type: ignore[unresolved-attribute]
                ).all()
            )

    def get_unnotified_for_follow_prompt(self, follow_prompt_id: int) -> list[Event]:
        """Get unnotified events for a specific follow prompt."""
        with self._session() as session:
            return list(
                session.exec(
                    select(Event)
                    .where(
                        Event.follow_prompt_id == follow_prompt_id,
                        Event.notified_at == None,  # noqa: E711
                    )
                    .order_by(Event.occurred_at.desc())  # type: ignore[unresolved-attribute]
                ).all()
            )

    def delete_for_follow_prompt(self, follow_prompt_id: int) -> int:
        """Delete all events for a follow prompt.

        Returns the number of events deleted.
        """
        try:
            with self._session() as session:
                events = list(
                    session.exec(
                        select(Event).where(Event.follow_prompt_id == follow_prompt_id)
                    ).all()
                )
                if not events:
                    return 0
                for event in events:
                    session.delete(event)
                session.commit()
                logger.info(
                    "Deleted %d events for follow prompt %d",
                    len(events),
                    follow_prompt_id,
                )
                return len(events)
        except Exception as e:
            logger.error("Failed to delete events for follow prompt %d: %s", follow_prompt_id, e)
            return 0

    def mark_notified(self, event_ids: list[int]) -> None:
        """Mark events as notified (communicated to user)."""
        if not event_ids:
            return
        try:
            now = datetime.now(UTC)
            with self._session() as session:
                for event_id in event_ids:
                    event = session.get(Event, event_id)
                    if event:
                        event.notified_at = now
                        session.add(event)
                session.commit()
                logger.debug("Marked %d events as notified", len(event_ids))
        except Exception as e:
            logger.error("Failed to mark events as notified: %s", e)

    def update_embedding(self, event_id: int, embedding: bytes) -> None:
        """Update the embedding for an event."""
        try:
            with self._session() as session:
                event = session.get(Event, event_id)
                if event:
                    event.embedding = embedding
                    session.add(event)
                    session.commit()
        except Exception as e:
            logger.error("Failed to update event %d embedding: %s", event_id, e)

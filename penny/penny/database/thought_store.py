"""Thought store — persistent inner monologue entries."""

import logging
from datetime import UTC, datetime, timedelta

from sqlmodel import Session, select

from penny.database.models import Thought

logger = logging.getLogger(__name__)


class ThoughtStore:
    """Manages Thought records: append-only log of Penny's inner monologue."""

    def __init__(self, engine):
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    def add(self, user: str, content: str, preference_id: int | None = None) -> Thought | None:
        """Append a thought to the log. Returns the created Thought or None."""
        try:
            with self._session() as session:
                thought = Thought(
                    user=user,
                    content=content,
                    preference_id=preference_id,
                    created_at=datetime.now(UTC),
                )
                session.add(thought)
                session.commit()
                session.refresh(thought)
                logger.debug("Thought logged for %s: %s", user, content[:80])
                return thought
        except Exception as e:
            logger.error("Failed to log thought: %s", e)
            return None

    def get_recent(self, user: str, limit: int = 50) -> list[Thought]:
        """Get recent thoughts for a user, oldest first (chronological order)."""
        with self._session() as session:
            thoughts = list(
                session.exec(
                    select(Thought)
                    .where(Thought.user == user)
                    .order_by(Thought.created_at.desc())  # type: ignore[unresolved-attribute]
                    .limit(limit)
                ).all()
            )
            thoughts.reverse()
            return thoughts

    def get_recent_by_preference(self, user: str, preference_id: int) -> list[Thought]:
        """Get all thoughts for a user seeded by a specific preference, oldest first."""
        with self._session() as session:
            thoughts = list(
                session.exec(
                    select(Thought)
                    .where(Thought.user == user, Thought.preference_id == preference_id)
                    .order_by(Thought.created_at.desc())  # type: ignore[unresolved-attribute]
                ).all()
            )
            thoughts.reverse()
            return thoughts

    @staticmethod
    def _freshness_cutoff(hours: int) -> datetime:
        """Rolling cutoff: now minus N hours, as naive UTC."""
        return datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=hours)

    def get_next_unnotified(self, user: str) -> Thought | None:
        """Get the oldest un-notified thought."""
        with self._session() as session:
            return session.exec(
                select(Thought)
                .where(
                    Thought.user == user,
                    Thought.notified_at == None,  # noqa: E711
                )
                .order_by(Thought.created_at.asc())  # type: ignore[unresolved-attribute]
                .limit(1)
            ).first()

    def get_all_unnotified(self, user: str) -> list[Thought]:
        """Get all un-notified thoughts, oldest first."""
        with self._session() as session:
            return list(
                session.exec(
                    select(Thought)
                    .where(
                        Thought.user == user,
                        Thought.notified_at == None,  # noqa: E711
                    )
                    .order_by(Thought.created_at.asc())  # type: ignore[unresolved-attribute]
                ).all()
            )

    def get_recent_notified(
        self, user: str, freshness_hours: int = 24, limit: int = 10
    ) -> list[Thought]:
        """Get recently notified thoughts within the freshness window, newest first."""
        cutoff = self._freshness_cutoff(freshness_hours)
        with self._session() as session:
            return list(
                session.exec(
                    select(Thought)
                    .where(
                        Thought.user == user,
                        Thought.notified_at != None,  # noqa: E711
                        Thought.created_at >= cutoff,
                    )
                    .order_by(Thought.notified_at.desc())  # type: ignore[unresolved-attribute]
                    .limit(limit)
                ).all()
            )

    def mark_notified(self, thought_id: int) -> None:
        """Mark a thought as notified (shared with user)."""
        try:
            with self._session() as session:
                thought = session.get(Thought, thought_id)
                if thought:
                    thought.notified_at = datetime.now(UTC)
                    session.add(thought)
                    session.commit()
                    logger.debug("Marked thought %d as notified", thought_id)
        except Exception as e:
            logger.error("Failed to mark thought %d as notified: %s", thought_id, e)

    def count(self, user: str) -> int:
        """Count total thoughts for a user."""
        with self._session() as session:
            from sqlalchemy import func

            return session.exec(
                select(func.count()).select_from(Thought).where(Thought.user == user)
            ).one()

"""Thought store — persistent inner monologue entries."""

import logging
from datetime import UTC, datetime

from sqlmodel import Session, select

from penny.database.models import Thought

logger = logging.getLogger(__name__)


class ThoughtStore:
    """Manages Thought records: append-only log of Penny's inner monologue."""

    def __init__(self, engine):
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    def add(self, user: str, content: str) -> Thought | None:
        """Append a thought to the log. Returns the created Thought or None."""
        try:
            with self._session() as session:
                thought = Thought(
                    user=user,
                    content=content,
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

    def count(self, user: str) -> int:
        """Count total thoughts for a user."""
        with self._session() as session:
            from sqlalchemy import func

            return session.exec(
                select(func.count()).select_from(Thought).where(Thought.user == user)
            ).one()

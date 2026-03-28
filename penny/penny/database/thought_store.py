"""Thought store — persistent inner monologue entries."""

import logging
from datetime import UTC, datetime

from sqlmodel import Session, func, select

from penny.database.models import Thought

logger = logging.getLogger(__name__)


class ThoughtStore:
    """Manages Thought records: append-only log of Penny's inner monologue."""

    def __init__(self, engine):
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    def add(
        self,
        user: str,
        content: str,
        preference_id: int | None = None,
        embedding: bytes | None = None,
        title: str | None = None,
        title_embedding: bytes | None = None,
        image_url: str | None = None,
    ) -> Thought | None:
        """Append a thought to the log. Returns the created Thought or None."""
        try:
            with self._session() as session:
                thought = Thought(
                    user=user,
                    content=content,
                    preference_id=preference_id,
                    embedding=embedding,
                    title=title,
                    title_embedding=title_embedding,
                    image_url=image_url,
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

    def get_by_id(self, thought_id: int) -> Thought | None:
        """Get a thought by its primary key."""
        with self._session() as session:
            return session.get(Thought, thought_id)

    def get_recent(self, user: str, limit: int = 50) -> list[Thought]:
        """Get recent thoughts for a user, oldest first (chronological order)."""
        with self._session() as session:
            thoughts = list(
                session.exec(
                    select(Thought)
                    .where(Thought.user == user)
                    .order_by(Thought.created_at.desc())
                    .limit(limit)
                ).all()
            )
            thoughts.reverse()
            return thoughts

    def get_newest(self, user: str, limit: int = 50) -> list[Thought]:
        """Get recent thoughts for a user, newest first."""
        with self._session() as session:
            return list(
                session.exec(
                    select(Thought)
                    .where(Thought.user == user)
                    .order_by(Thought.created_at.desc())
                    .limit(limit)
                ).all()
            )

    def get_recent_by_preference(
        self, user: str, preference_id: int | None, limit: int | None = None
    ) -> list[Thought]:
        """Get thoughts for a user scoped by preference_id, oldest first.

        Works for both seeded (preference_id=<int>) and free (preference_id=None) thoughts.
        """
        with self._session() as session:
            query = (
                select(Thought)
                .where(Thought.user == user, Thought.preference_id == preference_id)  # noqa: E711
                .order_by(Thought.created_at.desc())
            )
            if limit is not None:
                query = query.limit(limit)
            thoughts = list(session.exec(query).all())
            thoughts.reverse()
            return thoughts

    def get_next_unnotified(self, user: str) -> Thought | None:
        """Get the oldest un-notified thought."""
        with self._session() as session:
            return session.exec(
                select(Thought)
                .where(
                    Thought.user == user,
                    Thought.notified_at == None,  # noqa: E711
                )
                .order_by(Thought.created_at.asc())
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
                    .order_by(Thought.created_at.asc())
                ).all()
            )

    def count_unnotified(self, user: str) -> int:
        """Count un-notified thoughts for a user."""
        with self._session() as session:
            result = session.exec(
                select(func.count())
                .select_from(Thought)
                .where(
                    Thought.user == user,
                    Thought.notified_at == None,  # noqa: E711
                )
            ).one()
            return result

    def count_unnotified_free(self, user: str) -> int:
        """Count un-notified free thoughts (preference_id IS NULL)."""
        with self._session() as session:
            return session.exec(
                select(func.count())
                .select_from(Thought)
                .where(
                    Thought.user == user,
                    Thought.notified_at == None,  # noqa: E711
                    Thought.preference_id == None,  # noqa: E711
                )
            ).one()

    def get_recent_notified(self, user: str, limit: int = 1) -> list[Thought]:
        """Get most recently notified thoughts, newest first."""
        with self._session() as session:
            return list(
                session.exec(
                    select(Thought)
                    .where(
                        Thought.user == user,
                        Thought.notified_at != None,  # noqa: E711
                    )
                    .order_by(Thought.notified_at.desc())
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

    def update_embedding(self, thought_id: int, embedding: bytes) -> None:
        """Update the embedding for a thought."""
        try:
            with self._session() as session:
                thought = session.get(Thought, thought_id)
                if thought:
                    thought.embedding = embedding
                    session.add(thought)
                    session.commit()
        except Exception as e:
            logger.error("Failed to update thought %d embedding: %s", thought_id, e)

    def get_without_embeddings(self, limit: int = 50) -> list[Thought]:
        """Get thoughts that don't have embeddings yet, newest first."""
        with self._session() as session:
            return list(
                session.exec(
                    select(Thought)
                    .where(Thought.embedding == None)  # noqa: E711
                    .order_by(Thought.created_at.desc())
                    .limit(limit)
                ).all()
            )

    def get_without_images(self, limit: int = 50) -> list[Thought]:
        """Get thoughts with titles but no image_url (NULL only, not empty string)."""
        with self._session() as session:
            return list(
                session.exec(
                    select(Thought)
                    .where(
                        Thought.title != None,  # noqa: E711
                        Thought.title != "",
                        Thought.image_url == None,  # noqa: E711
                    )
                    .order_by(Thought.created_at.desc())
                    .limit(limit)
                ).all()
            )

    def update_image_url(self, thought_id: int, image_url: str) -> None:
        """Set the image_url for a thought."""
        try:
            with self._session() as session:
                thought = session.get(Thought, thought_id)
                if thought:
                    thought.image_url = image_url
                    session.add(thought)
                    session.commit()
        except Exception as e:
            logger.error("Failed to update thought %d image_url: %s", thought_id, e)

    def count(self, user: str) -> int:
        """Count total thoughts for a user."""
        with self._session() as session:
            from sqlalchemy import func

            return session.exec(
                select(func.count()).select_from(Thought).where(Thought.user == user)
            ).one()

"""Preference store — user sentiment and reaction preferences."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlmodel import Session, or_, select

from penny.database.models import Preference

logger = logging.getLogger(__name__)


class PreferenceStore:
    """Manages Preference records: add, query, and period tracking."""

    def __init__(self, engine):
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    def add(
        self,
        user: str,
        content: str,
        valence: str,
        embedding: bytes | None = None,
        source: str = "extracted",
        mention_count: int = 1,
    ) -> Preference | None:
        """Insert a preference. Returns the created record or None."""
        try:
            with self._session() as session:
                pref = Preference(
                    user=user,
                    content=content,
                    valence=valence,
                    embedding=embedding,
                    created_at=datetime.now(UTC),
                    source=source,
                    mention_count=mention_count,
                )
                session.add(pref)
                session.commit()
                session.refresh(pref)
                logger.debug("Preference added for %s: %s (%s)", user, content[:50], valence)
                return pref
        except Exception as e:
            logger.error("Failed to add preference: %s", e)
            return None

    def increment_mention_count(self, pref_id: int) -> None:
        """Increment the mention_count for a preference."""
        try:
            with self._session() as session:
                pref = session.get(Preference, pref_id)
                if pref:
                    current = pref.mention_count if pref.mention_count is not None else 0
                    pref.mention_count = current + 1
                    session.add(pref)
                    session.commit()
        except Exception as e:
            logger.error("Failed to increment mention count for preference %d: %s", pref_id, e)

    def get_by_id(self, pref_id: int) -> Preference | None:
        """Get a single preference by ID."""
        with self._session() as session:
            return session.get(Preference, pref_id)

    def get_by_ids(self, pref_ids: set[int]) -> list[Preference]:
        """Get multiple preferences by ID in a single query."""
        if not pref_ids:
            return []
        with self._session() as session:
            return list(session.exec(select(Preference).where(Preference.id.in_(pref_ids))).all())

    def get_for_user(self, user: str, limit: int = 100) -> list[Preference]:
        """Get all preferences for a user, newest first."""
        with self._session() as session:
            return list(
                session.exec(
                    select(Preference)
                    .where(Preference.user == user)
                    .order_by(Preference.created_at.desc())
                    .limit(limit)
                ).all()
            )

    def get_positive(self, user: str) -> list[Preference]:
        """Get positive preferences for a user."""
        with self._session() as session:
            return list(
                session.exec(
                    select(Preference).where(
                        Preference.user == user,
                        Preference.valence == "positive",
                    )
                ).all()
            )

    def get_least_recent_positive(
        self, user: str, pool_size: int = 5, mention_threshold: int = 1
    ) -> list[Preference]:
        """Get the N least-recently-thought-about positive preferences.

        NULLs (never thought about) come first, then oldest last_thought_at.
        Manual preferences always pass. Extracted preferences must meet mention_threshold.
        """
        with self._session() as session:
            return list(
                session.exec(
                    select(Preference)
                    .where(
                        Preference.user == user,
                        Preference.valence == "positive",
                        or_(
                            Preference.source == "manual",
                            Preference.mention_count >= mention_threshold,
                        ),
                    )
                    .order_by(
                        Preference.last_thought_at.is_(None).desc(),  # type: ignore[union-attr]
                        Preference.last_thought_at.asc(),  # type: ignore[union-attr]
                    )
                    .limit(pool_size)
                ).all()
            )

    def mark_thought_about(self, pref_id: int) -> None:
        """Update last_thought_at to now for a preference."""
        try:
            with self._session() as session:
                pref = session.get(Preference, pref_id)
                if pref:
                    pref.last_thought_at = datetime.now(UTC)
                    session.add(pref)
                    session.commit()
        except Exception as e:
            logger.error("Failed to mark preference %d as thought about: %s", pref_id, e)

    def get_with_embeddings(self, user: str) -> list[Preference]:
        """Get preferences with embeddings for similarity search."""
        with self._session() as session:
            return list(
                session.exec(
                    select(Preference).where(
                        Preference.user == user,
                        Preference.embedding != None,  # noqa: E711
                    )
                ).all()
            )

    def get_without_embeddings(self, limit: int) -> list[Preference]:
        """Get preferences that don't have embeddings yet, newest first."""
        with self._session() as session:
            return list(
                session.exec(
                    select(Preference)
                    .where(Preference.embedding == None)  # noqa: E711
                    .order_by(Preference.created_at.desc())
                    .limit(limit)
                ).all()
            )

    def update_embedding(self, pref_id: int, embedding: bytes) -> None:
        """Update the embedding for a preference."""
        try:
            with self._session() as session:
                pref = session.get(Preference, pref_id)
                if pref:
                    pref.embedding = embedding
                    session.add(pref)
                    session.commit()
        except Exception as e:
            logger.error("Failed to update preference embedding: %s", e)

    def delete(self, pref_id: int) -> bool:
        """Delete a preference by ID. Returns True if deleted."""
        try:
            with self._session() as session:
                pref = session.get(Preference, pref_id)
                if pref:
                    session.delete(pref)
                    session.commit()
                    return True
                return False
        except Exception as e:
            logger.error("Failed to delete preference %d: %s", pref_id, e)
            return False

    def get_for_user_by_valence(self, user: str, valence: str) -> list[Preference]:
        """Get preferences for a user filtered by valence, most mentioned first."""
        with self._session() as session:
            return list(
                session.exec(
                    select(Preference)
                    .where(Preference.user == user, Preference.valence == valence)
                    .order_by(
                        Preference.mention_count.desc(),
                        Preference.created_at.desc(),
                    )
                ).all()
            )

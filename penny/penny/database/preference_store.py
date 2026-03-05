"""Preference store — user sentiment and reaction preferences."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlmodel import Session, select

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
        source_period_start: datetime,
        source_period_end: datetime,
        embedding: bytes | None = None,
    ) -> Preference | None:
        """Insert a preference. Returns the created record or None."""
        try:
            with self._session() as session:
                pref = Preference(
                    user=user,
                    content=content,
                    valence=valence,
                    embedding=embedding,
                    source_period_start=source_period_start,
                    source_period_end=source_period_end,
                    created_at=datetime.now(UTC),
                )
                session.add(pref)
                session.commit()
                session.refresh(pref)
                logger.debug("Preference added for %s: %s (%s)", user, content[:50], valence)
                return pref
        except Exception as e:
            logger.error("Failed to add preference: %s", e)
            return None

    def get_for_user(self, user: str, limit: int = 100) -> list[Preference]:
        """Get all preferences for a user, newest first."""
        with self._session() as session:
            return list(
                session.exec(
                    select(Preference)
                    .where(Preference.user == user)
                    .order_by(Preference.created_at.desc())  # type: ignore[unresolved-attribute]
                    .limit(limit)
                ).all()
            )

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
                    .order_by(Preference.created_at.desc())  # type: ignore[unresolved-attribute]
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

    def exists_for_period(self, user: str, period_start: datetime) -> bool:
        """Check if preferences have already been extracted for a period."""
        with self._session() as session:
            result = session.exec(
                select(Preference)
                .where(
                    Preference.user == user,
                    Preference.source_period_start == period_start,
                )
                .limit(1)
            ).first()
            return result is not None

"""Engagement store — recording and querying user engagement signals."""

import logging

from sqlmodel import Session, select

from penny.database.models import Engagement

logger = logging.getLogger(__name__)


class EngagementStore:
    """Manages Engagement records: creation and queries by user/entity."""

    def __init__(self, engine):
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    def add(
        self,
        user: str,
        engagement_type: str,
        valence: str,
        strength: float,
        entity_id: int | None = None,
        source_message_id: int | None = None,
    ) -> Engagement | None:
        """Record a user engagement event. Returns the created Engagement or None."""
        try:
            with self._session() as session:
                engagement = Engagement(
                    user=user,
                    entity_id=entity_id,
                    engagement_type=engagement_type,
                    valence=valence,
                    strength=strength,
                    source_message_id=source_message_id,
                )
                session.add(engagement)
                session.commit()
                session.refresh(engagement)
                logger.debug(
                    "Added %s engagement (valence=%s, strength=%.2f) for user %s",
                    engagement_type,
                    valence,
                    strength,
                    user,
                )
                return engagement
        except Exception as e:
            logger.error("Failed to add engagement: %s", e)
            return None

    def get_for_entity(self, user: str, entity_id: int) -> list[Engagement]:
        """Get all engagements for a specific entity, newest first."""
        with self._session() as session:
            return list(
                session.exec(
                    select(Engagement)
                    .where(Engagement.user == user, Engagement.entity_id == entity_id)
                    .order_by(Engagement.created_at.desc())  # type: ignore[unresolved-attribute]
                ).all()
            )

    def get_for_user(self, user: str) -> list[Engagement]:
        """Get all engagements for a user, newest first."""
        with self._session() as session:
            return list(
                session.exec(
                    select(Engagement)
                    .where(Engagement.user == user)
                    .order_by(Engagement.created_at.desc())  # type: ignore[unresolved-attribute]
                ).all()
            )

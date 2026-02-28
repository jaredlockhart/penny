"""Fact store — CRUD and queries for entity facts."""

import logging
from datetime import UTC, datetime

from sqlmodel import Session, select

from penny.database.models import Entity, Fact

logger = logging.getLogger(__name__)


class FactStore:
    """Manages Fact records: creation, lookup, embedding updates, and notification tracking."""

    def __init__(self, engine):
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    def add(
        self,
        entity_id: int,
        content: str,
        source_url: str | None = None,
        source_search_log_id: int | None = None,
        source_message_id: int | None = None,
        embedding: bytes | None = None,
        notified_at: datetime | None = None,
    ) -> Fact | None:
        """Add a fact to an entity. Returns the created Fact, or None on failure."""
        try:
            with self._session() as session:
                fact = Fact(
                    entity_id=entity_id,
                    content=content,
                    source_url=source_url,
                    source_search_log_id=source_search_log_id,
                    source_message_id=source_message_id,
                    embedding=embedding,
                    notified_at=notified_at,
                )
                session.add(fact)
                self._touch_entity(session, entity_id)
                session.commit()
                session.refresh(fact)
                logger.debug("Added fact to entity %d: %s", entity_id, content[:50])
                return fact
        except Exception as e:
            logger.error("Failed to add fact to entity %d: %s", entity_id, e)
            return None

    def _touch_entity(self, session: Session, entity_id: int) -> None:
        """Update the parent entity's updated_at timestamp."""
        entity = session.get(Entity, entity_id)
        if entity:
            entity.updated_at = datetime.now(UTC)
            session.add(entity)

    def get_for_entity(self, entity_id: int) -> list[Fact]:
        """Get all facts for an entity, ordered by learned_at ascending."""
        with self._session() as session:
            return list(
                session.exec(
                    select(Fact).where(Fact.entity_id == entity_id).order_by(Fact.learned_at.asc())  # type: ignore[unresolved-attribute]
                ).all()
            )

    def get_by_search_log_ids(self, search_log_ids: list[int]) -> list[Fact]:
        """Get all Facts linked to any of the given SearchLog IDs."""
        if not search_log_ids:
            return []
        with self._session() as session:
            return list(
                session.exec(
                    select(Fact).where(
                        Fact.source_search_log_id.in_(search_log_ids)  # type: ignore[union-attr]
                    )
                ).all()
            )

    def get_without_embeddings(self, limit: int) -> list[Fact]:
        """Get facts that don't have embeddings yet."""
        with self._session() as session:
            return list(
                session.exec(
                    select(Fact)
                    .where(Fact.embedding == None)  # noqa: E711
                    .order_by(Fact.learned_at.desc())  # type: ignore[unresolved-attribute]
                    .limit(limit)
                ).all()
            )

    def get_unnotified(self, user: str) -> list[Fact]:
        """Get facts that haven't been communicated to a user yet."""
        with self._session() as session:
            return list(
                session.exec(
                    select(Fact)
                    .join(Entity, Fact.entity_id == Entity.id)  # type: ignore[invalid-argument-type]
                    .where(Entity.user == user, Fact.notified_at == None)  # noqa: E711
                    .order_by(Fact.learned_at.desc())  # type: ignore[unresolved-attribute]
                ).all()
            )

    def update_embedding(self, fact_id: int, embedding: bytes) -> None:
        """Update the embedding for a fact."""
        try:
            with self._session() as session:
                fact = session.get(Fact, fact_id)
                if fact:
                    fact.embedding = embedding
                    session.add(fact)
                    session.commit()
        except Exception as e:
            logger.error("Failed to update fact %d embedding: %s", fact_id, e)

    def count_notified(self, entity_id: int) -> int:
        """Count facts that have already been notified for an entity."""
        with self._session() as session:
            stmt = (
                select(Fact).where(Fact.entity_id == entity_id, Fact.notified_at != None)  # noqa: E711
            )
            return len(list(session.exec(stmt).all()))

    def mark_notified(self, fact_ids: list[int]) -> None:
        """Mark facts as notified (communicated to user)."""
        if not fact_ids:
            return
        try:
            now = datetime.now(UTC)
            with self._session() as session:
                for fact_id in fact_ids:
                    fact = session.get(Fact, fact_id)
                    if fact:
                        fact.notified_at = now
                        session.add(fact)
                session.commit()
                logger.debug("Marked %d facts as notified", len(fact_ids))
        except Exception as e:
            logger.error("Failed to mark facts as notified: %s", e)

"""Entity store — CRUD and metadata for knowledge entities."""

import logging
from datetime import UTC, datetime

from sqlmodel import Session, select

from penny.database.models import Engagement, Entity, Fact

logger = logging.getLogger(__name__)


class EntityStore:
    """Manages Entity records: creation, lookup, deletion, and metadata updates."""

    def __init__(self, engine):
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    def get(self, entity_id: int) -> Entity | None:
        """Get an entity by ID."""
        with self._session() as session:
            return session.get(Entity, entity_id)

    def get_or_create(self, user: str, name: str) -> Entity | None:
        """Get an existing entity or create a new one. Returns None on failure."""
        name = name.lower().strip()
        try:
            with self._session() as session:
                existing = session.exec(
                    select(Entity).where(Entity.user == user, Entity.name == name)
                ).first()
                if existing:
                    existing.updated_at = datetime.now(UTC)
                    session.add(existing)
                    session.commit()
                    session.refresh(existing)
                    return existing
                return self._create(session, user, name)
        except Exception as e:
            logger.error("Failed to get_or_create entity: %s", e)
            return None

    def _create(self, session: Session, user: str, name: str) -> Entity:
        """Insert a new entity into the database."""
        entity = Entity(
            user=user,
            name=name,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        session.add(entity)
        session.commit()
        session.refresh(entity)
        logger.debug("Created entity '%s' for user %s", name, user)
        return entity

    def get_for_user(self, user: str) -> list[Entity]:
        """Get all entities for a user, ordered by updated_at descending."""
        with self._session() as session:
            return list(
                session.exec(
                    select(Entity).where(Entity.user == user).order_by(Entity.updated_at.desc())  # type: ignore[unresolved-attribute]
                ).all()
            )

    def get_with_embeddings(self, user: str) -> list[Entity]:
        """Get all entities for a user that have embeddings."""
        with self._session() as session:
            return list(
                session.exec(
                    select(Entity).where(
                        Entity.user == user,
                        Entity.embedding != None,  # noqa: E711
                    )
                ).all()
            )

    def get_without_embeddings(self, limit: int) -> list[Entity]:
        """Get entities that don't have embeddings yet."""
        with self._session() as session:
            return list(
                session.exec(
                    select(Entity)
                    .where(Entity.embedding == None)  # noqa: E711
                    .order_by(Entity.created_at.desc())  # type: ignore[unresolved-attribute]
                    .limit(limit)
                ).all()
            )

    def delete(self, entity_id: int) -> bool:
        """Delete an entity and its associated facts and engagements."""
        try:
            with self._session() as session:
                entity = session.get(Entity, entity_id)
                if not entity:
                    return False
                self._delete_related(session, entity_id)
                session.delete(entity)
                session.commit()
                return True
        except Exception as e:
            logger.error("Failed to delete entity: %s", e)
            return False

    def _delete_related(self, session: Session, entity_id: int) -> None:
        """Delete engagements and facts associated with an entity."""
        for eng in session.exec(select(Engagement).where(Engagement.entity_id == entity_id)).all():
            session.delete(eng)
        for fact in session.exec(select(Fact).where(Fact.entity_id == entity_id)).all():
            session.delete(fact)

    def update_embedding(self, entity_id: int, embedding: bytes) -> None:
        """Update the embedding for an entity."""
        try:
            with self._session() as session:
                entity = session.get(Entity, entity_id)
                if entity:
                    entity.embedding = embedding
                    session.add(entity)
                    session.commit()
        except Exception as e:
            logger.error("Failed to update entity %d embedding: %s", entity_id, e)

    def update_tagline(self, entity_id: int, tagline: str) -> None:
        """Update the tagline for an entity."""
        try:
            with self._session() as session:
                entity = session.get(Entity, entity_id)
                if entity:
                    entity.tagline = tagline
                    session.add(entity)
                    session.commit()
        except Exception as e:
            logger.error("Failed to update entity %d tagline: %s", entity_id, e)

    def update_last_enriched_at(self, entity_id: int) -> None:
        """Record that an entity was just enriched."""
        try:
            with self._session() as session:
                entity = session.get(Entity, entity_id)
                if entity:
                    entity.last_enriched_at = datetime.now(UTC)
                    session.add(entity)
                    session.commit()
        except Exception as e:
            logger.error("Failed to update entity %d last_enriched_at: %s", entity_id, e)

    def update_last_notified_at(self, entity_id: int) -> None:
        """Record that an entity was just included in a notification."""
        try:
            with self._session() as session:
                entity = session.get(Entity, entity_id)
                if entity:
                    entity.last_notified_at = datetime.now(UTC)
                    session.add(entity)
                    session.commit()
        except Exception as e:
            logger.error("Failed to update entity %d last_notified_at: %s", entity_id, e)

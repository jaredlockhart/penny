"""Learn prompt store — CRUD for /learn topic tracking."""

import logging
from datetime import UTC, datetime

from sqlmodel import Session, select

from penny.constants import PennyConstants
from penny.database.models import Engagement, Entity, Fact, LearnPrompt, SearchLog

logger = logging.getLogger(__name__)


class LearnPromptStore:
    """Manages LearnPrompt records: creation, status tracking, and cascading deletion."""

    def __init__(self, engine):
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    def create(
        self,
        user: str,
        prompt_text: str,
        searches_remaining: int = 0,
    ) -> LearnPrompt | None:
        """Create a new LearnPrompt. Returns None on failure."""
        try:
            with self._session() as session:
                learn_prompt = LearnPrompt(
                    user=user,
                    prompt_text=prompt_text,
                    searches_remaining=searches_remaining,
                )
                session.add(learn_prompt)
                session.commit()
                session.refresh(learn_prompt)
                logger.debug("Created LearnPrompt %d for user %s", learn_prompt.id, user)
                return learn_prompt
        except Exception as e:
            logger.error("Failed to create LearnPrompt: %s", e)
            return None

    def get(self, learn_prompt_id: int) -> LearnPrompt | None:
        """Get a LearnPrompt by ID."""
        with self._session() as session:
            return session.get(LearnPrompt, learn_prompt_id)

    def get_next_active(self) -> LearnPrompt | None:
        """Get the oldest active LearnPrompt across all users."""
        with self._session() as session:
            return session.exec(
                select(LearnPrompt)
                .where(LearnPrompt.status == PennyConstants.LearnPromptStatus.ACTIVE)
                .order_by(LearnPrompt.created_at.asc())  # type: ignore[union-attr]
                .limit(1)
            ).first()

    def get_active(self, user: str) -> list[LearnPrompt]:
        """Get all active LearnPrompts for a user."""
        with self._session() as session:
            return list(
                session.exec(
                    select(LearnPrompt)
                    .where(LearnPrompt.user == user, LearnPrompt.status == "active")
                    .order_by(LearnPrompt.created_at.asc())  # type: ignore[union-attr]
                ).all()
            )

    def get_for_user(self, user: str) -> list[LearnPrompt]:
        """Get all LearnPrompts for a user."""
        with self._session() as session:
            return list(
                session.exec(
                    select(LearnPrompt)
                    .where(LearnPrompt.user == user)
                    .order_by(LearnPrompt.created_at.asc())  # type: ignore[union-attr]
                ).all()
            )

    def update_status(self, learn_prompt_id: int, status: str) -> None:
        """Update the status of a LearnPrompt."""
        try:
            with self._session() as session:
                lp = session.get(LearnPrompt, learn_prompt_id)
                if lp:
                    lp.status = status
                    lp.updated_at = datetime.now(UTC)
                    session.add(lp)
                    session.commit()
        except Exception as e:
            logger.error("Failed to update LearnPrompt %d status: %s", learn_prompt_id, e)

    def update_searches_remaining(self, learn_prompt_id: int, searches_remaining: int) -> None:
        """Set the searches_remaining count on a LearnPrompt."""
        try:
            with self._session() as session:
                lp = session.get(LearnPrompt, learn_prompt_id)
                if lp:
                    lp.searches_remaining = searches_remaining
                    lp.updated_at = datetime.now(UTC)
                    session.add(lp)
                    session.commit()
        except Exception as e:
            logger.error(
                "Failed to update LearnPrompt %d searches_remaining: %s", learn_prompt_id, e
            )

    def decrement_searches(self, learn_prompt_id: int) -> None:
        """Decrement searches_remaining by 1."""
        try:
            with self._session() as session:
                lp = session.get(LearnPrompt, learn_prompt_id)
                if lp and lp.searches_remaining > 0:
                    lp.searches_remaining -= 1
                    lp.updated_at = datetime.now(UTC)
                    session.add(lp)
                    session.commit()
        except Exception as e:
            logger.error("Failed to decrement LearnPrompt %d searches: %s", learn_prompt_id, e)

    def get_unannounced_completed(self, user: str) -> list[LearnPrompt]:
        """Get completed LearnPrompts that haven't been announced yet."""
        with self._session() as session:
            return list(
                session.exec(
                    select(LearnPrompt)
                    .where(
                        LearnPrompt.user == user,
                        LearnPrompt.status == "completed",
                        LearnPrompt.announced_at.is_(None),  # type: ignore[union-attr]
                    )
                    .order_by(LearnPrompt.created_at.asc())  # type: ignore[union-attr]
                ).all()
            )

    def mark_announced(self, learn_prompt_id: int) -> None:
        """Mark a LearnPrompt as announced."""
        try:
            with self._session() as session:
                lp = session.get(LearnPrompt, learn_prompt_id)
                if lp:
                    lp.announced_at = datetime.now(UTC)
                    session.add(lp)
                    session.commit()
        except Exception as e:
            logger.error("Failed to mark LearnPrompt %d as announced: %s", learn_prompt_id, e)

    def delete(self, learn_prompt_id: int) -> list[tuple[str, int]]:
        """Delete a LearnPrompt and all entities/facts from its searches.

        Returns list of (entity_name, fact_count) for each deleted entity.
        """
        try:
            with self._session() as session:
                lp = session.get(LearnPrompt, learn_prompt_id)
                if not lp:
                    return []
                search_log_ids = self._find_search_log_ids(session, learn_prompt_id)
                entity_ids = self._find_affected_entity_ids(session, search_log_ids)
                deleted = self._delete_entities(session, entity_ids)
                self._delete_search_logs(session, learn_prompt_id)
                session.delete(lp)
                session.commit()
                logger.debug("Deleted LearnPrompt %d (%d entities)", learn_prompt_id, len(deleted))
                return deleted
        except Exception as e:
            logger.error("Failed to delete LearnPrompt %d: %s", learn_prompt_id, e)
            return []

    def _find_search_log_ids(self, session: Session, learn_prompt_id: int) -> list[int]:
        """Find all search log IDs for a learn prompt."""
        search_logs = list(
            session.exec(
                select(SearchLog).where(SearchLog.learn_prompt_id == learn_prompt_id)
            ).all()
        )
        return [sl.id for sl in search_logs if sl.id is not None]

    def _find_affected_entity_ids(self, session: Session, search_log_ids: list[int]) -> set[int]:
        """Find entity IDs that have facts from the given search logs."""
        if not search_log_ids:
            return set()
        facts = list(
            session.exec(
                select(Fact).where(
                    Fact.source_search_log_id.in_(search_log_ids)  # type: ignore[union-attr]
                )
            ).all()
        )
        return {f.entity_id for f in facts}

    def _delete_entities(self, session: Session, entity_ids: set[int]) -> list[tuple[str, int]]:
        """Delete entities and their facts/engagements. Returns (name, fact_count) pairs."""
        deleted: list[tuple[str, int]] = []
        for entity_id in entity_ids:
            entity = session.get(Entity, entity_id)
            if not entity:
                continue
            for eng in session.exec(
                select(Engagement).where(Engagement.entity_id == entity_id)
            ).all():
                session.delete(eng)
            facts = list(session.exec(select(Fact).where(Fact.entity_id == entity_id)).all())
            for fact in facts:
                session.delete(fact)
            deleted.append((entity.name, len(facts)))
            session.delete(entity)
        return deleted

    def _delete_search_logs(self, session: Session, learn_prompt_id: int) -> None:
        """Delete all search logs for a learn prompt."""
        for sl in session.exec(
            select(SearchLog).where(SearchLog.learn_prompt_id == learn_prompt_id)
        ).all():
            session.delete(sl)

"""Follow prompt store — lifecycle management for event monitoring subscriptions."""

import logging
from datetime import UTC, datetime

from sqlmodel import Session, select

from penny.database.models import FollowPrompt

logger = logging.getLogger(__name__)


class FollowPromptStore:
    """Manages FollowPrompt records: creation, polling, cancellation."""

    def __init__(self, engine):
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    def create(
        self,
        user: str,
        prompt_text: str,
        query_terms: str,
        cron_expression: str = "0 9 * * *",
        timing_description: str = "daily",
        user_timezone: str = "UTC",
    ) -> FollowPrompt | None:
        """Create a new follow prompt. Returns the created FollowPrompt, or None on failure."""
        try:
            with self._session() as session:
                prompt = FollowPrompt(
                    user=user,
                    prompt_text=prompt_text,
                    query_terms=query_terms,
                    cron_expression=cron_expression,
                    timing_description=timing_description,
                    user_timezone=user_timezone,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
                session.add(prompt)
                session.commit()
                session.refresh(prompt)
                logger.debug("Created follow prompt for user %s: %s", user, prompt_text[:50])
                return prompt
        except Exception as e:
            logger.error("Failed to create follow prompt: %s", e)
            return None

    def get(self, follow_prompt_id: int) -> FollowPrompt | None:
        """Get a follow prompt by ID."""
        with self._session() as session:
            return session.get(FollowPrompt, follow_prompt_id)

    def get_active(self, user: str) -> list[FollowPrompt]:
        """Get all active follow prompts for a user."""
        with self._session() as session:
            return list(
                session.exec(
                    select(FollowPrompt)
                    .where(FollowPrompt.user == user, FollowPrompt.status == "active")
                    .order_by(FollowPrompt.created_at.desc())  # type: ignore[unresolved-attribute]
                ).all()
            )

    def get_active_by_poll_priority(self) -> list[FollowPrompt]:
        """Get all active prompts ordered by poll priority (never-polled first, then oldest)."""
        with self._session() as session:
            never_polled = list(
                session.exec(
                    select(FollowPrompt).where(
                        FollowPrompt.status == "active",
                        FollowPrompt.last_polled_at == None,  # noqa: E711
                    )
                ).all()
            )
            polled = list(
                session.exec(
                    select(FollowPrompt)
                    .where(
                        FollowPrompt.status == "active",
                        FollowPrompt.last_polled_at != None,  # noqa: E711
                    )
                    .order_by(FollowPrompt.last_polled_at.asc())  # type: ignore[unresolved-attribute]
                ).all()
            )
            return never_polled + polled

    def update_last_polled(self, follow_prompt_id: int) -> None:
        """Record that a follow prompt was just polled."""
        try:
            with self._session() as session:
                prompt = session.get(FollowPrompt, follow_prompt_id)
                if prompt:
                    prompt.last_polled_at = datetime.now(UTC)
                    prompt.updated_at = datetime.now(UTC)
                    session.add(prompt)
                    session.commit()
        except Exception as e:
            logger.error("Failed to update follow prompt %d last_polled: %s", follow_prompt_id, e)

    def update_last_notified(self, follow_prompt_id: int) -> None:
        """Record that a follow prompt just triggered a notification."""
        try:
            with self._session() as session:
                prompt = session.get(FollowPrompt, follow_prompt_id)
                if prompt:
                    prompt.last_notified_at = datetime.now(UTC)
                    prompt.updated_at = datetime.now(UTC)
                    session.add(prompt)
                    session.commit()
        except Exception as e:
            logger.error("Failed to update follow prompt %d last_notified: %s", follow_prompt_id, e)

    def cancel(self, follow_prompt_id: int) -> bool:
        """Delete a follow prompt. Returns True if deleted, False if not found."""
        try:
            with self._session() as session:
                prompt = session.get(FollowPrompt, follow_prompt_id)
                if not prompt:
                    return False
                session.delete(prompt)
                session.commit()
                return True
        except Exception as e:
            logger.error("Failed to delete follow prompt %d: %s", follow_prompt_id, e)
            return False

    def get_for_user(self, user: str) -> list[FollowPrompt]:
        """Get all follow prompts for a user (including cancelled)."""
        with self._session() as session:
            return list(
                session.exec(
                    select(FollowPrompt)
                    .where(FollowPrompt.user == user)
                    .order_by(FollowPrompt.created_at.desc())  # type: ignore[unresolved-attribute]
                ).all()
            )

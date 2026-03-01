"""App state store — internal key-value state persisted across restarts."""

import logging
from datetime import UTC, datetime

from sqlmodel import Session

from penny.database.models import AppState

logger = logging.getLogger(__name__)


class AppStateStore:
    """Manages internal app state persisted across container restarts.

    Stores arbitrary string values by key. Use get_datetime/set_datetime for
    timestamp values.
    """

    def __init__(self, engine):
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    def get(self, key: str) -> str | None:
        """Get a string value by key, or None if not set."""
        with self._session() as session:
            row = session.get(AppState, key)
            return row.value if row else None

    def set(self, key: str, value: str) -> None:
        """Upsert a string value by key."""
        try:
            with self._session() as session:
                row = session.get(AppState, key)
                if row:
                    row.value = value
                    row.updated_at = datetime.now(UTC)
                    session.add(row)
                else:
                    session.add(AppState(key=key, value=value))
                session.commit()
        except Exception as e:
            logger.error("Failed to set app state %s: %s", key, e)

    def get_datetime(self, key: str) -> datetime | None:
        """Get a UTC datetime by key (stored as ISO 8601 string), or None if not set."""
        raw = self.get(key)
        if raw is None:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError, TypeError:
            logger.warning("Invalid datetime in app state for key %s: %r", key, raw)
            return None

    def set_datetime(self, key: str, value: datetime) -> None:
        """Persist a UTC datetime by key as ISO 8601 string."""
        self.set(key, value.isoformat())

"""Media store — binary blobs referenced by memory entries via <media:ID> tokens."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlmodel import Session

from penny.database.models import Media

logger = logging.getLogger(__name__)


class MediaStore:
    """Put/get for binary media (images, etc.) stored out-of-line from entries."""

    def __init__(self, engine):
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    def put(self, data: bytes, mime_type: str, source_url: str | None = None) -> int:
        """Insert a media blob and return its assigned id."""
        with self._session() as session:
            row = Media(
                mime_type=mime_type,
                data=data,
                source_url=source_url,
                created_at=datetime.now(UTC),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            if row.id is None:
                raise RuntimeError("media row was inserted but has no id")
            logger.debug("Stored %d bytes as media %d (%s)", len(data), row.id, mime_type)
            return row.id

    def get(self, media_id: int) -> Media | None:
        with self._session() as session:
            return session.get(Media, media_id)

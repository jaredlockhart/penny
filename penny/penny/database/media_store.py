"""Media store — images captured while browsing, delivered side-channel at egress."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from similarity.embeddings import find_similar
from sqlmodel import Session, select

from penny.database.models import Media
from penny.llm.embeddings import deserialize_embedding

logger = logging.getLogger(__name__)


class MediaStore:
    """Store browsed images and retrieve the nearest match to an egress message."""

    def __init__(self, engine):
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    def put(
        self,
        data: bytes,
        mime_type: str,
        source_url: str | None = None,
        title: str | None = None,
        embedding: bytes | None = None,
    ) -> int:
        """Insert an image blob with its metadata and return its assigned id."""
        with self._session() as session:
            row = Media(
                mime_type=mime_type,
                data=data,
                source_url=source_url,
                title=title,
                embedding=embedding,
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

    def find_nearest(self, query_embedding: list[float]) -> Media | None:
        """Return the embedded image whose metadata is closest to ``query``.

        The single nearest image always wins. Returns None only when no
        embedded media exists at all.
        """
        with self._session() as session:
            rows = session.exec(
                select(Media).where(Media.embedding.is_not(None))  # type: ignore[union-attr]
            ).all()
        candidates = [(row, deserialize_embedding(row.embedding)) for row in rows if row.embedding]
        if not candidates:
            return None
        scored = find_similar(
            query_embedding,
            [(media.id, vector) for media, vector in candidates if media.id is not None],
            top_k=1,
            threshold=-1.0,
        )
        if not scored:
            return None
        best_id, score = scored[0]
        logger.debug("Matched media %d at cosine %.3f", best_id, score)
        return next(media for media, _ in candidates if media.id == best_id)

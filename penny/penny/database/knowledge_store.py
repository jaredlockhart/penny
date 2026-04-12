"""Knowledge store — summarized web page content for factual recall."""

import logging
from datetime import UTC, datetime

from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, select

from penny.database.models import Knowledge, PromptLog

logger = logging.getLogger(__name__)


class KnowledgeStore:
    """Manages Knowledge records: one summarized entry per browsed URL."""

    def __init__(self, engine):
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    def upsert_by_url(
        self,
        url: str,
        title: str,
        summary: str,
        embedding: bytes | None,
        source_prompt_id: int,
    ) -> Knowledge | None:
        """Insert or update a knowledge entry by URL. Returns the record or None."""
        try:
            with self._session() as session:
                existing = session.exec(select(Knowledge).where(Knowledge.url == url)).first()
                if existing:
                    return self._update_existing(
                        session, existing, title, summary, embedding, source_prompt_id
                    )
                return self._insert_new(session, url, title, summary, embedding, source_prompt_id)
        except SQLAlchemyError as error:
            logger.error("Failed to upsert knowledge for %s: %s", url, error)
            return None

    def _update_existing(
        self,
        session: Session,
        existing: Knowledge,
        title: str,
        summary: str,
        embedding: bytes | None,
        source_prompt_id: int,
    ) -> Knowledge:
        """Update an existing knowledge entry with new summary and embedding."""
        existing.title = title
        existing.summary = summary
        existing.embedding = embedding
        existing.source_prompt_id = source_prompt_id
        existing.updated_at = datetime.now(UTC)
        session.add(existing)
        session.commit()
        session.refresh(existing)
        logger.debug("Updated knowledge for %s", existing.url)
        return existing

    def _insert_new(
        self,
        session: Session,
        url: str,
        title: str,
        summary: str,
        embedding: bytes | None,
        source_prompt_id: int,
    ) -> Knowledge:
        """Insert a new knowledge entry."""
        now = datetime.now(UTC)
        entry = Knowledge(
            url=url,
            title=title,
            summary=summary,
            embedding=embedding,
            source_prompt_id=source_prompt_id,
            created_at=now,
            updated_at=now,
        )
        session.add(entry)
        session.commit()
        session.refresh(entry)
        logger.debug("Added knowledge for %s", url)
        return entry

    def get_by_url(self, url: str) -> Knowledge | None:
        """Look up a knowledge entry by URL."""
        with self._session() as session:
            return session.exec(select(Knowledge).where(Knowledge.url == url)).first()

    def get_with_embeddings(self) -> list[Knowledge]:
        """Get all knowledge entries that have embeddings, for similarity search."""
        with self._session() as session:
            return list(
                session.exec(
                    select(Knowledge).where(Knowledge.embedding != None)  # noqa: E711
                ).all()
            )

    def get_latest_prompt_timestamp(self) -> datetime | None:
        """Get the timestamp of the most recently processed prompt via FK join."""
        with self._session() as session:
            result = session.exec(
                select(PromptLog.timestamp)
                .join(Knowledge, Knowledge.source_prompt_id == PromptLog.id)  # ty: ignore[invalid-argument-type]
                .order_by(PromptLog.timestamp.desc())
                .limit(1)
            ).first()
            return result

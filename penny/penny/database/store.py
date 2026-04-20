"""Store access layer — collections and logs, unified.

Collections are keyed sets with dedup on write; logs are append-only streams.
Both share a single `store_entry` table with `key` nullable for logs. Entries
are immutable once written — `update` replaces whole content for a given key.

Dedup on collection writes uses a disjunction of three similarity checks
against existing entries in the same store (thresholds live in
``PennyConstants`` — STORE_KEY_SIM_THRESHOLD, STORE_CONTENT_SIM_THRESHOLD,
STORE_COMBINED_SIM_THRESHOLD). Any hit rejects the write; the caller gets a
per-entry outcome in the result.

Embeddings are optional: if the caller passes None, the similarity checks for
that axis simply don't fire and no dedup can happen on that axis. Stores
without any embedding still accept writes — they just can't dedup by similarity.
"""

from __future__ import annotations

import logging
import random
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel
from similarity.embeddings import (
    cosine_similarity,
    deserialize_embedding,
    serialize_embedding,
)
from sqlmodel import Session, select

from penny.constants import PennyConstants
from penny.database.models import Store, StoreEntry

logger = logging.getLogger(__name__)


class StoreType(StrEnum):
    COLLECTION = "collection"
    LOG = "log"


class RecallMode(StrEnum):
    OFF = "off"
    RECENT = "recent"
    RELEVANT = "relevant"
    ALL = "all"


class StoreTypeError(Exception):
    """Raised when an operation is called against the wrong store type."""


class StoreNotFoundError(Exception):
    """Raised when an operation targets a store that doesn't exist."""


class DedupThresholds(BaseModel):
    key_sim: float = PennyConstants.STORE_KEY_SIM_THRESHOLD
    content_sim: float = PennyConstants.STORE_CONTENT_SIM_THRESHOLD
    combined: float = PennyConstants.STORE_COMBINED_SIM_THRESHOLD


class EntryInput(BaseModel):
    """Input row for collection_write — key, content, and optional embeddings."""

    key: str
    content: str
    key_embedding: list[float] | None = None
    content_embedding: list[float] | None = None


class LogEntryInput(BaseModel):
    """Input row for log append — keyless content plus optional embedding."""

    content: str
    content_embedding: list[float] | None = None


WriteOutcome = Literal["written", "duplicate"]


class WriteResult(BaseModel):
    key: str
    outcome: WriteOutcome
    entry_id: int | None = None


MoveOutcome = Literal["ok", "not_found", "collision"]
UpdateOutcome = Literal["ok", "not_found"]


class StoreStore:
    """CRUD for collections, logs, and their entries.

    Summary of the public surface:
        * metadata: create_collection, create_log, get, list_all, archive, unarchive
        * collection writes: write, update, move, delete
        * log writes: append
        * reads: get_entry, read_latest, read_recent, read_since, read_random,
          read_similar, read_all, keys
        * introspection: exists

    Similarity operations require the caller to pass pre-computed embeddings;
    this layer stays sync. The tool layer (Stage 2) owns async embedding.
    """

    def __init__(self, engine):
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    # ── Metadata ────────────────────────────────────────────────────────────

    def create_collection(
        self, name: str, description: str, recall: RecallMode, archived: bool = False
    ) -> Store:
        return self._create_store(name, StoreType.COLLECTION, description, recall, archived)

    def create_log(
        self, name: str, description: str, recall: RecallMode, archived: bool = False
    ) -> Store:
        return self._create_store(name, StoreType.LOG, description, recall, archived)

    def _create_store(
        self,
        name: str,
        type_: StoreType,
        description: str,
        recall: RecallMode,
        archived: bool,
    ) -> Store:
        with self._session() as session:
            store = Store(
                name=name,
                type=type_.value,
                description=description,
                recall=recall.value,
                archived=archived,
                created_at=datetime.now(UTC),
            )
            session.add(store)
            session.commit()
            session.refresh(store)
            logger.debug("Created %s store %s", type_.value, name)
            return store

    def get(self, name: str) -> Store | None:
        with self._session() as session:
            return session.get(Store, name)

    def list_all(self) -> list[Store]:
        with self._session() as session:
            return list(session.exec(select(Store).order_by(Store.name)).all())

    def archive(self, name: str) -> None:
        self._set_archived(name, True)

    def unarchive(self, name: str) -> None:
        self._set_archived(name, False)

    def _set_archived(self, name: str, archived: bool) -> None:
        with self._session() as session:
            store = session.get(Store, name)
            if store is None:
                raise StoreNotFoundError(name)
            store.archived = archived
            session.add(store)
            session.commit()

    # ── Collection writes ───────────────────────────────────────────────────

    def write(
        self,
        name: str,
        entries: list[EntryInput],
        author: str,
        thresholds: DedupThresholds | None = None,
    ) -> list[WriteResult]:
        """Write entries to a collection with per-entry dedup.

        Returns one WriteResult per input entry with its outcome. Dedup is
        evaluated against existing entries in the same store using the
        configured thresholds (or the module defaults).
        """
        self._require_type(name, StoreType.COLLECTION)
        thresholds = thresholds or DedupThresholds()
        existing = self._load_entries_with_vectors(name)
        results: list[WriteResult] = []
        with self._session() as session:
            for entry in entries:
                results.append(self._write_one(session, name, entry, author, existing, thresholds))
            session.commit()
        return results

    def _write_one(
        self,
        session: Session,
        name: str,
        entry: EntryInput,
        author: str,
        existing: list[tuple[list[float] | None, list[float] | None]],
        thresholds: DedupThresholds,
    ) -> WriteResult:
        if self._is_duplicate(entry.key_embedding, entry.content_embedding, existing, thresholds):
            return WriteResult(key=entry.key, outcome="duplicate")
        row = StoreEntry(
            store_name=name,
            key=entry.key,
            content=entry.content,
            author=author,
            key_embedding=_maybe_serialize(entry.key_embedding),
            content_embedding=_maybe_serialize(entry.content_embedding),
            created_at=datetime.now(UTC),
        )
        session.add(row)
        session.flush()
        existing.append((entry.key_embedding, entry.content_embedding))
        return WriteResult(key=entry.key, outcome="written", entry_id=row.id)

    def update(self, name: str, key: str, content: str, author: str) -> UpdateOutcome:
        """Replace the content of every entry with `key` in a collection.

        Most collections have a single entry per key (dedup keeps it that way),
        but the method operates on all matching rows for safety.
        """
        self._require_type(name, StoreType.COLLECTION)
        with self._session() as session:
            rows = self._entries_by_key(session, name, key)
            if not rows:
                return "not_found"
            for row in rows:
                row.content = content
                row.author = author
                session.add(row)
            session.commit()
            return "ok"

    def move(self, key: str, from_name: str, to_name: str, author: str) -> MoveOutcome:
        """Move every entry with `key` from one collection to another.

        Returns "collision" if a target-collection entry with the same key
        already exists (the caller resolves the collision).
        """
        self._require_type(from_name, StoreType.COLLECTION)
        self._require_type(to_name, StoreType.COLLECTION)
        with self._session() as session:
            src_rows = self._entries_by_key(session, from_name, key)
            if not src_rows:
                return "not_found"
            if self._entries_by_key(session, to_name, key):
                return "collision"
            for row in src_rows:
                row.store_name = to_name
                row.author = author
                session.add(row)
            session.commit()
            return "ok"

    def delete(self, name: str, key: str) -> int:
        """Delete every entry with `key` in a collection. Returns rows removed."""
        self._require_type(name, StoreType.COLLECTION)
        with self._session() as session:
            rows = self._entries_by_key(session, name, key)
            for row in rows:
                session.delete(row)
            session.commit()
            return len(rows)

    # ── Log writes ──────────────────────────────────────────────────────────

    def append(self, name: str, entries: list[LogEntryInput], author: str) -> list[StoreEntry]:
        """Append one or more entries to a log store. No dedup; keyless."""
        self._require_type(name, StoreType.LOG)
        created: list[StoreEntry] = []
        with self._session() as session:
            for entry in entries:
                row = StoreEntry(
                    store_name=name,
                    key=None,
                    content=entry.content,
                    author=author,
                    key_embedding=None,
                    content_embedding=_maybe_serialize(entry.content_embedding),
                    created_at=datetime.now(UTC),
                )
                session.add(row)
                created.append(row)
            session.commit()
            for row in created:
                session.refresh(row)
        return created

    # ── Reads ───────────────────────────────────────────────────────────────

    def get_entry(self, name: str, key: str) -> list[StoreEntry]:
        with self._session() as session:
            return self._entries_by_key(session, name, key)

    def read_latest(self, name: str, k: int | None = None) -> list[StoreEntry]:
        """Return entries newest-first. With `k=None`, returns every entry."""
        with self._session() as session:
            query = (
                select(StoreEntry)
                .where(StoreEntry.store_name == name)
                .order_by(StoreEntry.created_at.desc())  # type: ignore[union-attr]
            )
            if k is not None:
                query = query.limit(k)
            return list(session.exec(query).all())

    def read_recent(
        self, name: str, window_seconds: int, cap: int | None = None
    ) -> list[StoreEntry]:
        cutoff = datetime.now(UTC).timestamp() - window_seconds
        cutoff_dt = datetime.fromtimestamp(cutoff, tz=UTC)
        return self.read_since(name, cutoff_dt, cap)

    def read_since(self, name: str, cursor: datetime, cap: int | None = None) -> list[StoreEntry]:
        with self._session() as session:
            query = (
                select(StoreEntry)
                .where(StoreEntry.store_name == name, StoreEntry.created_at > cursor)
                .order_by(StoreEntry.created_at.asc())  # type: ignore[union-attr]
            )
            if cap is not None:
                query = query.limit(cap)
            return list(session.exec(query).all())

    def read_random(self, name: str, k: int | None = None) -> list[StoreEntry]:
        """Return `k` entries sampled uniformly at random. `k=None` returns all."""
        with self._session() as session:
            rows = list(session.exec(select(StoreEntry).where(StoreEntry.store_name == name)).all())
        if k is None or len(rows) <= k:
            return rows
        return random.sample(rows, k)

    def read_similar(
        self,
        name: str,
        anchor: list[float],
        k: int | None = None,
        floor: float = 0.0,
    ) -> list[StoreEntry]:
        """Return entries sorted by content cosine similarity to ``anchor``.

        Entries without a content_embedding are skipped. Scores below ``floor``
        are excluded. With ``k=None`` every qualifying entry is returned.
        Caller embeds the anchor text ahead of time.
        """
        with self._session() as session:
            rows = list(
                session.exec(
                    select(StoreEntry).where(
                        StoreEntry.store_name == name,
                        StoreEntry.content_embedding.is_not(None),  # type: ignore[union-attr]
                    )
                ).all()
            )
        scored: list[tuple[float, StoreEntry]] = []
        for row in rows:
            if row.content_embedding is None:
                continue
            sim = cosine_similarity(anchor, deserialize_embedding(row.content_embedding))
            if sim >= floor:
                scored.append((sim, row))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        ordered = [row for _, row in scored]
        return ordered if k is None else ordered[:k]

    def read_all(self, name: str) -> list[StoreEntry]:
        with self._session() as session:
            return list(
                session.exec(
                    select(StoreEntry)
                    .where(StoreEntry.store_name == name)
                    .order_by(StoreEntry.created_at.asc())  # type: ignore[union-attr]
                ).all()
            )

    def keys(self, name: str) -> list[str]:
        with self._session() as session:
            rows = list(
                session.exec(
                    select(StoreEntry.key)
                    .where(
                        StoreEntry.store_name == name,
                        StoreEntry.key.is_not(None),  # type: ignore[union-attr]
                    )
                    .order_by(StoreEntry.created_at.asc())  # type: ignore[union-attr]
                ).all()
            )
        seen: set[str] = set()
        ordered: list[str] = []
        for key in rows:
            if key is None or key in seen:
                continue
            seen.add(key)
            ordered.append(key)
        return ordered

    # ── Introspection ───────────────────────────────────────────────────────

    def exists(
        self,
        names: list[str],
        key: str | None,
        key_embedding: list[float] | None,
        content_embedding: list[float] | None,
        thresholds: DedupThresholds | None = None,
    ) -> bool:
        """Check whether an equivalent entry already exists in any of the named stores.

        Runs the same similarity-based dedup used by `write`, plus an exact
        key-match shortcut when a key is supplied. Returns True on the first hit.
        """
        thresholds = thresholds or DedupThresholds()
        for name in names:
            if key is not None and self.get_entry(name, key):
                return True
            existing = self._load_entries_with_vectors(name)
            if self._is_duplicate(key_embedding, content_embedding, existing, thresholds):
                return True
        return False

    # ── Internals ───────────────────────────────────────────────────────────

    def _require_type(self, name: str, expected: StoreType) -> None:
        store = self.get(name)
        if store is None:
            raise StoreNotFoundError(name)
        if store.type != expected.value:
            raise StoreTypeError(f"store '{name}' is a {store.type}, not a {expected.value}")

    def _entries_by_key(self, session: Session, name: str, key: str) -> list[StoreEntry]:
        return list(
            session.exec(
                select(StoreEntry).where(StoreEntry.store_name == name, StoreEntry.key == key)
            ).all()
        )

    def _load_entries_with_vectors(
        self, name: str
    ) -> list[tuple[list[float] | None, list[float] | None]]:
        """Load every entry for `name` as (key_vec, content_vec) pairs.

        Entries without a given embedding contribute None on that axis.
        """
        with self._session() as session:
            rows = list(session.exec(select(StoreEntry).where(StoreEntry.store_name == name)).all())
        return [
            (_maybe_deserialize(r.key_embedding), _maybe_deserialize(r.content_embedding))
            for r in rows
        ]

    def _is_duplicate(
        self,
        key_vec: list[float] | None,
        content_vec: list[float] | None,
        existing: list[tuple[list[float] | None, list[float] | None]],
        thresholds: DedupThresholds,
    ) -> bool:
        for existing_key_vec, existing_content_vec in existing:
            if self._pair_is_duplicate(
                key_vec, content_vec, existing_key_vec, existing_content_vec, thresholds
            ):
                return True
        return False

    def _pair_is_duplicate(
        self,
        key_vec: list[float] | None,
        content_vec: list[float] | None,
        existing_key_vec: list[float] | None,
        existing_content_vec: list[float] | None,
        thresholds: DedupThresholds,
    ) -> bool:
        key_sim = _safe_cosine(key_vec, existing_key_vec)
        content_sim = _safe_cosine(content_vec, existing_content_vec)
        if key_sim is not None and key_sim >= thresholds.key_sim:
            return True
        if content_sim is not None and content_sim >= thresholds.content_sim:
            return True
        if key_sim is not None and content_sim is not None:
            combined = (key_sim + content_sim) / 2
            if combined >= thresholds.combined:
                return True
        return False


def _maybe_serialize(vec: list[float] | None) -> bytes | None:
    return serialize_embedding(vec) if vec is not None else None


def _maybe_deserialize(blob: bytes | None) -> list[float] | None:
    return deserialize_embedding(blob) if blob is not None else None


def _safe_cosine(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    return cosine_similarity(a, b)

"""Memory access layer — collections and logs, unified.

A *memory* is Penny's data primitive: a named, typed container of entries.
Two shapes share one schema:

  * collection — keyed set with similarity-based dedup on write
  * log        — append-only, keyless time-stream

Both live in a single `memory_entry` table with `key` nullable for logs.
Entries are immutable once written — `update` replaces whole content for a
given key.

Dedup on collection writes evaluates three signals against each existing entry
(thresholds live in ``PennyConstants``):

  1. ``tcr(candidate.key, existing.key)`` — token-containment ratio, lexical
  2. ``cos(candidate.key_embedding, existing.key_embedding)`` — paraphrase
  3. ``cos(candidate.content_embedding, existing.content_embedding)``

A candidate is a duplicate if ANY signal meets its strict threshold, OR if any
TWO signals meet their relaxed thresholds. Signals are skipped when either
side is missing (no key on a log entry, no embedding when no model configured),
so the rule degrades gracefully to "only what's comparable fires."
"""

from __future__ import annotations

import logging
import random
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, NamedTuple

from pydantic import BaseModel
from similarity.embeddings import (
    cosine_similarity,
    deserialize_embedding,
    serialize_embedding,
    token_containment_ratio,
)
from sqlmodel import Session, select

from penny.constants import PennyConstants
from penny.database.models import Memory, MemoryEntry

logger = logging.getLogger(__name__)


class MemoryType(StrEnum):
    COLLECTION = "collection"
    LOG = "log"


class RecallMode(StrEnum):
    OFF = "off"
    RECENT = "recent"
    RELEVANT = "relevant"
    ALL = "all"


class MemoryTypeError(Exception):
    """Raised when an operation is called against the wrong memory type."""


class MemoryNotFoundError(Exception):
    """Raised when an operation targets a memory that doesn't exist."""


class DedupThresholds(BaseModel):
    """Per-signal strict + relaxed thresholds for the memory dedup rule."""

    key_tcr_strict: float = PennyConstants.MEMORY_KEY_TCR_STRICT_THRESHOLD
    key_tcr_relaxed: float = PennyConstants.MEMORY_KEY_TCR_RELAXED_THRESHOLD
    key_sim_strict: float = PennyConstants.MEMORY_KEY_SIM_STRICT_THRESHOLD
    key_sim_relaxed: float = PennyConstants.MEMORY_KEY_SIM_RELAXED_THRESHOLD
    content_sim_strict: float = PennyConstants.MEMORY_CONTENT_SIM_STRICT_THRESHOLD
    content_sim_relaxed: float = PennyConstants.MEMORY_CONTENT_SIM_RELAXED_THRESHOLD


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


class EntrySide(NamedTuple):
    """One side of a dedup pair: the key plus its key/content embeddings."""

    key: str | None
    key_vec: list[float] | None
    content_vec: list[float] | None


class MemoryStore:
    """CRUD for memories (collections, logs) and their entries.

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
    ) -> Memory:
        return self._create_memory(name, MemoryType.COLLECTION, description, recall, archived)

    def create_log(
        self, name: str, description: str, recall: RecallMode, archived: bool = False
    ) -> Memory:
        return self._create_memory(name, MemoryType.LOG, description, recall, archived)

    def _create_memory(
        self,
        name: str,
        type_: MemoryType,
        description: str,
        recall: RecallMode,
        archived: bool,
    ) -> Memory:
        with self._session() as session:
            memory = Memory(
                name=name,
                type=type_.value,
                description=description,
                recall=recall.value,
                archived=archived,
                created_at=datetime.now(UTC),
            )
            session.add(memory)
            session.commit()
            session.refresh(memory)
            logger.debug("Created %s memory %s", type_.value, name)
            return memory

    def get(self, name: str) -> Memory | None:
        with self._session() as session:
            return session.get(Memory, name)

    def list_all(self) -> list[Memory]:
        with self._session() as session:
            return list(session.exec(select(Memory).order_by(Memory.name)).all())

    def archive(self, name: str) -> None:
        self._set_archived(name, True)

    def unarchive(self, name: str) -> None:
        self._set_archived(name, False)

    def _set_archived(self, name: str, archived: bool) -> None:
        with self._session() as session:
            memory = session.get(Memory, name)
            if memory is None:
                raise MemoryNotFoundError(name)
            memory.archived = archived
            session.add(memory)
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
        evaluated against existing entries in the same memory using the
        configured thresholds (or the module defaults).
        """
        self._require_type(name, MemoryType.COLLECTION)
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
        existing: list[EntrySide],
        thresholds: DedupThresholds,
    ) -> WriteResult:
        candidate = EntrySide(entry.key, entry.key_embedding, entry.content_embedding)
        if self._is_duplicate(candidate, existing, thresholds):
            return WriteResult(key=entry.key, outcome="duplicate")
        row = MemoryEntry(
            memory_name=name,
            key=entry.key,
            content=entry.content,
            author=author,
            key_embedding=_maybe_serialize(entry.key_embedding),
            content_embedding=_maybe_serialize(entry.content_embedding),
            created_at=datetime.now(UTC),
        )
        session.add(row)
        session.flush()
        existing.append(candidate)
        return WriteResult(key=entry.key, outcome="written", entry_id=row.id)

    def update(self, name: str, key: str, content: str, author: str) -> UpdateOutcome:
        """Replace the content of every entry with `key` in a collection.

        Most collections have a single entry per key (dedup keeps it that way),
        but the method operates on all matching rows for safety.
        """
        self._require_type(name, MemoryType.COLLECTION)
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
        self._require_type(from_name, MemoryType.COLLECTION)
        self._require_type(to_name, MemoryType.COLLECTION)
        with self._session() as session:
            src_rows = self._entries_by_key(session, from_name, key)
            if not src_rows:
                return "not_found"
            if self._entries_by_key(session, to_name, key):
                return "collision"
            for row in src_rows:
                row.memory_name = to_name
                row.author = author
                session.add(row)
            session.commit()
            return "ok"

    def delete(self, name: str, key: str) -> int:
        """Delete every entry with `key` in a collection. Returns rows removed."""
        self._require_type(name, MemoryType.COLLECTION)
        with self._session() as session:
            rows = self._entries_by_key(session, name, key)
            for row in rows:
                session.delete(row)
            session.commit()
            return len(rows)

    # ── Log writes ──────────────────────────────────────────────────────────

    def append(self, name: str, entries: list[LogEntryInput], author: str) -> list[MemoryEntry]:
        """Append one or more entries to a log memory. No dedup; keyless."""
        self._require_type(name, MemoryType.LOG)
        created: list[MemoryEntry] = []
        with self._session() as session:
            for entry in entries:
                row = MemoryEntry(
                    memory_name=name,
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

    def get_entry(self, name: str, key: str) -> list[MemoryEntry]:
        with self._session() as session:
            return self._entries_by_key(session, name, key)

    def read_latest(self, name: str, k: int | None = None) -> list[MemoryEntry]:
        """Return entries newest-first. With `k=None`, returns every entry."""
        with self._session() as session:
            query = (
                select(MemoryEntry)
                .where(MemoryEntry.memory_name == name)
                .order_by(MemoryEntry.created_at.desc())  # type: ignore[union-attr]
            )
            if k is not None:
                query = query.limit(k)
            return list(session.exec(query).all())

    def read_recent(
        self, name: str, window_seconds: int, cap: int | None = None
    ) -> list[MemoryEntry]:
        cutoff = datetime.now(UTC).timestamp() - window_seconds
        cutoff_dt = datetime.fromtimestamp(cutoff, tz=UTC)
        return self.read_since(name, cutoff_dt, cap)

    def read_since(self, name: str, cursor: datetime, cap: int | None = None) -> list[MemoryEntry]:
        with self._session() as session:
            query = (
                select(MemoryEntry)
                .where(MemoryEntry.memory_name == name, MemoryEntry.created_at > cursor)
                .order_by(MemoryEntry.created_at.asc())  # type: ignore[union-attr]
            )
            if cap is not None:
                query = query.limit(cap)
            return list(session.exec(query).all())

    def read_random(self, name: str, k: int | None = None) -> list[MemoryEntry]:
        """Return `k` entries sampled uniformly at random. `k=None` returns all."""
        with self._session() as session:
            rows = list(
                session.exec(select(MemoryEntry).where(MemoryEntry.memory_name == name)).all()
            )
        if k is None or len(rows) <= k:
            return rows
        return random.sample(rows, k)

    def read_similar(
        self,
        name: str,
        anchor: list[float],
        k: int | None = None,
        floor: float = 0.0,
    ) -> list[MemoryEntry]:
        """Return entries sorted by content cosine similarity to ``anchor``.

        Entries without a content_embedding are skipped. Scores below ``floor``
        are excluded. With ``k=None`` every qualifying entry is returned.
        Caller embeds the anchor text ahead of time.
        """
        rows = self._embedded_rows(name)
        ordered = self._rank_against_anchor(rows, anchor, floor)
        return ordered if k is None else ordered[:k]

    def _embedded_rows(self, name: str) -> list[MemoryEntry]:
        with self._session() as session:
            return list(
                session.exec(
                    select(MemoryEntry).where(
                        MemoryEntry.memory_name == name,
                        MemoryEntry.content_embedding.is_not(None),  # type: ignore[union-attr]
                    )
                ).all()
            )

    def _rank_against_anchor(
        self, rows: list[MemoryEntry], anchor: list[float], floor: float
    ) -> list[MemoryEntry]:
        scored: list[tuple[float, MemoryEntry]] = []
        for row in rows:
            if row.content_embedding is None:
                continue
            sim = cosine_similarity(anchor, deserialize_embedding(row.content_embedding))
            if sim >= floor:
                scored.append((sim, row))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [row for _, row in scored]

    def read_all(self, name: str) -> list[MemoryEntry]:
        with self._session() as session:
            return list(
                session.exec(
                    select(MemoryEntry)
                    .where(MemoryEntry.memory_name == name)
                    .order_by(MemoryEntry.created_at.asc())  # type: ignore[union-attr]
                ).all()
            )

    def keys(self, name: str) -> list[str]:
        with self._session() as session:
            rows = list(
                session.exec(
                    select(MemoryEntry.key)
                    .where(
                        MemoryEntry.memory_name == name,
                        MemoryEntry.key.is_not(None),  # type: ignore[union-attr]
                    )
                    .order_by(MemoryEntry.created_at.asc())  # type: ignore[union-attr]
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
        """Check whether an equivalent entry already exists in any of the named memories.

        Runs the same similarity-based dedup used by `write`, plus an exact
        key-match shortcut when a key is supplied. Returns True on the first hit.
        """
        thresholds = thresholds or DedupThresholds()
        candidate = EntrySide(key, key_embedding, content_embedding)
        for name in names:
            if key is not None and self.get_entry(name, key):
                return True
            existing = self._load_entries_with_vectors(name)
            if self._is_duplicate(candidate, existing, thresholds):
                return True
        return False

    # ── Internals ───────────────────────────────────────────────────────────

    def _require_type(self, name: str, expected: MemoryType) -> None:
        memory = self.get(name)
        if memory is None:
            raise MemoryNotFoundError(name)
        if memory.type != expected.value:
            raise MemoryTypeError(f"memory '{name}' is a {memory.type}, not a {expected.value}")

    def _entries_by_key(self, session: Session, name: str, key: str) -> list[MemoryEntry]:
        return list(
            session.exec(
                select(MemoryEntry).where(MemoryEntry.memory_name == name, MemoryEntry.key == key)
            ).all()
        )

    def _load_entries_with_vectors(self, name: str) -> list[EntrySide]:
        """Load every entry for `name` as EntrySide triples (key, key_vec, content_vec).

        Entries without a given embedding or key contribute None on that axis.
        """
        with self._session() as session:
            rows = list(
                session.exec(select(MemoryEntry).where(MemoryEntry.memory_name == name)).all()
            )
        return [
            EntrySide(
                r.key,
                _maybe_deserialize(r.key_embedding),
                _maybe_deserialize(r.content_embedding),
            )
            for r in rows
        ]

    def _is_duplicate(
        self,
        candidate: EntrySide,
        existing: list[EntrySide],
        thresholds: DedupThresholds,
    ) -> bool:
        return any(self._pair_is_duplicate(candidate, side, thresholds) for side in existing)

    def _pair_is_duplicate(
        self,
        candidate: EntrySide,
        existing: EntrySide,
        thresholds: DedupThresholds,
    ) -> bool:
        """Apply the three-signal dedup rule to a single candidate/existing pair.

        Signals that can't be computed (missing keys, missing embeddings) are
        skipped. Fire if any one signal hits its strict threshold or any two
        signals hit their relaxed thresholds.
        """
        signals = _score_signals(candidate, existing, thresholds)
        if any(score >= strict for score, strict, _ in signals):
            return True
        relaxed_hits = sum(1 for score, _, relaxed in signals if score >= relaxed)
        return relaxed_hits >= 2


def _score_signals(
    candidate: EntrySide,
    existing: EntrySide,
    thresholds: DedupThresholds,
) -> list[tuple[float, float, float]]:
    """Return (score, strict_threshold, relaxed_threshold) for every applicable signal."""
    out: list[tuple[float, float, float]] = []
    if candidate.key is not None and existing.key is not None:
        out.append(
            (
                token_containment_ratio(candidate.key, existing.key),
                thresholds.key_tcr_strict,
                thresholds.key_tcr_relaxed,
            )
        )
    key_cos = _safe_cosine(candidate.key_vec, existing.key_vec)
    if key_cos is not None:
        out.append((key_cos, thresholds.key_sim_strict, thresholds.key_sim_relaxed))
    content_cos = _safe_cosine(candidate.content_vec, existing.content_vec)
    if content_cos is not None:
        out.append((content_cos, thresholds.content_sim_strict, thresholds.content_sim_relaxed))
    return out


def _maybe_serialize(vec: list[float] | None) -> bytes | None:
    return serialize_embedding(vec) if vec is not None else None


def _maybe_deserialize(blob: bytes | None) -> list[float] | None:
    return deserialize_embedding(blob) if blob is not None else None


def _safe_cosine(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    return cosine_similarity(a, b)

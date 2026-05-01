"""Pydantic arg models for the memory tool surface.

Each tool validates its kwargs through one of these models as its first line,
per the Pydantic-everywhere rule. Most read tools accept ``k: int | None``
meaning "no cap — return every entry" when omitted; this matches the access
layer's signature.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ── Metadata ────────────────────────────────────────────────────────────────


class CreateMemoryArgs(BaseModel):
    """Shared shape for ``collection_create`` and ``log_create``.

    ``extraction_prompt`` only applies to collections — it's the body of
    instructions the per-collection collector subagent will run with on
    each cycle (read recent log entries, extract structured records,
    write/update/delete).  Logs ignore it.  Optional at the schema level
    so migrations and tests can create collections without one; the chat
    agent's prompt instructs it to always supply one for user-created
    collections so the new collection gets a collector immediately.
    """

    name: str
    description: str
    recall: str  # "off" | "recent" | "relevant" | "all" — validated in the store layer
    extraction_prompt: str | None = None


class MemoryNameArgs(BaseModel):
    """One-field args for ``archive`` / ``unarchive`` / read-all / keys."""

    memory: str


class CollectionUpdateArgs(BaseModel):
    """Update a collection's metadata.

    All fields after ``name`` are optional — only the ones explicitly set
    are applied.  ``recall`` is validated in the store layer.
    """

    name: str
    description: str | None = None
    recall: str | None = None
    extraction_prompt: str | None = None
    collector_interval_seconds: int | None = None


# ── Collection reads ────────────────────────────────────────────────────────


class CollectionGetArgs(BaseModel):
    """Exact key lookup in a collection."""

    memory: str
    key: str


class ReadLatestArgs(BaseModel):
    """Newest-first; ``k=None`` returns all."""

    memory: str
    k: int | None = None


class ReadRandomArgs(BaseModel):
    """Random sample; ``k=None`` returns all."""

    memory: str
    k: int | None = None


class ReadSimilarArgs(BaseModel):
    """Top-k by content cosine similarity to ``anchor`` (embedded by the tool).

    The similarity floor is fixed (``MEMORY_RELEVANT_ABSOLUTE_FLOOR`` plus
    the adaptive cluster gate) — the model can't override it, since cosine
    thresholds are opaque values it has no grounding to pick.
    """

    memory: str
    anchor: str
    k: int | None = None


# ── Log-specific reads ──────────────────────────────────────────────────────


class ReadRecentArgs(BaseModel):
    """Entries created within the past ``window_seconds`` seconds."""

    memory: str
    window_seconds: int
    cap: int | None = None


class ReadNextArgs(BaseModel):
    """Cursor-based read: entries newer than the agent's last committed cursor."""

    memory: str
    cap: int | None = None


# ── Collection writes ───────────────────────────────────────────────────────


class CollectionEntrySpec(BaseModel):
    """One entry in a ``collection_write`` batch."""

    key: str
    content: str


class CollectionWriteArgs(BaseModel):
    """Batched write to a collection with dedup applied per entry."""

    memory: str
    entries: list[CollectionEntrySpec] = Field(min_length=1)


class UpdateEntryArgs(BaseModel):
    """Replace content for an existing key in a collection."""

    memory: str
    key: str
    content: str


class CollectionMoveArgs(BaseModel):
    """Move an entry between collections by key."""

    key: str
    from_memory: str
    to_memory: str


class CollectionDeleteEntryArgs(BaseModel):
    """Delete an entry from a collection by key."""

    memory: str
    key: str


# ── Log writes ──────────────────────────────────────────────────────────────


class LogAppendArgs(BaseModel):
    """Append one keyless entry to a log."""

    memory: str
    content: str


# ── Introspection ───────────────────────────────────────────────────────────


class ExistsArgs(BaseModel):
    """Cross-memory dedup probe used by thinking-class agents before writes."""

    memories: list[str] = Field(min_length=1)
    content: str
    key: str | None = None


class DoneArgs(BaseModel):
    """Empty body — signals the orchestration loop to exit."""

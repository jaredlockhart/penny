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
    """Shared shape for ``collection_create`` and ``log_create``."""

    name: str
    description: str
    recall: str  # "off" | "recent" | "relevant" | "all" — validated in the store layer


class MemoryNameArgs(BaseModel):
    """One-field args for ``archive`` / ``unarchive`` / read-all / keys."""

    memory: str


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
    """Top-k by content cosine similarity to ``anchor`` (embedded by the tool)."""

    memory: str
    anchor: str
    k: int | None = None
    floor: float = 0.0


# ── Log-specific reads ──────────────────────────────────────────────────────


class ReadRecentArgs(BaseModel):
    """Entries created within the past ``window_seconds`` seconds."""

    memory: str
    window_seconds: int
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


class CollectionUpdateArgs(BaseModel):
    """Replace content for an existing key in a collection."""

    memory: str
    key: str
    content: str


class CollectionKeyArgs(BaseModel):
    """``collection_delete`` — one key in one collection."""

    memory: str
    key: str


class CollectionMoveArgs(BaseModel):
    """Move an entry between collections by key."""

    key: str
    from_memory: str
    to_memory: str


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

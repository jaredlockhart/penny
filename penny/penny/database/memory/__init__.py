"""The memory layer — polymorphic ``Memory`` objects + the ``MemoryStore`` registry.

Import surface: callers ask ``db.memory(name)`` for a ``Memory`` and call methods
on it (the object refuses ops that don't fit its shape / read-only-ness), and
``db.memories`` for registry operations (create/list/archive, ``exists``,
backfill).  The shapes and facades live in :mod:`objects`; the registry and
dispatch in :mod:`store`; shared value types in :mod:`types`.
"""

from penny.database.memory.objects import (
    Collection,
    Log,
    Memory,
    MessageLogMemory,
    RunLog,
)
from penny.database.memory.store import MemoryStore
from penny.database.memory.types import (
    DedupThresholds,
    EntryInput,
    EntrySide,
    Inclusion,
    LogEntryInput,
    MemoryAccessError,
    MemoryAlreadyExistsError,
    MemoryNotFoundError,
    MemoryType,
    MemoryTypeError,
    MoveOutcome,
    ReadOnlyMemoryError,
    RecallMode,
    UpdateOutcome,
    WriteOutcome,
    WriteResult,
    WrongShapeError,
    slug,
)

__all__ = [
    "Collection",
    "DedupThresholds",
    "EntryInput",
    "EntrySide",
    "Inclusion",
    "Log",
    "LogEntryInput",
    "Memory",
    "MemoryAccessError",
    "MemoryAlreadyExistsError",
    "MemoryNotFoundError",
    "MemoryStore",
    "MemoryType",
    "MemoryTypeError",
    "MessageLogMemory",
    "MoveOutcome",
    "ReadOnlyMemoryError",
    "RecallMode",
    "RunLog",
    "UpdateOutcome",
    "WriteOutcome",
    "WriteResult",
    "WrongShapeError",
    "slug",
]

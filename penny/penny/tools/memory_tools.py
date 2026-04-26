"""Tool-layer wrappers over the memory access layer.

Every tool validates its kwargs through a Pydantic args model as its first
line (per CLAUDE.md), calls ``db.memories.*``, and returns a serializable
string the model can reason over.

Author attribution is passed explicitly: write-capable tools take an
``author: str`` at construction time (the agent that owns the tool).
``build_memory_tools(db, embedding_client, author)`` is the factory each
agent calls with its own ``self.name`` so writes are attributed correctly.

Tools that need embeddings (writes, similarity reads, ``exists``) take an
``LlmClient`` in ``__init__``. If no embedding client is configured they
degrade gracefully: writes proceed without key/content vectors, similarity
reads return empty.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from penny.database import Database
from penny.database.memory_store import (
    DedupThresholds,
    EntryInput,
    LogEntryInput,
    RecallMode,
    WriteResult,
)
from penny.database.models import Memory, MemoryEntry
from penny.llm.similarity import embed_text
from penny.tools.base import Tool
from penny.tools.memory_args import (
    CollectionEntrySpec,
    CollectionGetArgs,
    CollectionMoveArgs,
    CollectionUpdateArgs,
    CollectionWriteArgs,
    CreateMemoryArgs,
    DoneArgs,
    ExistsArgs,
    LogAppendArgs,
    MemoryNameArgs,
    ReadLatestArgs,
    ReadRandomArgs,
    ReadRecentArgs,
    ReadSimilarArgs,
)

if TYPE_CHECKING:
    from penny.llm.client import LlmClient

logger = logging.getLogger(__name__)


_RECALL_MODES = ", ".join(m.value for m in RecallMode)


# ── Shared formatting ───────────────────────────────────────────────────────


def _format_entries(entries: list[MemoryEntry]) -> str:
    """Render a list of entries as a bulleted string the model can read.

    Keyed entries (collection) include the key; keyless entries (log) show
    just content. Empty lists produce a clear "no entries" sentinel so the
    model doesn't confuse absence with error.
    """
    if not entries:
        return "(no entries)"
    lines = []
    for entry in entries:
        prefix = f"[{entry.key}] " if entry.key else ""
        lines.append(f"- {prefix}{entry.content}")
    return "\n".join(lines)


def _format_memory_row(memory: Memory) -> str:
    archived = " [archived]" if memory.archived else ""
    return (
        f"- {memory.name} ({memory.type}, recall={memory.recall}){archived}: {memory.description}"
    )


# ── Metadata ────────────────────────────────────────────────────────────────


class CollectionCreateTool(Tool):
    """Create a new keyed collection."""

    name = "collection_create"
    description = (
        "Create a new keyed collection. Collections store entries by key with "
        "similarity-based dedup on write. Provide a short description and a "
        f"recall mode ({_RECALL_MODES})."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Unique collection name"},
            "description": {
                "type": "string",
                "description": "One-line summary shown in the memory registry",
            },
            "recall": {
                "type": "string",
                "enum": [m.value for m in RecallMode],
                "description": "How the chat agent surfaces this collection in ambient context",
            },
        },
        "required": ["name", "description", "recall"],
    }

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, **kwargs: Any) -> str:
        args = CreateMemoryArgs(**kwargs)
        self._db.memories.create_collection(args.name, args.description, RecallMode(args.recall))
        return f"Created collection '{args.name}'."


class LogCreateTool(Tool):
    """Create a new append-only log."""

    name = "log_create"
    description = (
        "Create a new append-only log. Logs store keyless entries in time order "
        "and are meant for streams of events (messages, measurements, etc.). "
        f"Provide a short description and a recall mode ({_RECALL_MODES})."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Unique log name"},
            "description": {"type": "string", "description": "One-line summary"},
            "recall": {
                "type": "string",
                "enum": [m.value for m in RecallMode],
                "description": "How the chat agent surfaces this log in ambient context",
            },
        },
        "required": ["name", "description", "recall"],
    }

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, **kwargs: Any) -> str:
        args = CreateMemoryArgs(**kwargs)
        self._db.memories.create_log(args.name, args.description, RecallMode(args.recall))
        return f"Created log '{args.name}'."


class CollectionArchiveTool(Tool):
    """Archive a collection — keeps data, removes it from ambient recall."""

    name = "collection_archive"
    description = (
        "Archive a collection. The data stays intact but the collection is "
        "excluded from the chat agent's ambient recall until unarchived."
    )
    parameters = {
        "type": "object",
        "properties": {"memory": {"type": "string"}},
        "required": ["memory"],
    }

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, **kwargs: Any) -> str:
        args = MemoryNameArgs(**kwargs)
        self._db.memories.archive(args.memory)
        return f"Archived '{args.memory}'."


class CollectionUnarchiveTool(Tool):
    """Restore a previously archived collection to ambient recall."""

    name = "collection_unarchive"
    description = "Unarchive a previously archived collection."
    parameters = {
        "type": "object",
        "properties": {"memory": {"type": "string"}},
        "required": ["memory"],
    }

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, **kwargs: Any) -> str:
        args = MemoryNameArgs(**kwargs)
        self._db.memories.unarchive(args.memory)
        return f"Unarchived '{args.memory}'."


class ListMemoriesTool(Tool):
    """List every memory's name, type, recall mode, and description."""

    name = "list_memories"
    description = (
        "List every memory (collection or log) with its type, recall mode, "
        "archived state, and description. Use this to discover what's available."
    )
    parameters = {"type": "object", "properties": {}}

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, **kwargs: Any) -> str:
        memories = self._db.memories.list_all()
        if not memories:
            return "(no memories)"
        return "\n".join(_format_memory_row(m) for m in memories)


# ── Collection reads ────────────────────────────────────────────────────────


class CollectionGetTool(Tool):
    """Exact-key lookup in a collection."""

    name = "collection_get"
    description = (
        "Look up an entry by its exact key in a collection. Returns the entry's "
        "content if found, or a 'not found' message otherwise."
    )
    parameters = {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "key": {"type": "string"},
        },
        "required": ["memory", "key"],
    }

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, **kwargs: Any) -> str:
        args = CollectionGetArgs(**kwargs)
        rows = self._db.memories.get_entry(args.memory, args.key)
        if not rows:
            return f"Key '{args.key}' not found in '{args.memory}'."
        return _format_entries(rows)


class CollectionReadLatestTool(Tool):
    """Return the newest entries in a collection."""

    name = "collection_read_latest"
    description = (
        "Return the newest entries in a collection, newest first. Omit ``k`` to return every entry."
    )
    parameters = {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "k": {"type": "integer", "description": "Max entries; omit for all"},
        },
        "required": ["memory"],
    }

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, **kwargs: Any) -> str:
        args = ReadLatestArgs(**kwargs)
        entries = self._db.memories.read_latest(args.memory, args.k)
        return _format_entries(entries)


class CollectionReadRandomTool(Tool):
    """Return entries sampled uniformly at random from a collection."""

    name = "collection_read_random"
    description = "Return ``k`` entries sampled uniformly at random. Omit ``k`` to return all."
    parameters = {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "k": {"type": "integer"},
        },
        "required": ["memory"],
    }

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, **kwargs: Any) -> str:
        args = ReadRandomArgs(**kwargs)
        entries = self._db.memories.read_random(args.memory, args.k)
        return _format_entries(entries)


class CollectionReadSimilarTool(Tool):
    """Return collection entries most similar to an anchor phrase."""

    name = "collection_read_similar"
    description = (
        "Return entries from a collection ordered by content similarity to an "
        "``anchor`` phrase. Useful for finding related preferences, facts, etc."
    )
    parameters = {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "anchor": {
                "type": "string",
                "description": "Text whose meaning drives the similarity search",
            },
            "k": {"type": "integer", "description": "Max entries; omit for all above ``floor``"},
            "floor": {
                "type": "number",
                "description": "Minimum cosine similarity; default 0.0 (include everything)",
            },
        },
        "required": ["memory", "anchor"],
    }

    def __init__(self, db: Database, llm_client: LlmClient | None) -> None:
        self._db = db
        self._llm = llm_client

    async def execute(self, **kwargs: Any) -> str:
        args = ReadSimilarArgs(**kwargs)
        vec = await embed_text(self._llm, args.anchor)
        if vec is None:
            logger.warning(
                "%s: similarity search unavailable — no embedding model configured", self.name
            )
            return "(similarity search unavailable — no embedding model configured)"
        entries = self._db.memories.read_similar(args.memory, vec, args.k, args.floor)
        return _format_entries(entries)


class CollectionReadAllTool(Tool):
    """Return every entry in a collection, oldest first."""

    name = "collection_read_all"
    description = "Return every entry in a collection, oldest first."
    parameters = {
        "type": "object",
        "properties": {"memory": {"type": "string"}},
        "required": ["memory"],
    }

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, **kwargs: Any) -> str:
        args = MemoryNameArgs(**kwargs)
        entries = self._db.memories.read_all(args.memory)
        return _format_entries(entries)


class CollectionKeysTool(Tool):
    """List the unique keys currently in a collection."""

    name = "collection_keys"
    description = "List the unique keys in a collection (insertion order)."
    parameters = {
        "type": "object",
        "properties": {"memory": {"type": "string"}},
        "required": ["memory"],
    }

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, **kwargs: Any) -> str:
        args = MemoryNameArgs(**kwargs)
        keys = self._db.memories.keys(args.memory)
        if not keys:
            return "(no keys)"
        return "\n".join(f"- {key}" for key in keys)


# ── Collection writes ───────────────────────────────────────────────────────


class CollectionWriteTool(Tool):
    """Write entries to a collection with similarity-based dedup."""

    name = "collection_write"
    description = (
        "Write one or more entries to a collection. Each entry has a short "
        "``key`` (topic/identifier) and a longer ``content`` body. Dedup runs "
        "per entry — duplicates are reported but not treated as errors."
    )
    parameters = {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "entries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["key", "content"],
                },
            },
        },
        "required": ["memory", "entries"],
    }

    def __init__(self, db: Database, llm_client: LlmClient | None, author: str) -> None:
        self._db = db
        self._llm = llm_client
        self._author = author

    async def execute(self, **kwargs: Any) -> str:
        args = CollectionWriteArgs(**kwargs)
        entries = [await self._build_entry(spec) for spec in args.entries]
        results = self._db.memories.write(args.memory, entries, author=self._author)
        return self._format_results(args.memory, results)

    async def _build_entry(self, spec: CollectionEntrySpec) -> EntryInput:
        return EntryInput(
            key=spec.key,
            content=spec.content,
            key_embedding=await embed_text(self._llm, spec.key),
            content_embedding=await embed_text(self._llm, spec.content),
        )

    def _format_results(self, memory: str, results: list[WriteResult]) -> str:
        written = [r.key for r in results if r.outcome == "written"]
        duplicates = [r.key for r in results if r.outcome == "duplicate"]
        if duplicates:
            logger.info(
                "collection_write: %d duplicate(s) rejected in %s: %s",
                len(duplicates),
                memory,
                ", ".join(duplicates),
            )
        parts: list[str] = []
        if written:
            noun = "entry" if len(written) == 1 else "entries"
            parts.append(f"Wrote {len(written)} {noun} to '{memory}': {', '.join(written)}.")
        if duplicates:
            parts.append(f"Rejected as duplicates: {', '.join(duplicates)}.")
        return " ".join(parts) if parts else "(no entries written)"


class CollectionUpdateTool(Tool):
    """Replace the content of an existing entry."""

    name = "collection_update"
    description = (
        "Replace the content of an existing entry in a collection, identified "
        "by key. Returns an error if the key doesn't exist."
    )
    parameters = {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "key": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["memory", "key", "content"],
    }

    def __init__(self, db: Database, author: str) -> None:
        self._db = db
        self._author = author

    async def execute(self, **kwargs: Any) -> str:
        args = CollectionUpdateArgs(**kwargs)
        outcome = self._db.memories.update(args.memory, args.key, args.content, self._author)
        if outcome == "not_found":
            return f"Key '{args.key}' not found in '{args.memory}'."
        return f"Updated '{args.key}' in '{args.memory}'."


class CollectionMoveTool(Tool):
    """Move an entry between collections by key."""

    name = "collection_move"
    description = (
        "Move the entry with the given key from one collection to another. "
        "Fails with 'collision' if the target already has an entry with that key."
    )
    parameters = {
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "from_memory": {"type": "string"},
            "to_memory": {"type": "string"},
        },
        "required": ["key", "from_memory", "to_memory"],
    }

    def __init__(self, db: Database, author: str) -> None:
        self._db = db
        self._author = author

    async def execute(self, **kwargs: Any) -> str:
        args = CollectionMoveArgs(**kwargs)
        outcome = self._db.memories.move(
            args.key, args.from_memory, args.to_memory, author=self._author
        )
        if outcome == "not_found":
            return f"Key '{args.key}' not found in '{args.from_memory}'."
        if outcome == "collision":
            return f"Cannot move: '{args.to_memory}' already has a '{args.key}' entry."
        return f"Moved '{args.key}' from '{args.from_memory}' to '{args.to_memory}'."


# ── Log reads ───────────────────────────────────────────────────────────────


class LogReadLatestTool(Tool):
    """Return the newest entries in a log, newest first."""

    name = "log_read_latest"
    description = "Return the newest entries in a log, newest first. Omit ``k`` to return all."
    parameters = {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "k": {"type": "integer"},
        },
        "required": ["memory"],
    }

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, **kwargs: Any) -> str:
        args = ReadLatestArgs(**kwargs)
        entries = self._db.memories.read_latest(args.memory, args.k)
        return _format_entries(entries)


class LogReadRecentTool(Tool):
    """Return log entries created within the past ``window_seconds`` seconds."""

    name = "log_read_recent"
    description = (
        "Return entries created within the past ``window_seconds`` seconds, "
        "oldest first. Use for 'what just happened' queries."
    )
    parameters = {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "window_seconds": {"type": "integer"},
            "cap": {"type": "integer", "description": "Max entries; omit for all"},
        },
        "required": ["memory", "window_seconds"],
    }

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, **kwargs: Any) -> str:
        args = ReadRecentArgs(**kwargs)
        entries = self._db.memories.read_recent(args.memory, args.window_seconds, args.cap)
        return _format_entries(entries)


class LogReadSimilarTool(Tool):
    """Return log entries most similar to an anchor phrase."""

    name = "log_read_similar"
    description = (
        "Return log entries ordered by content similarity to an ``anchor`` phrase. "
        "Useful for finding historically-relevant statements, past browse results, etc."
    )
    parameters = {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "anchor": {"type": "string"},
            "k": {"type": "integer"},
            "floor": {"type": "number"},
        },
        "required": ["memory", "anchor"],
    }

    def __init__(self, db: Database, llm_client: LlmClient | None) -> None:
        self._db = db
        self._llm = llm_client

    async def execute(self, **kwargs: Any) -> str:
        args = ReadSimilarArgs(**kwargs)
        vec = await embed_text(self._llm, args.anchor)
        if vec is None:
            logger.warning(
                "%s: similarity search unavailable — no embedding model configured", self.name
            )
            return "(similarity search unavailable — no embedding model configured)"
        entries = self._db.memories.read_similar(args.memory, vec, args.k, args.floor)
        return _format_entries(entries)


class LogReadAllTool(Tool):
    """Return every entry in a log, oldest first."""

    name = "log_read_all"
    description = "Return every entry in a log, oldest first."
    parameters = {
        "type": "object",
        "properties": {"memory": {"type": "string"}},
        "required": ["memory"],
    }

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, **kwargs: Any) -> str:
        args = MemoryNameArgs(**kwargs)
        entries = self._db.memories.read_all(args.memory)
        return _format_entries(entries)


# ── Log writes ──────────────────────────────────────────────────────────────


class LogAppendTool(Tool):
    """Append a keyless entry to a log."""

    name = "log_append"
    description = "Append one keyless entry to a log. No dedup runs; every append is stored."
    parameters = {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["memory", "content"],
    }

    def __init__(self, db: Database, llm_client: LlmClient | None, author: str) -> None:
        self._db = db
        self._llm = llm_client
        self._author = author

    async def execute(self, **kwargs: Any) -> str:
        args = LogAppendArgs(**kwargs)
        vec = await embed_text(self._llm, args.content)
        self._db.memories.append(
            args.memory,
            [LogEntryInput(content=args.content, content_embedding=vec)],
            author=self._author,
        )
        return f"Appended to '{args.memory}'."


# ── Introspection / lifecycle ───────────────────────────────────────────────


class ExistsTool(Tool):
    """Probe whether an equivalent entry already exists across a set of memories."""

    name = "exists"
    description = (
        "Check whether an entry equivalent to the given key/content already "
        "exists in any of the listed memories. Uses the same similarity-based "
        "dedup rule as ``collection_write``. Use this before writing to avoid "
        "duplicates that span multiple collections."
    )
    parameters = {
        "type": "object",
        "properties": {
            "memories": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Names of memories to search",
            },
            "content": {"type": "string"},
            "key": {"type": "string", "description": "Optional — enables exact-key shortcut"},
        },
        "required": ["memories", "content"],
    }

    def __init__(
        self,
        db: Database,
        llm_client: LlmClient | None,
        thresholds: DedupThresholds | None = None,
    ) -> None:
        self._db = db
        self._llm = llm_client
        self._thresholds = thresholds

    async def execute(self, **kwargs: Any) -> str:
        args = ExistsArgs(**kwargs)
        key_vec = await embed_text(self._llm, args.key) if args.key else None
        content_vec = await embed_text(self._llm, args.content)
        found = self._db.memories.exists(
            args.memories,
            args.key,
            key_vec,
            content_vec,
            thresholds=self._thresholds,
        )
        return "yes" if found else "no"


class DoneTool(Tool):
    """Signal the orchestration loop that the agent has finished its work."""

    name = "done"
    description = (
        "Call this when you have completed the task and have no more tool calls "
        "to make. Takes no arguments."
    )
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        DoneArgs(**kwargs)
        return "done"


# ── Factory ─────────────────────────────────────────────────────────────────


def build_memory_tools(db: Database, llm_client: LlmClient | None, author: str) -> list[Tool]:
    """Construct the full memory tool surface for an agent.

    Each agent calls this with its own ``self.name`` as ``author``; that
    value is baked into every write-capable tool so entries get attributed
    correctly without any ambient/contextvar state.

    Callers can slice this list by tool name to give each agent a narrower
    surface (e.g. the preference extractor only wants ``collection_write``
    and ``done``).  The factory centralizes dependency wiring so individual
    agents don't have to juggle ``db`` / ``llm_client`` / ``author`` across
    21 constructors.
    """
    return [
        # Metadata
        CollectionCreateTool(db),
        LogCreateTool(db),
        CollectionArchiveTool(db),
        CollectionUnarchiveTool(db),
        ListMemoriesTool(db),
        # Collection reads
        CollectionGetTool(db),
        CollectionReadLatestTool(db),
        CollectionReadRandomTool(db),
        CollectionReadSimilarTool(db, llm_client),
        CollectionReadAllTool(db),
        CollectionKeysTool(db),
        # Collection writes
        CollectionWriteTool(db, llm_client, author),
        CollectionUpdateTool(db, author),
        CollectionMoveTool(db, author),
        # Log reads
        LogReadLatestTool(db),
        LogReadRecentTool(db),
        LogReadSimilarTool(db, llm_client),
        LogReadAllTool(db),
        # Log writes
        LogAppendTool(db, llm_client, author),
        # Introspection / lifecycle
        ExistsTool(db, llm_client),
        DoneTool(),
    ]

"""Memory inventory and ambient recall assembly for agent system prompts.

Two helpers, used together in chat and individually in background.

``build_memory_inventory`` lists every non-archived memory's name, type,
and description.  Goes in EVERY agent's system prompt so the model can
discover what's available without calling ``list_memories``.

``build_recall_block`` assembles ambient recall content from the active
memories — only chat needs this, since background agents read memory
state explicitly per their task.  Each memory is rendered by its recall
mode:

  off      — skipped
  recent   — newest-first slice (``read_latest``)
  relevant — hybrid similarity over the conversation window
             (``read_similar_hybrid``; skipped without embedding)
  all      — full set in insertion order (``read_all``)

Relevant-mode recall scores each candidate as
``max(weighted_decay_over_history, cosine_to_current) - α * centrality``.
Strong direct hits on the current message stand alone; vague follow-ups
still benefit from earlier conversation drift.  When ``conversation_history``
is omitted the behaviour collapses cleanly to single-anchor cosine ranking.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory_store import MemoryType, RecallMode
from penny.database.models import Memory, MemoryEntry
from penny.llm.models import LlmError

if TYPE_CHECKING:
    from penny.llm.client import LlmClient

logger = logging.getLogger(__name__)


def build_memory_inventory(db: Database) -> str | None:
    """Inventory of every non-archived memory: name, type, description.

    Includes memories with ``recall=off`` so the model knows what tool
    calls are possible for on-demand reads.  Sorted alphabetically by
    name for stable prompt structure.  Goes in every agent's system
    prompt — chat and background alike — so the model never needs to
    call ``list_memories``.
    """
    memories = sorted(
        (m for m in db.memories.list_all() if not m.archived),
        key=lambda m: m.name,
    )
    if not memories:
        return None
    lines = ["### Memory Inventory"]
    for memory in memories:
        lines.append(f"- {memory.name} ({memory.type}) — {memory.description}")
    return "\n".join(lines)


async def build_recall_block(
    db: Database,
    llm_client: LlmClient | None,
    current_message: str | None,
    similarity_floor: float = 0.0,
    conversation_history: list[str] | None = None,
    limit: int = 99,
) -> str | None:
    """Assemble ambient recall content for all active memories.

    Renders verbatim entries from each non-archived memory whose
    ``recall`` mode is not ``'off'``.  Chat-only — background agents
    stay focused and read memory explicitly per their task.

    ``limit`` caps how many entries each memory contributes, applied to all
    three modes (recent, relevant, all).  Without it a large log/collection
    would dump every entry into the prompt.  Production callers (ChatAgent)
    pass ``RECALL_LIMIT`` from runtime config; the in-test default of 99
    is high enough that existing assertions on small fixture sets still
    cover the same entries they did before the cap.
    """
    anchors = await _embed_conversation_anchors(llm_client, current_message, conversation_history)
    sections: list[str] = []
    for memory in _active_memories(db):
        section = _render_memory(db, memory, anchors, similarity_floor, limit)
        if section:
            sections.append(section)
    return "\n\n".join(sections) if sections else None


# ── Helpers ──────────────────────────────────────────────────────────────────


def _active_memories(db: Database) -> list[Memory]:
    """Non-archived memories with recall != 'off'."""
    return [m for m in db.memories.list_all() if not m.archived and m.recall != RecallMode.OFF]


async def _embed_conversation_anchors(
    llm_client: LlmClient | None,
    current_message: str | None,
    history: list[str] | None,
) -> list[list[float]] | None:
    """Embed history + current_message as ordered anchors (oldest→newest).

    Returns ``None`` when no current message is available, when no client
    is configured, or when the embed call fails.  Empty history is fine —
    the result is just ``[current_embedding]``.
    """
    if not current_message or llm_client is None:
        return None
    texts = [*(history or []), current_message]
    try:
        return await llm_client.embed(texts)
    except LlmError:
        logger.warning("Skipping relevant recall — conversation embedding failed")
        return None


def _render_memory(
    db: Database,
    memory: Memory,
    anchors: list[list[float]] | None,
    similarity_floor: float,
    limit: int,
) -> str | None:
    """Dispatch to the correct renderer for a single memory's recall mode."""
    mode = RecallMode(memory.recall)
    if mode == RecallMode.RECENT:
        entries = db.memories.read_latest(memory.name, k=limit)
    elif mode == RecallMode.RELEVANT:
        entries = _relevant_entries(db, memory, anchors, similarity_floor, limit)
    elif mode == RecallMode.ALL:
        entries = db.memories.read_all(memory.name)[:limit]
    else:
        return None
    if not entries:
        return None
    return _format_memory_section(memory, entries)


def _relevant_entries(
    db: Database,
    memory: Memory,
    anchors: list[list[float]] | None,
    floor: float,
    limit: int,
) -> list[MemoryEntry]:
    """Run hybrid similarity, expanding logs with their temporal neighbors.

    For log-shaped memories the similarity hits are augmented with every
    entry within ±``MEMORY_RELEVANT_NEIGHBOR_WINDOW_MINUTES`` of any hit's
    timestamp, so a single keyword match pulls in the surrounding
    conversation rather than a single line stripped of context.

    Logs also exclude entries newer than ``now -
    MEMORY_RELEVANT_SELF_MATCH_CUTOFF_SECONDS`` from the corpus before
    scoring.  Channel ingress writes the user's incoming message to
    ``user-messages`` immediately, and that entry would otherwise self-
    match the current-message anchor at cosine ≈ 1.0; temporal
    expansion would then anchor on that self-match and pull in the
    recent conversation, drowning out historical hits.
    """
    if not anchors:
        return []
    not_after = _self_match_cutoff_for(memory)
    hits = db.memories.read_similar_hybrid(
        memory.name, anchors, k=limit, floor=floor, not_after=not_after
    )
    if not hits or memory.type != MemoryType.LOG.value:
        return hits
    return db.memories.expand_with_temporal_neighbors(
        memory.name,
        hits,
        PennyConstants.MEMORY_RELEVANT_NEIGHBOR_WINDOW_MINUTES,
    )


def _self_match_cutoff_for(memory: Memory) -> datetime | None:
    """Self-match exclusion timestamp for log-shaped memories, ``None`` for collections.

    Collections are keyed sets — the current message isn't an entry in
    them, so there's nothing to exclude.  Logs receive the current turn
    via channel ingress and need the cutoff.
    """
    if memory.type != MemoryType.LOG.value:
        return None
    return datetime.now(UTC) - timedelta(
        seconds=PennyConstants.MEMORY_RELEVANT_SELF_MATCH_CUTOFF_SECONDS
    )


def _format_memory_section(memory: Memory, entries: list[MemoryEntry]) -> str:
    """Render a single memory's header + entries as a context subsection."""
    lines = [f"### {memory.name}", memory.description, ""]
    for entry in entries:
        prefix = f"[{entry.key}] " if entry.key else ""
        lines.append(f"- {prefix}{entry.content}")
    return "\n".join(lines)

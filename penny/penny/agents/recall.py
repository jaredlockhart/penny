"""Ambient recall assembly for the chat agent system prompt.

``build_recall_block`` assembles recall context from all non-archived memories
whose ``recall`` mode is not ``'off'``.  Each memory is rendered by mode:

  off      — skipped
  recent   — newest-first slice (``read_latest``)
  relevant — similarity-ranked slice (``read_similar``; skipped without embedding)
  all      — full set in insertion order (``read_all``)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from penny.database import Database
from penny.database.memory_store import RecallMode
from penny.database.models import Memory, MemoryEntry
from penny.llm.similarity import embed_text

if TYPE_CHECKING:
    from penny.llm.client import LlmClient

logger = logging.getLogger(__name__)


async def build_recall_block(
    db: Database,
    llm_client: LlmClient | None,
    current_message: str | None,
    similarity_floor: float = 0.0,
) -> str | None:
    """Assemble recall context for all active memories — summary method."""
    sections: list[str] = []
    for memory in _active_memories(db):
        section = await _render_memory(db, llm_client, memory, current_message, similarity_floor)
        if section:
            sections.append(section)
    return "\n\n".join(sections) if sections else None


# ── Helpers ──────────────────────────────────────────────────────────────────


def _active_memories(db: Database) -> list[Memory]:
    """Non-archived memories with recall != 'off'."""
    return [m for m in db.memories.list_all() if not m.archived and m.recall != RecallMode.OFF]


async def _render_memory(
    db: Database,
    llm_client: LlmClient | None,
    memory: Memory,
    current_message: str | None,
    similarity_floor: float,
) -> str | None:
    """Dispatch to the correct renderer for a single memory's recall mode."""
    mode = RecallMode(memory.recall)
    if mode == RecallMode.RECENT:
        entries = db.memories.read_latest(memory.name)
    elif mode == RecallMode.RELEVANT:
        entries = await _relevant_entries(
            db, llm_client, memory.name, current_message, similarity_floor
        )
    elif mode == RecallMode.ALL:
        entries = db.memories.read_all(memory.name)
    else:
        return None
    if not entries:
        return None
    return _format_memory_section(memory, entries)


async def _relevant_entries(
    db: Database,
    llm_client: LlmClient | None,
    name: str,
    current_message: str | None,
    floor: float,
) -> list[MemoryEntry]:
    """Embed current_message and return similarity-ranked entries."""
    if not current_message:
        return []
    anchor = await embed_text(llm_client, current_message)
    if anchor is None:
        logger.warning("Skipping relevant recall for '%s' — embedding unavailable", name)
        return []
    return db.memories.read_similar(name, anchor, floor=floor)


def _format_memory_section(memory: Memory, entries: list[MemoryEntry]) -> str:
    """Render a single memory's header + entries as a context subsection."""
    lines = [f"### {memory.name}", memory.description, ""]
    for entry in entries:
        prefix = f"[{entry.key}] " if entry.key else ""
        lines.append(f"- {prefix}{entry.content}")
    return "\n".join(lines)

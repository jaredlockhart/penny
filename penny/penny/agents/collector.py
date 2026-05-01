"""Collector — single dispatcher agent for per-collection extraction.

One ``Collector`` instance runs in the background.  Each cycle it picks
the most-overdue ready collection from ``memory`` (where
``extraction_prompt IS NOT NULL`` and
``now - last_collected_at >= collector_interval_seconds``), binds itself
to that target, runs the agent loop with the target's extraction prompt
as instructions and a tool surface scoped to writes against that
collection only, then stamps ``last_collected_at = now``.

Dispatcher pattern (vs. one stateful ``CollectorAgent`` per collection):
  - No agent registry to keep in sync with the DB; reading the DB each
    cycle IS the source of truth.
  - Hot-add for free — chat creates a new collection mid-session, the
    next dispatcher tick picks it up.
  - Per-collection cadence respected naturally via the readiness check.
  - The agent_cursor table still partitions per (agent_name, memory_name),
    so log read cursors stay correctly partitioned per collection even
    though one agent identity (``"collector"``) drives all of them.
"""

from __future__ import annotations

from datetime import UTC, datetime

from penny.agents.base import BackgroundAgent
from penny.constants import PennyConstants
from penny.database.models import Memory
from penny.tools.base import Tool
from penny.tools.memory_tools import DoneTool, build_memory_tools


class Collector(BackgroundAgent):
    """Single dispatcher agent — picks the most-overdue ready collection per cycle."""

    name = "collector"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Set per-cycle inside ``execute``.  The cycle is single-threaded
        # by the scheduler, so transient instance state is safe.
        self._current_target: Memory | None = None

    async def execute(self) -> bool:
        target = self._next_ready_collection()
        if target is None:
            return False
        try:
            self._current_target = target
            success = await super().execute()
        finally:
            # Stamp regardless of success — what matters for cadence is
            # that the *check* happened.  A persistently-failing collection
            # otherwise gets re-attempted every tick.
            self.db.memories.mark_collected(target.name)
            self._current_target = None
        return success

    # ── Per-cycle prompt + tool surface ───────────────────────────────────

    async def _build_system_prompt(self, user: str | None) -> str:
        """System prompt for the bound target — re-fetched each cycle.

        Reading from the DB instead of caching means a chat-side
        ``collection_update`` call that changes ``extraction_prompt`` is
        picked up on the very next collector cycle, no restart needed.
        """
        target = self._require_target()
        fresh = self.db.memories.get(target.name) or target
        return self._compose_prompt(fresh)

    @staticmethod
    def _compose_prompt(target: Memory) -> str:
        return (
            f"You are the collector for the `{target.name}` collection.\n"
            f"Description: {target.description}\n\n"
            f"{target.extraction_prompt}"
        )

    def get_tools(self) -> list[Tool]:
        """Scoped surface — entry mutations pinned to the bound target.

        Includes ``send_message`` from ``BackgroundAgent.get_tools`` when
        a channel is wired (notify-shaped collectors deliver via this).
        Excludes lifecycle tools entirely — those are chat-only.
        """
        target = self._require_target()
        tools: list[Tool] = build_memory_tools(
            self.db,
            self._embedding_model_client,
            agent_name=self.name,
            scope=target.name,
        )
        tools.append(self._build_browse_tool(author=self.name))
        tools.append(DoneTool())
        if self._channel is not None:
            from penny.tools.send_message import SendMessageTool

            tools.append(
                SendMessageTool(
                    channel=self._channel,
                    agent_name=self.name,
                    db=self.db,
                    config=self.config,
                )
            )
        return tools

    def _require_target(self) -> Memory:
        if self._current_target is None:
            raise RuntimeError(
                "Collector tool surface accessed outside an execute() cycle "
                "— self._current_target is None"
            )
        return self._current_target

    # ── Dispatcher selection ──────────────────────────────────────────────

    def _next_ready_collection(self) -> Memory | None:
        """Pick the most-overdue ready collection, or None if all caught up."""
        now = datetime.now(UTC)
        ready = [m for m in self.db.memories.list_all() if self._is_ready(m, now)]
        if not ready:
            return None
        return min(ready, key=self._overdue_sort_key)

    @staticmethod
    def _is_ready(memory: Memory, now: datetime) -> bool:
        if memory.archived or memory.extraction_prompt is None:
            return False
        if memory.last_collected_at is None:
            return True  # Never run — always ready
        interval = memory.collector_interval_seconds or PennyConstants.COLLECTOR_DEFAULT_INTERVAL
        elapsed = (now - _aware(memory.last_collected_at)).total_seconds()
        return elapsed >= interval

    @staticmethod
    def _overdue_sort_key(memory: Memory) -> datetime:
        # Earliest last_collected_at runs first; never-collected sorts to the front.
        return (
            _aware(memory.last_collected_at)
            if memory.last_collected_at
            else datetime.min.replace(tzinfo=UTC)
        )


def _aware(dt: datetime) -> datetime:
    """SQLite returns naive datetimes; assume UTC and attach tzinfo."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)

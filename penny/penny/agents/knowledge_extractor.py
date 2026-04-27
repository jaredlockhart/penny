"""KnowledgeExtractorAgent — turns browse results into durable knowledge.

Reads new entries from ``browse-results`` via ``log_read_next``,
summarizes each page, and writes summaries to the ``knowledge``
collection.  User-independent — runs once per cycle regardless of
which user is active.  Cursor commits only on a clean ``done()``
exit.
"""

from __future__ import annotations

from penny.agents.base import Agent
from penny.constants import HistoryPromptType


class KnowledgeExtractorAgent(Agent):
    """Background worker that builds the knowledge base from browse output."""

    name = "knowledge-extractor"
    prompt_type = HistoryPromptType.KNOWLEDGE_EXTRACTION

    # Cap on agentic loop iterations.  The expected flow is
    # read_next → (get + write/update)*N → done.  N scales with the
    # number of new page entries, so 16 gives headroom for several
    # pages per cycle.
    MAX_STEPS = 16

    def get_max_steps(self) -> int:
        return self.MAX_STEPS

    async def execute(self) -> bool:
        """User-independent: run a single cycle without iterating users."""
        return await self._run_cycle(user=None)

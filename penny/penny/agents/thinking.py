"""ThinkingAgent — Penny's autonomous inner monologue.

Runs on the scheduler when the system is idle.  Each cycle is a
fully model-driven agent loop: the prompt steers the model through
reading a seed topic from ``likes``, scanning ``dislikes``,
browsing the web, drafting a thought, deduping against existing
thoughts via ``exists``, and writing the result to
``unnotified-thoughts``.

The agent class is the system prompt + an unnotified-thoughts
queue cap.  Everything else is the shared agent shell.
"""

from __future__ import annotations

import logging

from penny.agents.base import Agent
from penny.constants import PennyConstants, ThinkingPromptType

logger = logging.getLogger(__name__)


class ThinkingAgent(Agent):
    """Background worker that produces inner-monologue thoughts."""

    name = "thinking"
    prompt_type = ThinkingPromptType.CYCLE

    def get_max_steps(self) -> int:
        """Read from config so /config changes take effect immediately."""
        return int(self.config.runtime.INNER_MONOLOGUE_MAX_STEPS)

    async def execute_for_user(self, user: str) -> bool:
        """Skip when the unnotified queue is full; otherwise run a cycle."""
        max_unnotified = int(self.config.runtime.MAX_UNNOTIFIED_THOUGHTS)
        unnotified = self.db.memories.read_latest(PennyConstants.MEMORY_UNNOTIFIED_THOUGHTS)
        if len(unnotified) >= max_unnotified:
            logger.info(
                "Skipping thinking: %d unnotified thoughts (max %d)",
                len(unnotified),
                max_unnotified,
            )
            return True
        return await self._run_cycle(user)

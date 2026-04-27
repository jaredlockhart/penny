"""ThinkingAgent — Penny's autonomous inner monologue.

Runs on the scheduler when the system is idle.  Each cycle is a
fully model-driven agent loop: the system prompt steers the model
through reading a seed topic from ``likes``, scanning ``dislikes``,
browsing the web, drafting a thought, deduping against existing
thoughts via ``exists``, and writing the result to
``unnotified-thoughts``.

All bespoke logic — seed selection, dislike matching, dedup, summary
LLM, JSON parsing, last_thought_at tracking — is gone.  The agent is
just its prompt + the shared tool surface.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from penny.agents.base import Agent
from penny.constants import ThinkingPromptType
from penny.prompts import Prompt
from penny.tools.memory_tools import DoneTool

logger = logging.getLogger(__name__)


class ThinkingAgent(Agent):
    """Background worker that produces inner-monologue thoughts."""

    name = "thinking"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Tools stay available on the final agentic step — thinking
        # doesn't have a "speak to the user" terminator like chat does.
        self._keep_tools_on_final_step = True

    def get_max_steps(self) -> int:
        """Read from config so /config changes take effect immediately."""
        return int(self.config.runtime.INNER_MONOLOGUE_MAX_STEPS)

    async def execute_for_user(self, user: str) -> bool:
        """Skip when the unnotified queue is full; otherwise run a cycle."""
        max_unnotified = int(self.config.runtime.MAX_UNNOTIFIED_THOUGHTS)
        total = self.db.thoughts.count_unnotified(user)
        if total >= max_unnotified:
            logger.info("Skipping thinking: %d unnotified thoughts (max %d)", total, max_unnotified)
            return True
        return await self._run_thinking_cycle()

    async def _run_thinking_cycle(self) -> bool:
        """Run one model-driven thinking cycle.

        Tool surface is the agent's default (memory + browse).  Cursor
        and write attribution use ``self.name``.  Success means the
        model called ``done()`` to exit gracefully — anything else
        (max steps, model error) returns False.
        """
        self._install_tools(self.get_tools(user=None))
        run_id = uuid.uuid4().hex
        response = await self.run(
            prompt="",
            max_steps=self.get_max_steps(),
            system_prompt=Prompt.THINKING_SYSTEM_PROMPT,
            run_id=run_id,
            prompt_type=ThinkingPromptType.CYCLE,
        )
        return any(record.tool == DoneTool.name for record in response.tool_calls)

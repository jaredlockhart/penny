"""HistoryAgent — preference and knowledge extraction.

Runs on a schedule.  Each cycle:
1. Runs the knowledge-extractor agent loop (user-independent),
   which reads new browse-results entries via ``log_read_next``,
   summarizes each page, and writes durable summaries to the
   ``knowledge`` collection.
2. Per user: runs the preference-extractor agent loop, which reads
   new user messages via ``log_read_next``, identifies likes /
   dislikes, and writes them via ``collection_write``.

Both flows are pure model-driven shells: their bespoke prompts
steer the model through ``log_read_next`` → ``collection_*`` →
``done`` against the standard memory tool surface.  Cursor
advancement is two-phase per flow — pending while the loop runs,
committed only on a clean ``done()`` exit so a failed run replays
on the next schedule.
"""

from __future__ import annotations

import logging
import uuid

from penny.agents.base import Agent
from penny.agents.models import ControllerResponse
from penny.constants import HistoryPromptType
from penny.prompts import Prompt
from penny.tools.memory_tools import DoneTool, LogReadNextTool

logger = logging.getLogger(__name__)


class HistoryAgent(Agent):
    """Background worker that extracts knowledge and user preferences."""

    name = "history"

    # Identities for the two flows hosted on this agent.  Each is used
    # both as the cursor key for the source log it reads from and as
    # the author stamped on writes the flow produces.  Distinct from
    # ``name`` because one HistoryAgent instance hosts multiple flows
    # with their own per-flow cursors and attribution.
    PREFERENCE_EXTRACTOR_NAME = "preference-extractor"
    KNOWLEDGE_EXTRACTOR_NAME = "knowledge-extractor"

    # Cap on agentic loop iterations for the preference extractor.  The
    # expected flow is read_next → write(likes) → write(dislikes) → done,
    # so 8 leaves headroom for re-reads or batched writes without
    # letting a runaway loop tail through forever.
    PREFERENCE_EXTRACTOR_MAX_STEPS = 8

    # Cap on agentic loop iterations for the knowledge extractor.  The
    # expected flow is read_next → (get + write/update)*N → done.  N
    # scales with the number of new page entries, so 16 gives headroom
    # for several pages per cycle.
    KNOWLEDGE_EXTRACTOR_MAX_STEPS = 16

    def get_max_steps(self) -> int:
        """Cap on agentic loop iterations for HistoryAgent flows."""
        return self.PREFERENCE_EXTRACTOR_MAX_STEPS

    async def execute(self) -> bool:
        """Run knowledge extraction (user-independent), then per-user work."""
        knowledge_work = await self._run_knowledge_extractor()
        user_work = await super().execute()
        return knowledge_work or user_work

    async def execute_for_user(self, user: str) -> bool:
        """Run the preference-extractor agent loop for the user."""
        return await self._run_preference_extractor()

    # ── Knowledge extraction (agent loop) ─────────────────────────────────

    async def _run_knowledge_extractor(self) -> bool:
        """Read new browse-results pages and persist summaries to ``knowledge``.

        Same shape as the preference extractor: install full tools
        attributed to the extractor identity, run the loop with the
        extractor system prompt, commit the cursor only on a clean
        ``done()`` exit.  Failed runs leave the cursor untouched so
        the next pass replays the same pages.
        """
        tools = self._build_full_tools(agent_name=self.KNOWLEDGE_EXTRACTOR_NAME)
        log_read_next = next(t for t in tools if isinstance(t, LogReadNextTool))
        self._install_tools(tools)

        run_id = uuid.uuid4().hex
        response = await self.run(
            prompt="",
            max_steps=self.KNOWLEDGE_EXTRACTOR_MAX_STEPS,
            system_prompt=Prompt.KNOWLEDGE_EXTRACTOR_SYSTEM_PROMPT,
            run_id=run_id,
            prompt_type=HistoryPromptType.KNOWLEDGE_EXTRACTION,
        )
        if self._extractor_run_succeeded(response):
            log_read_next.commit_pending()
            return True
        log_read_next.discard_pending()
        return False

    # ── Preference extraction (agent loop) ────────────────────────────────

    async def _run_preference_extractor(self) -> bool:
        """Read new user-messages, identify likes/dislikes, write them.

        The flow is fully model-driven: the agent loop is given the full
        memory tool surface stamped with the extractor identity, the
        prompt steers it to call ``log_read_next``/``collection_write``/
        ``done``, and the cursor advances only on a clean run.  Failed
        runs (max steps, model error) leave the cursor untouched so the
        next pass sees the same messages.

        Every agent gets every tool — narrowing the surface per flow
        only encodes the prompt's intent twice.  The model decides what
        to call from the prompt's instructions.
        """
        tools = self._build_full_tools(agent_name=self.PREFERENCE_EXTRACTOR_NAME)
        log_read_next = next(t for t in tools if isinstance(t, LogReadNextTool))
        self._install_tools(tools)

        run_id = uuid.uuid4().hex
        response = await self.run(
            prompt="",
            max_steps=self.get_max_steps(),
            system_prompt=Prompt.PREFERENCE_EXTRACTOR_SYSTEM_PROMPT,
            run_id=run_id,
            prompt_type=HistoryPromptType.PREFERENCE_EXTRACTION,
        )
        if self._extractor_run_succeeded(response):
            log_read_next.commit_pending()
            return True
        log_read_next.discard_pending()
        return False

    @staticmethod
    def _extractor_run_succeeded(response: ControllerResponse) -> bool:
        """Success iff the model called ``done()`` to exit the loop.

        Done is the only graceful terminator for these background
        flows.  Hitting max_steps, raising a model error, or producing
        fallback text all leave the cursor untouched so the next run
        can retry.
        """
        return any(record.tool == DoneTool.name for record in response.tool_calls)

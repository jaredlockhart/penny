"""ThinkingAgent — Penny's autonomous inner monologue.

Runs on the scheduler after extraction. Each cycle is a continuous
thinking loop where Penny thinks out loud, uses tools, and accumulates
reasoning. At the end, the monologue is summarized and stored as a thought.
"""

from __future__ import annotations

import logging
import random

from penny.agents.base import Agent
from penny.agents.models import ChatMessage, MessageRole
from penny.prompts import Prompt

logger = logging.getLogger(__name__)

# Probability of a free-thinking cycle (no seed, no context, just vibes)
FREE_THINKING_PROBABILITY = 1 / 3


class ThinkingAgent(Agent):
    """Autonomous inner monologue — Penny's conscious mind.

    Each cycle picks ONE random seed topic from history
    to focus on, keeping thinking rotating across interests.

    Context matrix — each mode gets tailored context:

        Mode       | Entities    | Thoughts | Dislikes | Tools       | Steps
        ---------- | ----------- | -------- | -------- | ----------- | -----
        Seeded     | anchor=seed | 10       | yes      | search+news | 10
        Browse News| -           | 10       | yes      | search+news | 10
        Free Think | -           | -        | -        | search+news | 10
        Step N     | anchor=mono | 10       | yes      | (kept)      | -

    All modes include profile (user name) except free think.
    Step N rebuilds system prompt each step, anchoring entities
    to accumulated monologue text.

    Thinking loop::

        [user]       Think about {seed topic}...
        [assistant]  <inner monologue text>          <- captured
        [user]       keep exploring
        [assistant]  <tool call: search(...)>         <- tool executed
        [tool]       <search results>
        [assistant]  <inner monologue reflecting>     <- captured
        ...

    Summary step: monologue summarized via THINKING_REPORT_PROMPT,
    stored as a thought in db.thoughts.

    Seed topic sources: positive user preferences.
    """

    THOUGHT_CONTEXT_LIMIT = 10

    name = "inner_monologue"

    def __init__(self, **kwargs: object) -> None:
        kwargs["system_prompt"] = Prompt.THINKING_SYSTEM_PROMPT
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.max_steps = int(self.config.runtime.INNER_MONOLOGUE_MAX_STEPS)
        self._inner_monologue: list[str] = []
        self._free_thinking: bool = False
        self._seed_topic: str | None = None
        self._seed_pref_id: int | None = None

    # ── Execution hooks ──────────────────────────────────────────────────

    async def get_prompt(self, user: str) -> str | None:
        """Pick a seed topic or let Penny free-think (~1/3 of the time)."""
        self._inner_monologue = []
        self._free_thinking = False
        self._seed_topic = None
        self._seed_pref_id = None

        if random.random() < FREE_THINKING_PROBABILITY:
            logger.info("Free thinking cycle for %s", user)
            self._free_thinking = True
            return Prompt.THINKING_FREE

        pool = self.db.preferences.get_least_recent_positive(user)
        if not pool:
            logger.info("No preferences for %s, browsing news", user)
            return Prompt.THINKING_BROWSE_NEWS

        pref = random.choice(pool)
        self._seed_topic = pref.content
        self._seed_pref_id = pref.id
        logger.info("Thinking seed: %s", pref.content)
        return Prompt.THINKING_SEED.format(seed=pref.content)

    async def get_context(self, user: str) -> str:
        """Slim context — profile, entities (seed-anchored), thoughts, and dislikes.

        Free-thinking cycles get no context so Penny explores freely.
        Browse news skips entities (no meaningful anchor).
        Seeded cycles anchor entities to the seed topic.
        """
        if self._free_thinking:
            return ""
        sections: list[str | None] = [
            self._build_profile_context(user, None),
            self._build_thought_context(user),
            self._build_dislike_context(user),
        ]
        return "\n\n".join(s for s in sections if s)

    async def after_run(self, user: str) -> bool:
        """Produce a detailed research report and store as a thought."""
        if not self._inner_monologue:
            return False
        combined = "\n\n---\n\n".join(self._inner_monologue)
        report = await self._summarize_text(combined, Prompt.THINKING_REPORT_PROMPT)
        if report:
            self.db.thoughts.add(user, report)
            if self._seed_pref_id is not None:
                self.db.preferences.mark_thought_about(self._seed_pref_id)
            logger.info("[inner_monologue] %s", report[:200])
        return True

    # ── Loop hooks ─────────────────────────────────────────────────────────

    def on_response(self, response) -> None:
        """Capture text content from every response (even tool-call ones)."""
        content = response.content.strip()
        if content:
            self._inner_monologue.append(content)

    async def handle_text_step(
        self, response, messages: list[dict], step: int, is_final: bool
    ) -> bool:
        """Inject 'keep exploring' continuation to drive the thinking loop."""
        if is_final:
            return False
        content = response.content.strip()
        messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=content).to_dict())
        if not self._free_thinking:
            await self._rebuild_system_prompt(messages)
        messages.append(ChatMessage(role=MessageRole.USER, content="keep exploring").to_dict())
        return True

    async def _rebuild_system_prompt(self, messages: list[dict]) -> None:
        """Rebuild system prompt with entities anchored to accumulated monologue."""
        assert self._current_user is not None
        anchor = "\n".join(self._inner_monologue)
        sections: list[str | None] = [
            self._build_profile_context(self._current_user, anchor),
            self._build_thought_context(self._current_user),
            self._build_dislike_context(self._current_user),
        ]
        context_text = "\n\n".join(s for s in sections if s)
        fresh = self._build_messages(
            prompt="",
            history=None,
            system_prompt=Prompt.THINKING_SYSTEM_PROMPT,
            context=context_text,
        )
        messages[0] = fresh[0]

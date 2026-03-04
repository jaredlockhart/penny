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


class ThinkingAgent(Agent):
    """Autonomous inner monologue — Penny's conscious mind.

    Each cycle picks ONE random seed topic from history
    to focus on, keeping thinking rotating across interests.

    System message (slim — profile, entities, thoughts only)::

        [system]
        <current datetime>
        <Prompt.PENNY_IDENTITY>
        <user profile>

        ## Relevant Knowledge
        - <top K entities by embedding similarity to accumulated monologue>

        ## Recent Background Thinking
        <today's thought summaries — used to avoid repetition>

        <Prompt.THINKING_SYSTEM_PROMPT with {tools} listing>

    Thinking loop::

        [user]       Think about {seed topic} and explore interesting related topics.
        [assistant]  <inner monologue text>          ← captured
        [user]       keep exploring
        [assistant]  <tool call: search(...)>         ← tool executed
        [tool]       <search results>
        [assistant]  <inner monologue reflecting>     ← captured
        ...
        <INNER_MONOLOGUE_MAX_STEPS iterations. Entity context rebuilt
        each text step, anchored to accumulated monologue.>

    Summary step::

        [system]  <Prompt.SUMMARIZE_TO_PARAGRAPH>
        [user]    <all inner monologue text joined with --->
        <background model, no tools. Result stored in db.thoughts.>

    Seed topic sources:
        - history entries (daily conversation topic bullets)
    """

    THOUGHT_CONTEXT_LIMIT = 50

    name = "inner_monologue"

    def __init__(self, **kwargs: object) -> None:
        kwargs["system_prompt"] = Prompt.THINKING_SYSTEM_PROMPT
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.max_steps = int(self.config.runtime.INNER_MONOLOGUE_MAX_STEPS)
        self._inner_monologue: list[str] = []

    # ── Execution hooks ──────────────────────────────────────────────────

    def get_prompt(self, user: str) -> str | None:
        """Pick a random seed topic from conversation history, or browse news."""
        self._inner_monologue = []
        topics: list[str] = []
        limit = int(self.config.runtime.HISTORY_CONTEXT_LIMIT)
        for entry in self.db.history.get_recent(user, "daily", limit=limit):
            for line in entry.topics.splitlines():
                topic = line.strip().lstrip("- ").strip()
                if topic:
                    topics.append(topic)
        if not topics:
            logger.info("No seed topics for %s, browsing news", user)
            return Prompt.THINKING_BROWSE_NEWS
        seed = random.choice(topics)
        logger.info("Thinking seed: %s", seed)
        return Prompt.THINKING_SEED.format(seed=seed)

    async def get_context(self, user: str) -> str:
        """Slim context — profile, entities, and thoughts only."""
        sections: list[str | None] = [
            self._build_profile_context(user, None),
            await self._build_entity_context(user, None),
            self._build_thought_context(user),
        ]
        return "\n\n".join(s for s in sections if s)

    async def after_run(self, user: str) -> bool:
        """Summarize the inner monologue and store as a thought."""
        if not self._inner_monologue:
            return False
        combined = "\n\n---\n\n".join(self._inner_monologue)
        summary = await self._summarize_text(combined, Prompt.SUMMARIZE_TO_PARAGRAPH)
        if summary:
            self.db.thoughts.add(user, summary)
            logger.info("[inner_monologue] %s", summary[:200])
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
        assert self._current_user is not None
        anchor = "\n".join(self._inner_monologue)
        sections: list[str | None] = [
            self._build_profile_context(self._current_user, anchor),
            await self._build_entity_context(self._current_user, anchor),
            self._build_thought_context(self._current_user),
        ]
        context_text = "\n\n".join(s for s in sections if s)
        fresh = self._build_messages(
            prompt="",
            history=None,
            system_prompt=Prompt.THINKING_SYSTEM_PROMPT,
            context=context_text,
        )
        messages[0] = fresh[0]
        messages.append(ChatMessage(role=MessageRole.USER, content="keep exploring").to_dict())
        return True

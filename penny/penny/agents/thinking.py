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
from penny.ollama.embeddings import cosine_similarity
from penny.ollama.similarity import embed_text
from penny.prompts import Prompt

logger = logging.getLogger(__name__)

# Probability of a free-thinking cycle (no seed, no context, just vibes)
FREE_THINKING_PROBABILITY = 1 / 3

# Minimum word count for a thought to be stored (filters model planning text)
MIN_THOUGHT_WORDS = 50


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

    def _build_thought_context(self, sender: str) -> str | None:
        """Build thought context scoped to the current seed preference.

        Only prior thoughts about the same preference are relevant —
        they show what Penny already found so she can dig deeper or
        find new angles instead of repeating herself.
        """
        if not self._seed_pref_id:
            return None
        try:
            thoughts = self.db.thoughts.get_recent_by_preference(sender, self._seed_pref_id)
            if not thoughts:
                return None
            lines = [t.content for t in thoughts]
            logger.debug("Built preference-scoped thought context (%d thoughts)", len(thoughts))
            return "## Recent Background Thinking\n" + "\n\n".join(lines)
        except Exception:
            logger.warning("Thought context retrieval failed, proceeding without")
            return None

    async def after_run(self, user: str) -> bool:
        """Summarize the monologue, dedup against same-seed thoughts, and store."""
        if not self._inner_monologue:
            return False
        combined = "\n\n---\n\n".join(self._inner_monologue)
        report = await self._summarize_text(combined, Prompt.THINKING_REPORT_PROMPT)
        if report and len(report.split()) < MIN_THOUGHT_WORDS:
            logger.info(
                "[inner_monologue] report too short (%d words), skipping", len(report.split())
            )
            report = ""
        if report and not await self._is_duplicate_thought(user, report):
            self.db.thoughts.add(user, report, preference_id=self._seed_pref_id)
            logger.info("[inner_monologue] %s", report[:200])
        elif report:
            logger.info("[inner_monologue] duplicate thought, skipping storage")
        if self._seed_pref_id is not None:
            self.db.preferences.mark_thought_about(self._seed_pref_id)
        return True

    # ── Model calls ────────────────────────────────────────────────────────

    async def _summarize_text(self, content: str, prompt: str) -> str:
        """Summarize content using the model. Returns empty string on failure."""
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ]
        try:
            response = await self._model_client.chat(messages=messages)
            return response.content.strip()
        except Exception as e:
            logger.error("Summarization failed: %s", e)
            return ""

    async def _is_duplicate_thought(self, user: str, report: str) -> bool:
        """Check if report is too similar to a same-preference thought via embedding similarity."""
        if not self._embedding_model_client or not self._seed_pref_id:
            return False
        threshold = float(self.config.runtime.THOUGHT_DEDUP_EMBEDDING_THRESHOLD)
        report_vec = await embed_text(self._embedding_model_client, report)
        if report_vec is None:
            return False
        recent = self.db.thoughts.get_recent_by_preference(user, self._seed_pref_id)
        for thought in recent:
            thought_vec = await embed_text(self._embedding_model_client, thought.content)
            if thought_vec is None:
                continue
            sim = cosine_similarity(report_vec, thought_vec)
            if sim >= threshold:
                logger.info(
                    "[inner_monologue] sim=%.3f >= %.2f vs thought #%s (pref #%s)",
                    sim,
                    threshold,
                    thought.id,
                    self._seed_pref_id,
                )
                return True
        return False

    # ── Loop hooks ─────────────────────────────────────────────────────────

    def on_response(self, response) -> None:
        """Capture text content from every response (even tool-call ones)."""
        content = response.content.strip()
        if content:
            self._inner_monologue.append(content)

    async def handle_text_step(
        self, response, messages: list[dict], step: int, is_final: bool
    ) -> bool:
        """Inject 'keep exploring' continuation to drive the thinking loop.

        On the final step, return True to prevent the base agent from injecting
        a 'provide your final answer' synthesis nudge — on_response already
        captured the text, and after_run handles summarization via
        THINKING_REPORT_PROMPT.
        """
        if is_final:
            return True
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

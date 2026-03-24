"""ThinkingAgent — Penny's autonomous inner monologue.

Runs on the scheduler after extraction. Each cycle is a continuous
thinking loop where Penny searches for information using tools. At the
end, the raw search results are summarized and stored as a thought.
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

# Probability of a free-thinking cycle (no seed topic — Penny picks her own)
FREE_THINKING_PROBABILITY = 1 / 3

# Minimum word count for a thought to be stored (filters model planning text)
MIN_THOUGHT_WORDS = 50


class ThinkingAgent(Agent):
    """Autonomous inner monologue — Penny's conscious mind.

    Each cycle picks ONE random seed topic from history
    to focus on, keeping thinking rotating across interests.

    All modes get the same context: profile, recent thoughts
    (scoped by preference_id), and dislikes. The only difference
    is the initial prompt (seed topic vs free exploration).

    Thinking loop::

        [user]       Think about {seed topic}...
        [assistant]  <tool call: search(...)>
        [tool]       <search results>                <- captured via _tool_result_text
        [user]       dig deeper
        [assistant]  <tool call: search(...)>
        [tool]       <search results>                <- captured
        ...

    Summary step: raw search results summarized via THINKING_REPORT_PROMPT,
    stored as a thought in db.thoughts.

    Seed topic sources: positive user preferences.
    """

    THOUGHT_CONTEXT_LIMIT = 10

    name = "inner_monologue"

    def __init__(self, **kwargs: object) -> None:
        kwargs["system_prompt"] = Prompt.THINKING_SYSTEM_PROMPT
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.max_steps = int(self.config.runtime.INNER_MONOLOGUE_MAX_STEPS)
        self._keep_tools_on_final_step = True
        self._include_identity = False
        self._seed_topic: str | None = None
        self._seed_pref_id: int | None = None

    # ── Execution hooks ──────────────────────────────────────────────────

    async def get_prompt(self, user: str) -> str | None:
        """Pick a seed topic or let Penny free-think (~1/3 of the time)."""
        self._seed_topic = None
        self._seed_pref_id = None

        if random.random() < FREE_THINKING_PROBABILITY:
            logger.info("Free thinking cycle for %s", user)
            return Prompt.THINKING_FREE

        threshold = int(self.config.runtime.PREFERENCE_MENTION_THRESHOLD)
        pool = self.db.preferences.get_least_recent_positive(user, mention_threshold=threshold)
        if not pool:
            logger.info("No preferences for %s, browsing news", user)
            return Prompt.THINKING_BROWSE_NEWS

        pref = random.choice(pool)
        self._seed_topic = pref.content
        self._seed_pref_id = pref.id
        logger.info("Thinking seed: %s", pref.content)
        return Prompt.THINKING_SEED.format(seed=pref.content)

    async def get_context(self, user: str) -> str:
        """Slim context — thoughts and dislikes only.

        No identity or profile — thinking never communicates with the user.
        Seeded cycles get scoped thought context (what was explored for
        this preference). Free/news cycles get no thought context — injecting
        previous free thoughts just primes the model to revisit them.
        Embedding dedup catches true repeats at storage time.
        """
        sections: list[str | None] = [
            self._build_thought_context(user) if self._seed_pref_id is not None else None,
            self._build_dislike_context(user),
        ]
        return "\n\n".join(s for s in sections if s)

    def _build_thought_context(self, sender: str) -> str | None:
        """Build thought context scoped to the current preference_id.

        Shows what Penny already explored so she avoids repeating herself.
        Only used for seeded cycles (preference_id is not None).
        """
        try:
            thoughts = self.db.thoughts.get_recent_by_preference(
                sender, self._seed_pref_id, limit=self.THOUGHT_CONTEXT_LIMIT
            )
            if not thoughts:
                return None
            lines = [t.content for t in thoughts]
            logger.debug("Built thought context (%d thoughts)", len(thoughts))
            return "### Recent Background Thinking\n" + "\n\n---\n\n".join(lines)
        except Exception:
            logger.warning("Thought context retrieval failed, proceeding without")
            return None

    # Max retries for summary URL validation
    SUMMARY_URL_RETRIES = 2

    async def after_run(self, user: str) -> bool:
        """Summarize the search results, dedup against same-seed thoughts, and store."""
        if not self._tool_result_text:
            return False
        combined = "\n\n---\n\n".join(self._tool_result_text)
        report = await self._summarize_with_url_validation(combined)
        if report and len(report.split()) < MIN_THOUGHT_WORDS:
            logger.info(
                "[inner_monologue] report too short (%d words), skipping", len(report.split())
            )
            report = ""
        if report and not await self._is_duplicate_thought(user, report):
            self.db.thoughts.add(user, report, preference_id=self._seed_pref_id)
            logger.info(
                "[inner_monologue] stored thought (seed=%s): %s",
                self._seed_topic or "free",
                report[:200],
            )
        elif report:
            logger.info(
                "[inner_monologue] discarded duplicate (seed=%s): %s",
                self._seed_topic or "free",
                report[:100],
            )
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

    async def _summarize_with_url_validation(self, combined: str) -> str:
        """Summarize monologue, retrying if the report contains hallucinated URLs."""
        source_text = self._get_source_text()
        report = ""
        for attempt in range(1 + self.SUMMARY_URL_RETRIES):
            report = await self._summarize_text(combined, Prompt.THINKING_REPORT_PROMPT)
            if not report:
                return ""
            bad_urls = self._find_hallucinated_urls(report, source_text)
            if not bad_urls:
                return report
            logger.warning(
                "[inner_monologue] summary attempt %d/%d has %d hallucinated URL(s): %s",
                attempt + 1,
                1 + self.SUMMARY_URL_RETRIES,
                len(bad_urls),
                ", ".join(u[:80] for u in bad_urls),
            )
        logger.warning("[inner_monologue] exhausted URL validation retries, using last attempt")
        return report

    async def _is_duplicate_thought(self, user: str, report: str) -> bool:
        """Check if report is too similar to a same-scope thought via embedding similarity."""
        if not self._embedding_model_client:
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

    async def handle_text_step(
        self, response, messages: list[dict], step: int, is_final: bool
    ) -> bool:
        """Inject 'keep exploring' continuation to drive the thinking loop.

        On the final step, return True to prevent the base agent from
        treating the text as the final answer — after_run handles
        summarization of the raw search results via THINKING_REPORT_PROMPT.
        """
        if is_final:
            return True
        content = response.content.strip()
        messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=content).to_dict())
        nudge = "dig deeper into what you just found"
        messages.append(ChatMessage(role=MessageRole.USER, content=nudge).to_dict())
        return True

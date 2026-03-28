"""ThinkingAgent — Penny's autonomous inner monologue.

Runs on the scheduler after extraction. Each cycle is a continuous
thinking loop where Penny searches for information using tools. At the
end, the raw search results are summarized and stored as a thought.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Any

from similarity.dedup import DedupStrategy, is_embedding_duplicate

from penny.agents.base import Agent
from penny.agents.models import ChatMessage, MessageRole
from penny.constants import PennyConstants
from penny.ollama.embeddings import serialize_embedding
from penny.ollama.similarity import embed_text
from penny.prompts import Prompt

logger = logging.getLogger(__name__)


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

    THOUGHT_CONTEXT_LIMIT = PennyConstants.THOUGHT_CONTEXT_LIMIT

    name = "inner_monologue"

    def __init__(self, **kwargs: Any) -> None:
        kwargs["system_prompt"] = Prompt.THINKING_SYSTEM_PROMPT
        super().__init__(**kwargs)
        self._keep_tools_on_final_step = True
        self._seed_topic: str | None = None
        self._seed_pref_id: int | None = None

    def get_max_steps(self) -> int:
        """Read from config each cycle so /config changes take effect immediately."""
        return int(self.config.runtime.INNER_MONOLOGUE_MAX_STEPS)

    # ── Execution hooks ──────────────────────────────────────────────────

    async def execute_for_user(self, user: str) -> bool:
        """Check unnotified cap before running. Returns True to reset the schedule timer."""
        max_unnotified = int(self.config.runtime.MAX_UNNOTIFIED_THOUGHTS)
        total = self.db.thoughts.count_unnotified(user)
        if total >= max_unnotified:
            logger.info("Skipping thinking: %d unnotified thoughts (max %d)", total, max_unnotified)
            return True
        return await super().execute_for_user(user)

    async def get_prompt(self, user: str) -> str | None:
        """Pick thinking mode based on unnotified thought distribution.

        Compares the current free/seeded ratio against the target
        probabilities and picks whichever type is underrepresented.
        Falls back to random when there are no unnotified thoughts yet.
        """
        self._seed_topic = None
        self._seed_pref_id = None

        total = self.db.thoughts.count_unnotified(user)
        if self._should_think_free(user, total):
            return self._pick_free_prompt(user)
        return self._pick_seeded_prompt(user)

    def _should_think_free(self, user: str, total_unnotified: int) -> bool:
        """Decide free vs seeded based on distribution gap, random as tiebreak.

        Free and news both produce preference_id=NULL thoughts, so the
        target free ratio is FREE + NEWS combined.
        """
        free_prob = float(self.config.runtime.FREE_THINKING_PROBABILITY)
        news_prob = float(self.config.runtime.NEWS_THINKING_PROBABILITY)
        target_free = free_prob + news_prob
        if total_unnotified == 0:
            return random.random() < target_free
        free_count = self.db.thoughts.count_unnotified_free(user)
        actual_free_ratio = free_count / total_unnotified
        return actual_free_ratio < target_free

    def _pick_free_prompt(self, user: str) -> str:
        """Pick between free thinking and news based on their relative weights."""
        free_weight = float(self.config.runtime.FREE_THINKING_PROBABILITY)
        news_weight = float(self.config.runtime.NEWS_THINKING_PROBABILITY)
        total = free_weight + news_weight
        if total > 0 and random.random() < news_weight / total:
            logger.info("News thinking cycle for %s", user)
            return Prompt.THINKING_NEWS
        logger.info("Free thinking cycle for %s", user)
        return Prompt.THINKING_FREE

    def _pick_seeded_prompt(self, user: str) -> str | None:
        """Pick a preference-seeded prompt, falling back to news if none available."""
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

    async def _build_system_prompt(self, user: str) -> str:
        """No identity, no profile — just thoughts (if seeded) + dislikes + instructions.

        Free/news cycles get no thought context — injecting previous free
        thoughts primes the model to revisit them. Embedding dedup catches
        true repeats at storage time.
        """
        return "\n\n".join(
            s
            for s in [
                self._context_block(
                    self._thought_section(user) if self._seed_pref_id is not None else None,
                    self._dislike_section(user),
                ),
                self._instructions_section(),
            ]
            if s
        )

    def _thought_section(self, sender: str) -> str | None:
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
    SUMMARY_URL_RETRIES = PennyConstants.SUMMARY_URL_RETRIES

    async def after_run(self, user: str) -> bool:
        """Summarize the search results, dedup against same-seed thoughts, and store."""
        if not self._tool_result_text:
            return False
        combined = "\n\n---\n\n".join(self._tool_result_text)
        report = await self._summarize_with_url_validation(combined)
        if report and len(report.split()) < PennyConstants.MIN_THOUGHT_WORDS:
            logger.info(
                "[inner_monologue] report too short (%d words), skipping", len(report.split())
            )
            report = ""
        if report and not await self._is_duplicate_thought(user, report):
            title, content = self._parse_title(report)
            content_embedding = await self._embed_and_serialize(content)
            title_embedding = await self._embed_and_serialize(title.lower()) if title else None
            image_url = await self._search_thought_image(title) if title else None
            self.db.thoughts.add(
                user,
                content,
                preference_id=self._seed_pref_id,
                embedding=content_embedding,
                title=title,
                title_embedding=title_embedding,
                image_url=image_url,
            )
            logger.info(
                "[inner_monologue] stored thought (seed=%s, title=%s): %s",
                self._seed_topic or "free",
                title or "none",
                content[:200],
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

    async def _search_thought_image(self, title: str) -> str | None:
        """Search for an image URL to accompany a thought."""
        try:
            from penny.serper.client import search_image_url

            api_key = self.config.serper_api_key if self.config else None
            return await search_image_url(title, api_key=api_key, max_results=3, timeout=5.0)
        except Exception:
            return None

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

    async def _embed_and_serialize(self, text: str) -> bytes | None:
        """Embed text and serialize to bytes for storage."""
        vec = await embed_text(self._embedding_model_client, text)
        if vec is None:
            return None
        return serialize_embedding(vec)

    # Pattern to match "Topic: <title>" on the last line
    _TOPIC_LINE_PATTERN = re.compile(r"\n?Topic:\s*(.+?)\s*$")

    @classmethod
    def _parse_title(cls, report: str) -> tuple[str | None, str]:
        """Extract 'Topic: ...' from the last line, return (title, content)."""
        match = cls._TOPIC_LINE_PATTERN.search(report)
        if not match:
            return None, report
        title = match.group(1).strip()
        content = report[: match.start()].rstrip()
        return title, content

    async def _is_duplicate_thought(self, user: str, report: str) -> bool:
        """Check if report title duplicates any existing thought title."""
        title, _ = self._parse_title(report)
        if not title:
            return False
        title_vec = (
            await embed_text(self._embedding_model_client, title.lower())
            if self._embedding_model_client
            else None
        )
        existing_items: list[tuple[str, bytes | None]] = [
            (t.title, t.title_embedding) for t in self.db.thoughts.get_recent(user) if t.title
        ]
        match_idx = is_embedding_duplicate(
            title,
            title_vec,
            existing_items,
            DedupStrategy.TCR_AND_EMBEDDING,
            embedding_threshold=self.config.runtime.THOUGHT_DEDUP_EMBEDDING_THRESHOLD,
            tcr_threshold=self.config.runtime.THOUGHT_DEDUP_TCR_THRESHOLD,
        )
        if match_idx is not None:
            matched_title = existing_items[match_idx][0]
            logger.info(
                "[inner_monologue] duplicate title %r matches %r (pref #%s)",
                title,
                matched_title,
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

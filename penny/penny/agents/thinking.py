"""ThinkingAgent — Penny's autonomous inner monologue.

Runs on the scheduler after extraction. Each cycle is a continuous
thinking loop where Penny searches for information using tools. At the
end, the raw search results are summarized and stored as a thought.
"""

from __future__ import annotations

import logging
import random
import re
from datetime import UTC, datetime
from typing import Any

from similarity.dedup import DedupStrategy, is_embedding_duplicate
from similarity.embeddings import cosine_similarity, deserialize_embedding

from penny.agents.base import Agent
from penny.agents.models import ChatMessage, MessageRole
from penny.constants import PennyConstants
from penny.llm.embeddings import serialize_embedding
from penny.llm.similarity import embed_text
from penny.prompts import Prompt

logger = logging.getLogger(__name__)


class ThinkingPromptType:
    """Prompt types for ThinkingAgent flows."""

    FREE = "free"
    SEEDED = "seeded"


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
        self._seed_prompt: str | None = None

    def get_max_steps(self) -> int:
        """Read from config each cycle so /config changes take effect immediately."""
        return int(self.config.runtime.INNER_MONOLOGUE_MAX_STEPS)

    def get_prompt_type(self) -> str:
        """Seed topic or 'free', used as the prompt_type label in prompt logs."""
        if self._seed_topic is not None:
            return self._seed_topic.title()
        return ThinkingPromptType.FREE

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
        self._seed_prompt = None

        total = self.db.thoughts.count_unnotified(user)
        if self._should_think_free(user, total):
            self._seed_prompt = Prompt.THINKING_FREE
            return self._seed_prompt
        prompt = self._pick_seeded_prompt(user)
        self._seed_prompt = prompt
        return prompt

    def _should_think_free(self, user: str, total_unnotified: int) -> bool:
        """Decide free vs seeded based on distribution gap, random as tiebreak."""
        free_prob = float(self.config.runtime.FREE_THINKING_PROBABILITY)
        if total_unnotified == 0:
            return random.random() < free_prob
        free_count = self.db.thoughts.count_unnotified_free(user)
        actual_free_ratio = free_count / total_unnotified
        return actual_free_ratio < free_prob

    def _pick_seeded_prompt(self, user: str) -> str | None:
        """Pick a preference-seeded prompt, falling back to free if none available."""
        threshold = int(self.config.runtime.PREFERENCE_MENTION_THRESHOLD)
        pool = self.db.preferences.get_least_recent_positive(user, mention_threshold=threshold)
        if not pool:
            logger.info("No preferences for %s, free thinking", user)
            return Prompt.THINKING_FREE

        pref = random.choice(pool)
        self._seed_topic = pref.content
        self._seed_pref_id = pref.id
        logger.info("Thinking seed: %s", pref.content)
        return Prompt.THINKING_SEED.format(seed=pref.content)

    async def _build_system_prompt(self, user: str) -> str:
        """No identity, no profile — just thoughts (if seeded) + dislikes + instructions.

        Free cycles get no thought context — injecting previous free
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

        Lists titles of what Penny already explored so she avoids repeating.
        Only titles — full bodies prime the model to re-search the same things.
        Only used for seeded cycles (preference_id is not None).
        """
        try:
            thoughts = self.db.thoughts.get_recent_by_preference(
                sender, self._seed_pref_id, limit=self.THOUGHT_CONTEXT_LIMIT
            )
            if not thoughts:
                return None
            titles = [t.title for t in thoughts if t.title]
            if not titles:
                return None
            logger.debug("Built thought context (%d titles)", len(titles))
            lines = [f"- {title}" for title in titles]
            return "### Already Explored (do NOT repeat)\n" + "\n".join(lines)
        except Exception:
            logger.warning("Thought context retrieval failed, proceeding without")
            return None

    # Max retries for summary URL validation
    SUMMARY_URL_RETRIES = PennyConstants.SUMMARY_URL_RETRIES

    async def after_run(self, user: str, run_id: str, prompt_type: str | None = None) -> bool:
        """Summarize the search results, dedup, and store."""
        if not self._tool_result_text:
            self._record_outcome(run_id, "Discard: no search results")
            return False
        combined = self._filter_page_reads()
        if not combined:
            self._record_outcome(run_id, "Discard: no page reads")
            return False
        if not self._tool_result_images:
            self._record_outcome(run_id, "Discard: no image")
            return False
        report = await self._summarize_with_url_validation(combined, user, run_id, prompt_type)
        if not report:
            self._record_outcome(run_id, "Discard: no thought generated")
        elif len(report.split()) < PennyConstants.MIN_THOUGHT_WORDS:
            logger.info(
                "[inner_monologue] report too short (%d words), skipping", len(report.split())
            )
            self._record_outcome(run_id, f"Discard: too short ({len(report.split())} words)")
        else:
            title, content = self._parse_title(report)
            content_vec = await embed_text(self._embedding_model_client, content)
            duplicate_match = self._find_duplicate_thought(user, title, content_vec)
            if duplicate_match:
                logger.info(
                    "[inner_monologue] discarded duplicate (seed=%s): %s",
                    self._seed_topic or "free",
                    report[:100],
                )
                self._record_outcome(run_id, f"Discard: duplicate of {duplicate_match!r}")
            else:
                await self._store_thought(user, run_id, title, content, content_vec)
        if self._seed_pref_id is not None:
            self.db.preferences.mark_thought_about(self._seed_pref_id)
        return True

    def _record_outcome(self, run_id: str, outcome: str) -> None:
        """Record the outcome of a thinking run on its last prompt log."""
        self.db.messages.set_run_outcome(run_id, outcome)

    async def _store_thought(
        self,
        user: str,
        run_id: str,
        title: str | None,
        content: str,
        content_vec: list[float] | None,
    ) -> None:
        """Embed title, check dislike filter, and persist thought."""
        content_embedding = serialize_embedding(content_vec) if content_vec else None
        title_vec = await embed_text(self._embedding_model_client, title.lower()) if title else None
        title_embedding = serialize_embedding(title_vec) if title_vec else None
        matched_dislike = self._matches_dislike(user, title_vec)
        if matched_dislike:
            logger.info(
                "[inner_monologue] filtered by dislike %r (seed=%s): %s",
                matched_dislike,
                self._seed_topic or "free",
                content[:100],
            )
            self._record_outcome(run_id, f"Discard: matches dislike {matched_dislike!r}")
            return
        image = self._tool_result_images[0] if self._tool_result_images else None
        self.db.thoughts.add(
            user,
            content,
            preference_id=self._seed_pref_id,
            embedding=content_embedding,
            title=title,
            title_embedding=title_embedding,
            image=image,
            run_id=run_id,
        )
        self._record_outcome(run_id, f"Stored: {title or 'untitled'}")
        logger.info(
            "[inner_monologue] stored thought (seed=%s, title=%s): %s",
            self._seed_topic or "free",
            title or "none",
            content[:200],
        )

    def _matches_dislike(self, user: str, vec: list[float] | None) -> str | None:
        """Return the dislike content if the thought title is too similar to any dislike.

        Compares the thought's title embedding against each negative preference.
        Short titles and short dislike labels produce strong similarity signals,
        avoiding the dilution that full-content embeddings suffer from.
        Returns the matched dislike content, or None if no match.
        """
        if vec is None:
            return None
        threshold = PennyConstants.DISLIKE_FILTER_THRESHOLD
        dislikes = self.db.preferences.get_negative_with_embeddings(user)
        for pref in dislikes:
            if not pref.embedding:
                continue
            similarity = cosine_similarity(vec, deserialize_embedding(pref.embedding))
            if similarity >= threshold:
                logger.debug(
                    "[inner_monologue] dislike match: %.3f for %r", similarity, pref.content
                )
                return pref.content
        return None

    def _filter_page_reads(self) -> str:
        """Extract only page-read sections from tool results.

        Only includes sections starting with BROWSE_PAGE_HEADER — these are
        actual page content. Search snippets and other sections are excluded.
        Returns empty string if no page reads were captured.
        """
        separator = PennyConstants.SECTION_SEPARATOR
        page_sections: list[str] = []
        for tool_result in self._tool_result_text:
            for section in tool_result.split(separator):
                if section.startswith(PennyConstants.BROWSE_PAGE_HEADER):
                    page_sections.append(section)
        return separator.join(page_sections)

    # ── Model calls ────────────────────────────────────────────────────────

    async def _summarize_text(
        self, content: str, prompt: str, run_id: str, prompt_type: str | None = None
    ) -> str:
        """Summarize content using the model. Returns empty string on failure."""
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ]
        try:
            response = await self._model_client.chat(
                messages=messages, agent_name=self.name, prompt_type=prompt_type, run_id=run_id
            )
            return response.content.strip()
        except Exception as e:
            logger.error("Summarization failed: %s", e)
            return ""

    def _build_report_prompt(self, user: str) -> str:
        """Build the summarization system prompt with identity, profile, and seed context.

        The report is user-facing content, so it gets the same identity and
        profile context as other user-facing prompts (chat, notify).
        """
        now = datetime.now(UTC).strftime("%A, %B %d, %Y at %I:%M %p UTC")
        parts = [
            f"Current date and time: {now}",
            self._identity_section(),
            self._context_block(self._profile_section(user)),
            f"## Instructions\n{Prompt.THINKING_REPORT_PROMPT}",
        ]
        if self._seed_prompt:
            parts.append(f"Original research goal:\n{self._seed_prompt}")
        return "\n\n".join(s for s in parts if s)

    async def _summarize_with_url_validation(
        self, combined: str, user: str, run_id: str, prompt_type: str | None = None
    ) -> str:
        """Summarize monologue, retrying on empty, missing Topic:, or hallucinated URLs."""
        source_text = self._get_source_text()
        report_prompt = self._build_report_prompt(user)
        report = ""
        for attempt in range(1 + self.SUMMARY_URL_RETRIES):
            label = (
                f"[inner_monologue] summary attempt {attempt + 1}/{1 + self.SUMMARY_URL_RETRIES}"
            )
            report = await self._summarize_text(combined, report_prompt, run_id, prompt_type)
            if not report:
                logger.warning("%s returned empty, retrying", label)
                continue
            title, _ = self._parse_title(report)
            if not title:
                logger.warning("%s missing Topic: line, retrying", label)
                continue
            bad_urls = self._find_hallucinated_urls(report, source_text)
            if not bad_urls:
                return report
            logger.warning(
                "%s has %d hallucinated URL(s): %s",
                label,
                len(bad_urls),
                ", ".join(u[:80] for u in bad_urls),
            )
        if report:
            logger.warning("[inner_monologue] exhausted summary retries, using last attempt")
        return report

    async def _embed_and_serialize(self, text: str) -> bytes | None:
        """Embed text and serialize to bytes for storage."""
        vec = await embed_text(self._embedding_model_client, text)
        if vec is None:
            return None
        return serialize_embedding(vec)

    # Pattern to match "Topic: <title>" anywhere in the text
    _TOPIC_LINE_PATTERN = re.compile(r"^Topic:\s*(.+?)\s*$", re.MULTILINE)

    # How many lines from the end to search for the Topic: line
    TOPIC_SEARCH_LINES = 5

    @classmethod
    def _parse_title(cls, report: str) -> tuple[str | None, str]:
        """Extract 'Topic: ...' from the last few lines, return (title, content).

        The model sometimes puts sources or emoji after the Topic: line,
        so we search the last TOPIC_SEARCH_LINES lines instead of
        requiring it at the very end.
        """
        lines = report.rstrip().split("\n")
        tail = "\n".join(lines[-cls.TOPIC_SEARCH_LINES :])
        match = cls._TOPIC_LINE_PATTERN.search(tail)
        if not match:
            return None, report
        title = match.group(1).strip()
        # Remove everything from the Topic: line onwards
        topic_line_start = report.rindex(f"Topic: {match.group(1).rstrip()}")
        content = report[:topic_line_start].rstrip()
        return title, content

    def _find_duplicate_thought(
        self, user: str, title: str | None, content_vec: list[float] | None
    ) -> str | None:
        """Check new thought against all existing thoughts.

        Uses title TCR OR content embedding similarity — either signal
        triggers a match.  Scans all past thoughts (O(N), N is small).
        Falls back to EMBEDDING_ONLY when the candidate has no title.
        Returns the matched thought's title, or None if no match.
        """
        if not title and content_vec is None:
            return None
        all_thoughts = self.db.thoughts.get_all(user)
        existing_items: list[tuple[str, bytes | None]] = [
            (t.title or "", t.embedding) for t in all_thoughts
        ]
        if not existing_items:
            return None
        strategy = DedupStrategy.TCR_OR_EMBEDDING if title else DedupStrategy.EMBEDDING_ONLY
        match_idx = is_embedding_duplicate(
            title or "",
            content_vec,
            existing_items,
            strategy,
            embedding_threshold=self.config.runtime.THOUGHT_DEDUP_EMBEDDING_THRESHOLD,
            tcr_threshold=self.config.runtime.THOUGHT_DEDUP_TCR_THRESHOLD,
        )
        if match_idx is not None:
            matched_title = existing_items[match_idx][0]
            logger.info(
                "[inner_monologue] duplicate %r matches %r (pref #%s)",
                title,
                matched_title,
                self._seed_pref_id,
            )
            return matched_title
        return None

    # ── Loop hooks ─────────────────────────────────────────────────────────

    async def after_step(
        self,
        step_records: list,
        step_messages: list[dict],
        conversation: list[dict] | None = None,
    ) -> None:
        """After tool execution, nudge model to browse if it only searched."""
        await super().after_step(step_records, step_messages, conversation)
        if conversation is None:
            return
        last_tool = next(
            (m for m in reversed(step_messages) if m.get("role") == MessageRole.TOOL),
            None,
        )
        if last_tool is None:
            return
        content = last_tool.get("content", "")
        has_page_read = PennyConstants.BROWSE_PAGE_HEADER in content
        has_search = PennyConstants.BROWSE_SEARCH_HEADER in content
        if has_search and not has_page_read:
            conversation.append(
                ChatMessage(
                    role=MessageRole.USER,
                    content=Prompt.BROWSE_NUDGE,
                ).to_dict()
            )

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

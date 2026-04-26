"""HistoryAgent — preference extraction and knowledge extraction.

Runs on a schedule. Each cycle:
1. Extracts knowledge from browse tool results (user-independent)
2. Per user: runs the preference-extractor agent loop, which reads new
   user messages via ``log_read_next``, identifies likes/dislikes, and
   writes them via ``collection_write``.

Preference dedup, mention counting, and reaction handling all moved out
of Python into the tool layer (dedup-on-write) or were dropped entirely
in the port to the memory framework.  Knowledge extraction stays
bespoke for now (Stage 7 territory).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit

from penny.agents.base import Agent
from penny.agents.models import ControllerResponse
from penny.constants import HistoryPromptType, PennyConstants
from penny.database.models import PromptLog
from penny.llm.embeddings import serialize_embedding
from penny.prompts import Prompt
from penny.tools.memory_tools import DoneTool, LogReadNextTool

logger = logging.getLogger(__name__)


class HistoryAgent(Agent):
    """Background worker that extracts knowledge and user preferences."""

    name = "history"

    # Identity for the preference-extraction flow — used both as the
    # cursor key for ``user-messages`` and as the author stamped on
    # writes to ``likes``/``dislikes``.  Distinct from ``name`` because
    # the same HistoryAgent class may host multiple flows (knowledge,
    # reactions) with their own per-flow cursors and attribution.
    PREFERENCE_EXTRACTOR_NAME = "preference-extractor"

    # Cap on agentic loop iterations for the preference extractor.  The
    # expected flow is read_next → write(likes) → write(dislikes) → done,
    # so 8 leaves headroom for re-reads or batched writes without
    # letting a runaway loop tail through forever.
    PREFERENCE_EXTRACTOR_MAX_STEPS = 8

    def get_max_steps(self) -> int:
        """Cap on agentic loop iterations for HistoryAgent flows."""
        return self.PREFERENCE_EXTRACTOR_MAX_STEPS

    async def execute(self) -> bool:
        """Extract knowledge (user-independent), then run per-user work."""
        knowledge_work = await self._extract_knowledge()
        user_work = await super().execute()
        return knowledge_work or user_work

    async def execute_for_user(self, user: str) -> bool:
        """Run the preference-extractor agent loop for the user."""
        return await self._run_preference_extractor()

    # ── Knowledge extraction ──────────────────────────────────────────────

    async def _extract_knowledge(self) -> bool:
        """Scan prompt logs for browse results and summarize into knowledge entries.

        Within a batch the same URL often appears in many prompts because each
        step of an agentic loop re-logs the prior tool result messages. Dedup by
        URL keeping the latest occurrence so each page is summarized at most once
        per batch instead of re-aggregating identical content N times.
        """
        watermark = self.db.knowledge.get_latest_prompt_timestamp() or datetime.min
        batch_limit = int(self.config.runtime.KNOWLEDGE_EXTRACTION_BATCH_LIMIT)
        prompts = self.db.messages.get_prompts_with_browse_after(watermark, batch_limit)
        if not prompts:
            return False

        unique_by_url = self._dedup_browse_results_by_url(prompts)
        if not unique_by_url:
            return False

        run_id = uuid.uuid4().hex
        for url, (title, content, prompt_id) in unique_by_url.items():
            await self._summarize_knowledge(url, title, content, prompt_id, run_id)
        return True

    @staticmethod
    def _dedup_browse_results_by_url(
        prompts: list[PromptLog],
    ) -> dict[str, tuple[str, str, int]]:
        """Collapse browse results across the batch to one entry per URL.

        URLs are normalized (fragment stripped, host lowercased) before keying
        so `/page` and `/page#anchor` collapse to a single entry. Iterates
        prompts in order; later occurrences overwrite earlier ones so the
        freshest content for each URL wins. Returns {url: (title, content,
        prompt_id)} keyed by the normalized URL.
        """
        unique: dict[str, tuple[str, str, int]] = {}
        for prompt in prompts:
            if prompt.id is None:
                continue
            for url, title, content in HistoryAgent._parse_browse_results(prompt):
                unique[HistoryAgent._normalize_url(url)] = (title, content, prompt.id)
        return unique

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Canonicalize a URL for dedup and storage.

        Strips the `#fragment` (client-side anchor, never affects page content)
        and lowercases the scheme and host (case-insensitive per RFC 3986).
        Path, query, and userinfo are preserved as-is — they can be
        case-sensitive on the server side. URLs that fail to parse are
        returned unchanged so a malformed string still keys consistently.
        """
        try:
            parts = urlsplit(url)
        except ValueError:
            return url
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))

    @staticmethod
    def _parse_browse_results(prompt: PromptLog) -> list[tuple[str, str, str]]:
        """Extract (url, title, page_content) tuples from browse tool results."""
        results: list[tuple[str, str, str]] = []
        for message in prompt.get_messages():
            if message.get("role") != "tool":
                continue
            content = message.get("content", "")
            for section in content.split(PennyConstants.SECTION_SEPARATOR):
                if section.startswith(PennyConstants.BROWSE_PAGE_HEADER):
                    parsed = HistoryAgent._parse_browse_section(section)
                    if parsed:
                        results.append(parsed)
        return results

    @staticmethod
    def _parse_browse_section(section: str) -> tuple[str, str, str] | None:
        """Parse a successful browse section into (url, title, page_content).

        Only matches the healthy format: header + Title: + URL: + content.
        Error responses (disconnects, timeouts, blocked domains) are skipped.
        """
        lines = section.split("\n", 3)
        if len(lines) < 3:
            return None
        url = lines[0][len(PennyConstants.BROWSE_PAGE_HEADER) :].strip()
        if not url:
            return None
        if not lines[1].startswith(PennyConstants.BROWSE_TITLE_PREFIX):
            return None
        if not lines[2].startswith(PennyConstants.BROWSE_URL_PREFIX):
            return None
        title = lines[1][len(PennyConstants.BROWSE_TITLE_PREFIX) :]
        page_content = lines[3] if len(lines) > 3 else ""
        if not page_content.strip():
            return None
        return (url, title, page_content)

    async def _summarize_knowledge(
        self, url: str, title: str, content: str, prompt_id: int, run_id: str
    ) -> None:
        """Summarize page content and upsert into knowledge store."""
        existing = self.db.knowledge.get_by_url(url)
        if existing:
            summary = await self._aggregate_knowledge(existing.summary, content, run_id)
        else:
            summary = await self._summarize_page(content, run_id)
        if not summary:
            return
        embedding = await self._embed_text(summary)
        self.db.knowledge.upsert_by_url(url, title, summary, embedding, prompt_id)

    async def _summarize_page(self, content: str, run_id: str) -> str | None:
        """Summarize a single page via LLM."""
        messages = [
            {"role": "system", "content": Prompt.KNOWLEDGE_SUMMARIZE},
            {"role": "user", "content": content},
        ]
        response = await self._model_client.chat(
            messages,
            agent_name=self.name,
            prompt_type=HistoryPromptType.KNOWLEDGE_SUMMARIZE,
            run_id=run_id,
        )
        return response.content.strip() if response.content else None

    async def _aggregate_knowledge(
        self, existing_summary: str, new_content: str, run_id: str
    ) -> str | None:
        """Merge existing summary with new page content via LLM."""
        user_content = f"Existing summary:\n{existing_summary}\n\nNew content:\n{new_content}"
        messages = [
            {"role": "system", "content": Prompt.KNOWLEDGE_AGGREGATE},
            {"role": "user", "content": user_content},
        ]
        response = await self._model_client.chat(
            messages,
            agent_name=self.name,
            prompt_type=HistoryPromptType.KNOWLEDGE_SUMMARIZE,
            run_id=run_id,
        )
        return response.content.strip() if response.content else None

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

        Done is the only graceful terminator for this background flow.
        Hitting max_steps, raising a model error, or producing fallback
        text all leave the cursor untouched so the next run can retry.
        """
        return any(record.tool == DoneTool.name for record in response.tool_calls)

    # ── Shared helpers ────────────────────────────────────────────────────

    async def _embed_text(self, text: str) -> bytes | None:
        """Compute and serialize embedding for a text string."""
        if not self._embedding_model_client:
            return None
        try:
            vecs = await self._embedding_model_client.embed(text)
            return serialize_embedding(vecs[0])
        except Exception as e:
            logger.warning("Failed to embed text: %s", e)
            return None

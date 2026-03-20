"""NotifyAgent — Penny's notification outreach.

Sends notifications to users when idle: thought candidates,
news updates, and periodic check-ins. Runs on a schedule via the
BackgroundScheduler.
"""

from __future__ import annotations

import logging
import random
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from penny.agents.base import Agent
from penny.agents.models import ToolCallRecord
from penny.constants import PennyConstants
from penny.database.models import Thought
from penny.ollama.similarity import (
    compute_sentiment_score,
    embed_text,
    load_preference_vectors,
    novelty_score,
)
from penny.prompts import Prompt
from penny.responses import PennyResponse

if TYPE_CHECKING:
    from penny.channels import MessageChannel

logger = logging.getLogger(__name__)


class NotifyCandidate(BaseModel):
    """A candidate notification message awaiting scoring."""

    answer: str
    thought: Thought | None = None
    attachments: list[str] = Field(default_factory=list)
    image_prompt: str | None = None


class NotifyAgent(Agent):
    """Notification outreach agent — sends thoughts, news, and check-ins.

    Context matrix — each mode gets tailored context:

        Mode     | Entities | History | Thought    | Turns | Tools | Steps
        -------- | -------- | ------- | ---------- | ----- | ----- | -----
        Checkin  | -        | 7d      | 1 notified | yes   | none  | 1
        News     | -        | 7d      | -          | -     | all   | 5
        Thought  | thought  | 7d      | the thought| -     | all   | 5

    All modes include profile (user name).
    """

    name: str = "notify"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._boot_time = datetime.now(UTC).replace(tzinfo=None)
        self._channel: MessageChannel | None = None
        self._pending_thought: Thought | None = None

    def set_channel(self, channel: MessageChannel) -> None:
        """Set the channel for sending notifications."""
        self._channel = channel

    # ── Scheduled entry point ─────────────────────────────────────────

    async def execute_for_user(self, user: str) -> bool:
        """Scheduled cycle: send a notification if the user has been idle."""
        if not self._should_notify(user):
            return False
        return await self._send_notification(user)

    def _should_notify(self, user: str) -> bool:
        """Python-space eligibility checks for notifications."""
        if not self._channel:
            return False
        if self.db.users.is_muted(user):
            return False
        if not self._has_recent_thoughts(user):
            return False
        return self._cooldown_elapsed(user)

    # ── Cooldown ──────────────────────────────────────────────────────

    def _has_recent_thoughts(self, user: str) -> bool:
        """Check if user has un-notified thoughts within freshness window."""
        hours = int(self.config.runtime.THOUGHT_FRESHNESS_HOURS)
        return self.db.thoughts.get_next_unnotified(user, freshness_hours=hours) is not None

    def _cooldown_elapsed(self, user: str) -> bool:
        """Check if enough time since last autonomous outgoing message.

        Uses exponential backoff: cooldown doubles with each consecutive
        autonomous message since the user's last incoming message.
        """
        latest = self.db.messages.get_latest_autonomous_outgoing_time(user)
        if latest is None:
            return True
        elapsed = (datetime.now(UTC).replace(tzinfo=None) - latest).total_seconds()
        count = self.db.messages.count_autonomous_since_last_incoming(user, self._boot_time)
        cooldown = min(
            self.config.runtime.NOTIFY_COOLDOWN_MIN * (2 ** max(count - 1, 0)),
            self.config.runtime.NOTIFY_COOLDOWN_MAX,
        )
        return elapsed >= cooldown

    # ── Check-in gating ───────────────────────────────────────────────

    # Check-in requires user activity within this window (seconds)
    CHECKIN_ACTIVE_WINDOW = 1800  # 30 minutes

    def _should_checkin(self, user: str) -> bool:
        """Check-in if 24h cooldown elapsed AND user was active in last 30m."""
        return self._checkin_cooldown_elapsed() and self._user_recently_active(user)

    def _checkin_cooldown_elapsed(self) -> bool:
        """At most one check-in per 24 hours (rolling window)."""
        last = self.db.messages.get_last_checkin_time(Prompt.NOTIFY_CHECKIN, hours=24)
        if last is None:
            return True
        elapsed = (datetime.now(UTC).replace(tzinfo=None) - last).total_seconds()
        return elapsed >= 86400

    def _user_recently_active(self, user: str) -> bool:
        """User sent a message within the active window."""
        last = self.db.messages.get_latest_incoming_time(user)
        if last is None:
            return False
        elapsed = (datetime.now(UTC).replace(tzinfo=None) - last).total_seconds()
        return elapsed <= self.CHECKIN_ACTIVE_WINDOW

    # ── Notification pipeline ─────────────────────────────────────────

    # 1-in-3 chance of sending news instead of thought candidates
    NEWS_CHANCE = 1 / 3

    def _news_cooldown_elapsed(self) -> bool:
        """Check if enough time has passed since the last news notification."""
        last = self.db.messages.get_last_checkin_time(Prompt.NOTIFY_NEWS, hours=48)
        if last is None:
            return True
        elapsed = (datetime.now(UTC).replace(tzinfo=None) - last).total_seconds()
        return elapsed >= self.config.runtime.NEWS_COOLDOWN

    async def _send_notification(self, user: str) -> bool:
        """Check-in if eligible, then coin-flip news, otherwise thought candidates."""
        assert self._channel is not None
        try:
            if self._should_checkin(user):
                return await self._send_checkin(user)
            if random.random() < self.NEWS_CHANCE and self._news_cooldown_elapsed():
                return await self._send_news(user)
            return await self._send_best_candidate(user)
        except Exception:
            logger.exception("Failed to send notification to %s", user)
            return False
        finally:
            self._pending_thought = None

    async def _send_checkin(self, user: str) -> bool:
        """Send a check-in message — slim context, no tools, single step."""
        logger.info("Notify check-in for %s", user)
        self._pending_thought = None
        context = self._build_checkin_context(user)
        self._install_tools([])
        response = await self.run(
            prompt=Prompt.NOTIFY_CHECKIN,
            history=self._build_conversation(user),
            context=context,
            max_steps=1,
        )
        answer = response.answer.strip() if response.answer else None
        if not answer:
            return False
        return await self._send_candidate(
            user, NotifyCandidate(answer=answer, attachments=response.attachments or [])
        )

    def _build_checkin_context(self, user: str) -> str:
        """Checkin context: profile + history rollups + last notified thought."""
        sections: list[str | None] = [
            self._build_profile_context(user, None),
            self._build_history_context(user),
            self._build_notified_thought_context(user),
        ]
        return "\n\n".join(s for s in sections if s)

    async def _send_news(self, user: str) -> bool:
        """Send a news message — profile + history only, tools enabled."""
        logger.info("Notify news for %s", user)
        context = self._build_news_context(user)
        self._install_tools(self.get_tools(user))
        response = await self.run(
            prompt=Prompt.NOTIFY_NEWS,
            context=context,
        )
        answer = response.answer.strip() if response.answer else None
        if not answer:
            return False
        image_prompt = self._extract_first_headline(answer)
        return await self._send_candidate(
            user,
            NotifyCandidate(
                answer=answer,
                attachments=response.attachments or [],
                image_prompt=image_prompt,
            ),
        )

    def _build_news_context(self, user: str) -> str:
        """News context: profile + history rollups. No entities, thoughts, or conv turns."""
        sections: list[str | None] = [
            self._build_profile_context(user, None),
            self._build_history_context(user),
        ]
        return "\n\n".join(s for s in sections if s)

    @staticmethod
    def _extract_search_query(tool_calls: list[ToolCallRecord]) -> str | None:
        """Extract the search query from tool calls for use as image prompt."""
        for tc in tool_calls:
            if tc.tool == "search" and tc.arguments.get("query"):
                return tc.arguments["query"]
        return None

    # Matches **bold text** in markdown (first occurrence)
    _BOLD_PATTERN = re.compile(r"\*\*(.+?)\*\*")

    @classmethod
    def _extract_first_headline(cls, text: str) -> str | None:
        """Extract the first bold headline from response text for image search."""
        match = cls._BOLD_PATTERN.search(text)
        return match.group(1) if match else None

    # ── Thought candidates ────────────────────────────────────────────

    async def _send_best_candidate(self, user: str) -> bool:
        """Generate thought candidates, score, send the best."""
        n = int(self.config.runtime.NOTIFY_CANDIDATES)
        candidates = await self._generate_thought_candidates(user, n)
        if not candidates:
            logger.warning("No viable notification candidates for %s", user)
            return False
        winner = await self._pick_best_candidate(user, candidates)
        return await self._send_candidate(user, winner)

    async def _generate_thought_candidates(self, user: str, n: int) -> list[NotifyCandidate]:
        """Generate N thought candidates ranked by preference affinity."""
        candidates: list[NotifyCandidate] = []
        thoughts = self._get_top_thoughts(user, n)
        for i, thought in enumerate(thoughts):
            candidate = await self._generate_one_candidate(
                user, Prompt.NOTIFY_PROMPT, thought=thought
            )
            if candidate:
                logger.info(
                    "Candidate thought %d/%d: %s", i + 1, len(thoughts), candidate.answer[:60]
                )
                candidates.append(candidate)
        return candidates

    def _get_top_thoughts(self, user: str, n: int) -> list[Thought]:
        """Select diverse unnotified thoughts: 1 free-thinking + N-1 least-recently-notified prefs.

        Selection algorithm:
        1. Most recent unnotified free-thinking thought (preference_id IS NULL)
        2. For each preference with unnotified thoughts, find when it was last
           notified. Pick the N-1 least-recently-notified preferences, and from
           each take the most recent unnotified thought.
        """
        hours = int(self.config.runtime.THOUGHT_FRESHNESS_HOURS)
        all_unnotified = self.db.thoughts.get_all_unnotified(user, freshness_hours=hours)
        if not all_unnotified:
            return []

        result: list[Thought] = []

        # 1. Most recent free-thinking thought
        free = [t for t in all_unnotified if t.preference_id is None]
        if free:
            result.append(free[-1])

        # 2. Group seeded thoughts by preference, find last notified time per pref
        seeded = [t for t in all_unnotified if t.preference_id is not None]
        by_pref: dict[int, list[Thought]] = {}
        for t in seeded:
            assert t.preference_id is not None
            by_pref.setdefault(t.preference_id, []).append(t)

        pref_last_notified = self._get_pref_last_notified_times(user, list(by_pref.keys()))
        ranked = sorted(by_pref.keys(), key=lambda pid: pref_last_notified.get(pid) or "")
        slots = n - len(result)
        for pref_id in ranked[:slots]:
            result.append(by_pref[pref_id][-1])  # most recent unnotified

        return result

    def _get_pref_last_notified_times(self, user: str, preference_ids: list[int]) -> dict[int, str]:
        """Get the most recent notified_at per preference (for ranking)."""
        if not preference_ids:
            return {}
        all_thoughts = self.db.thoughts.get_recent(user)
        last_notified: dict[int, str] = {}
        for t in all_thoughts:
            if t.preference_id in preference_ids and t.notified_at is not None:
                ts = str(t.notified_at)
                if t.preference_id not in last_notified or ts > last_notified[t.preference_id]:
                    last_notified[t.preference_id] = ts
        return last_notified

    async def _generate_one_candidate(
        self, user: str, prompt: str, thought: Thought | None
    ) -> NotifyCandidate | None:
        """Generate a single notification candidate via the agentic loop.

        Uses thought-specific context (profile + thought) without
        conversation turns, so the model focuses on the thought
        rather than continuing the conversation.
        """
        self._pending_thought = thought
        context = self._build_thought_candidate_context(user)
        self._install_tools(self.get_tools(user))
        response = await self.run(prompt=prompt, context=context)
        self._pending_thought = None
        answer = response.answer.strip() if response.answer else None
        if not answer:
            return None
        if self._is_disqualified(answer):
            logger.info("Disqualified candidate: %s", answer[:60])
            return None
        image_prompt = self._extract_search_query(response.tool_calls)
        return NotifyCandidate(
            answer=answer,
            thought=thought,
            attachments=response.attachments or [],
            image_prompt=image_prompt,
        )

    @classmethod
    def _is_disqualified(cls, answer: str) -> bool:
        """Check if a candidate is an error fallback or model refusal."""
        error_strings = (
            PennyResponse.AGENT_MAX_STEPS,
            PennyResponse.AGENT_MODEL_ERROR,
            PennyResponse.AGENT_EMPTY_RESPONSE,
            PennyResponse.FALLBACK_RESPONSE,
        )
        if answer in error_strings:
            return True
        return cls._is_refusal(answer)

    def _build_thought_candidate_context(self, user: str) -> str:
        """Thought candidate context: profile + thought only. No history or conv turns."""
        sections: list[str | None] = [
            self._build_profile_context(user, None),
            self._build_pending_thought_context(),
        ]
        return "\n\n".join(s for s in sections if s)

    def _build_pending_thought_context(self) -> str | None:
        """Build context for the specific thought being shared."""
        if self._pending_thought is not None:
            return f"## Your Latest Thought\n{self._pending_thought.content}"
        return None

    def _build_notified_thought_context(self, user: str) -> str | None:
        """Build context from recently notified thoughts (already shared with user)."""
        hours = int(self.config.runtime.THOUGHT_FRESHNESS_HOURS)
        thoughts = self.db.thoughts.get_recent_notified(user, freshness_hours=hours, limit=1)
        if not thoughts:
            return None
        return f"## Recent Background Thinking\n{thoughts[0].content}"

    # ── Candidate scoring ─────────────────────────────────────────────

    async def _pick_best_candidate(
        self, user: str, candidates: list[NotifyCandidate]
    ) -> NotifyCandidate:
        """Score candidates on novelty + sentiment and return the best."""
        if len(candidates) == 1 or not self._embedding_model_client:
            return candidates[0]

        recent_vecs = await self._embed_recent_messages(user)
        likes, dislikes = load_preference_vectors(
            self.db.preferences.get_with_embeddings(user),
            PennyConstants.PreferenceValence.POSITIVE,
            PennyConstants.PreferenceValence.NEGATIVE,
        )
        best: NotifyCandidate | None = None
        best_score = float("-inf")

        for candidate in candidates:
            vec = await embed_text(self._embedding_model_client, candidate.answer)
            if vec is None:
                continue
            novelty = novelty_score(vec, recent_vecs)
            sentiment = compute_sentiment_score(vec, likes, dislikes)
            score = 0.5 * novelty + 0.5 * sentiment
            logger.info(
                "Candidate score: %.3f (novelty=%.3f, sentiment=%.3f) %s",
                score,
                novelty,
                sentiment,
                candidate.answer[:60],
            )
            if score > best_score:
                best_score = score
                best = candidate

        return best if best is not None else candidates[0]

    async def _embed_recent_messages(self, user: str) -> list[list[float]]:
        """Embed recent outgoing messages for novelty comparison."""
        if not self._embedding_model_client:
            return []
        hours = int(self.config.runtime.THOUGHT_FRESHNESS_HOURS)
        contents = self.db.messages.get_recent_outgoing_content(user, hours=hours)
        vecs: list[list[float]] = []
        for content in contents:
            vec = await embed_text(self._embedding_model_client, content)
            if vec is not None:
                vecs.append(vec)
        return vecs

    # ── Send ──────────────────────────────────────────────────────────

    async def _send_candidate(self, user: str, candidate: NotifyCandidate) -> bool:
        """Send the winning candidate and mark its thought as notified."""
        assert self._channel is not None
        thought_id = candidate.thought.id if candidate.thought else None
        await self._channel.send_response(
            user,
            candidate.answer,
            parent_id=None,
            attachments=candidate.attachments or None,
            quote_message=None,
            image_prompt=candidate.image_prompt,
            thought_id=thought_id,
        )
        if candidate.thought and candidate.thought.id is not None:
            self.db.thoughts.mark_notified(candidate.thought.id)
        logger.info("Notification sent to %s", user)
        return True

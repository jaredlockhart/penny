"""NotifyAgent — Penny's notification outreach.

Sends notifications to users when idle: thought candidates,
news updates, and periodic check-ins. Runs on a schedule via the
BackgroundScheduler.

Each notification mode (checkin, news, thought) is a NotificationMode
subclass that declares its tools, prompt, context, and image extraction.
NotifyAgent orchestrates the shared pipeline: install tools, build prompt,
run model, validate, extract image, send.
"""

from __future__ import annotations

import logging
import random
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from penny.agents.base import Agent
from penny.agents.models import ControllerResponse, ToolCallRecord
from penny.constants import PennyConstants
from penny.database.models import Thought
from penny.ollama.embeddings import deserialize_embedding
from penny.ollama.similarity import (
    compute_sentiment_score,
    load_preference_vectors,
    novelty_score,
)
from penny.prompts import Prompt
from penny.responses import PennyResponse
from penny.tools.models import SearchArgs

if TYPE_CHECKING:
    from penny.channels import MessageChannel
    from penny.config import Config
    from penny.tools import Tool

logger = logging.getLogger(__name__)


class NotifyCandidate(BaseModel):
    """A candidate notification message awaiting scoring."""

    answer: str
    thought: Thought | None = None
    attachments: list[str] = Field(default_factory=list)
    image_prompt: str


# ── Notification modes ─────────────────────────────────────────────────


class NotificationMode:
    """Declares what varies per notification mode.

    Each mode specifies its tools, system prompt, user prompt, run
    parameters, and image-prompt extraction.  The agent orchestrates
    the shared pipeline around these declarations.

    Subclasses must override: get_tools, build_system_prompt, prompt,
    and extract_image_prompt.  Other methods have sensible defaults.
    """

    def get_tools(self, agent: NotifyAgent, user: str) -> list[Tool]:
        """Tools available during generation."""
        raise NotImplementedError

    def build_system_prompt(self, agent: NotifyAgent, user: str) -> str:
        """Full system prompt for this mode."""
        raise NotImplementedError

    @property
    def prompt(self) -> str:
        """User prompt sent to the model."""
        raise NotImplementedError

    def get_max_steps(self) -> int:
        """Max agentic loop steps. Subclasses must override."""
        raise NotImplementedError

    def get_history(self, agent: NotifyAgent, user: str) -> list[tuple[str, str]] | None:
        """Conversation turns to include (None = no turns)."""
        return None

    @property
    def validate_urls(self) -> bool:
        """Whether to retry on hallucinated URLs."""
        return False

    @property
    def check_disqualified(self) -> bool:
        """Whether to reject error fallbacks and model refusals."""
        return True

    def extra_source(self) -> str:
        """Additional source text for URL validation."""
        return ""

    def extract_image_prompt(
        self, agent: NotifyAgent, response: ControllerResponse, answer: str
    ) -> str:
        """Extract an image search query from the response."""
        raise NotImplementedError

    def prepare(self, agent: NotifyAgent) -> None:
        """Hook called before generation (e.g., set pending thought)."""
        pass

    def cleanup(self, agent: NotifyAgent) -> None:
        """Hook called after generation."""
        pass


class CheckinMode(NotificationMode):
    """Check-in: slim context, no tools, single step, no validation."""

    def get_tools(self, agent: NotifyAgent, user: str) -> list[Tool]:
        return []

    def build_system_prompt(self, agent: NotifyAgent, user: str) -> str:
        return "\n\n".join(
            s
            for s in [
                agent._identity_section(),
                agent._context_block(
                    agent._profile_section(user),
                    agent._history_section(user),
                    agent._notified_thought_section(user),
                ),
                agent._instructions_section(),
            ]
            if s
        )

    @property
    def prompt(self) -> str:
        return Prompt.NOTIFY_CHECKIN

    def get_max_steps(self) -> int:
        return PennyConstants.CHECKIN_MAX_STEPS

    @property
    def check_disqualified(self) -> bool:
        return False

    def get_history(self, agent: NotifyAgent, user: str) -> list[tuple[str, str]] | None:
        return agent._build_conversation(user)

    def extract_image_prompt(
        self, agent: NotifyAgent, response: ControllerResponse, answer: str
    ) -> str:
        return str(agent.config.runtime.CHECKIN_IMAGE_PROMPT)

    def prepare(self, agent: NotifyAgent) -> None:
        agent._pending_thought = None


class NewsMode(NotificationMode):
    """News: profile + history context, news tool only, URL validation."""

    def get_tools(self, agent: NotifyAgent, user: str) -> list[Tool]:
        if agent._news_tool:
            return [agent._news_tool]
        return []

    def build_system_prompt(self, agent: NotifyAgent, user: str) -> str:
        return "\n\n".join(
            s
            for s in [
                agent._identity_section(),
                agent._context_block(
                    agent._profile_section(user),
                    agent._history_section(user),
                ),
                agent._instructions_section(),
            ]
            if s
        )

    @property
    def prompt(self) -> str:
        return Prompt.NOTIFY_NEWS

    def get_max_steps(self) -> int:
        return PennyConstants.NEWS_NOTIFY_MAX_STEPS

    @property
    def validate_urls(self) -> bool:
        return True

    def extract_image_prompt(
        self, agent: NotifyAgent, response: ControllerResponse, answer: str
    ) -> str:
        return NotifyAgent._extract_first_headline(answer) or "latest news"


class ThoughtMode(NotificationMode):
    """Thought candidate: profile + thought context, all tools, URL validation."""

    def __init__(self, thought: Thought | None, config: Config) -> None:
        self._thought = thought
        self._config = config

    def get_tools(self, agent: NotifyAgent, user: str) -> list[Tool]:
        return agent.get_tools(user)

    def build_system_prompt(self, agent: NotifyAgent, user: str) -> str:
        return "\n\n".join(
            s
            for s in [
                agent._identity_section(),
                agent._context_block(
                    agent._profile_section(user),
                    agent._pending_thought_section(),
                ),
                agent._instructions_section(),
            ]
            if s
        )

    @property
    def prompt(self) -> str:
        return Prompt.NOTIFY_PROMPT

    def get_max_steps(self) -> int:
        return int(self._config.runtime.MESSAGE_MAX_STEPS)

    @property
    def validate_urls(self) -> bool:
        return True

    def extra_source(self) -> str:
        return self._thought.content if self._thought else ""

    def extract_image_prompt(
        self, agent: NotifyAgent, response: ControllerResponse, answer: str
    ) -> str:
        from_tools = NotifyAgent._extract_search_query(response.tool_calls)
        if from_tools:
            return from_tools
        if self._thought:
            return self._thought.title or self._thought.content[:300]
        return ""

    def prepare(self, agent: NotifyAgent) -> None:
        agent._pending_thought = self._thought

    def cleanup(self, agent: NotifyAgent) -> None:
        agent._pending_thought = None


# ── Agent ──────────────────────────────────────────────────────────────


class NotifyAgent(Agent):
    """Notification outreach agent — sends thoughts, news, and check-ins.

    Uses the template method pattern: each NotificationMode declares what
    varies (tools, prompt, context, image extraction) and _execute_mode()
    orchestrates the shared pipeline.

    Context matrix — each mode gets tailored context:

        Mode     | History | Thought    | Turns | Tools | Steps
        -------- | ------- | ---------- | ----- | ----- | -----
        Checkin  | 7d      | 1 notified | yes   | none  | 1
        News     | 7d      | -          | -     | news  | 5
        Thought  | 7d      | the thought| -     | all   | 5

    All modes include profile (user name).
    """

    name: str = "notify"

    def get_max_steps(self) -> int:
        """Read from config so /config changes take effect immediately."""
        return int(self.config.runtime.MESSAGE_MAX_STEPS)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
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
        """Check if user has un-notified thoughts."""
        return self.db.thoughts.get_next_unnotified(user) is not None

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
    CHECKIN_ACTIVE_WINDOW = PennyConstants.CHECKIN_ACTIVE_WINDOW

    def _should_checkin(self, user: str) -> bool:
        """Check-in if 24h cooldown elapsed AND user was active in last 30m."""
        return self._checkin_cooldown_elapsed() and self._user_recently_active(user)

    def _checkin_cooldown_elapsed(self) -> bool:
        """At most one check-in per 24 hours (rolling window)."""
        last = self.db.messages.get_last_checkin_time(Prompt.NOTIFY_CHECKIN, hours=24)
        if last is None:
            return True
        elapsed = (datetime.now(UTC).replace(tzinfo=None) - last).total_seconds()
        return elapsed >= PennyConstants.CHECKIN_COOLDOWN_SECONDS

    def _user_recently_active(self, user: str) -> bool:
        """User sent a message within the active window."""
        last = self.db.messages.get_latest_incoming_time(user)
        if last is None:
            return False
        elapsed = (datetime.now(UTC).replace(tzinfo=None) - last).total_seconds()
        return elapsed <= self.CHECKIN_ACTIVE_WINDOW

    # ── Notification pipeline ─────────────────────────────────────────

    # 1-in-3 chance of sending news instead of thought candidates
    NEWS_CHANCE = PennyConstants.NEWS_CHANCE

    def _news_cooldown_elapsed(self) -> bool:
        """Check if enough time has passed since the last news notification."""
        last = self.db.messages.get_last_checkin_time(Prompt.NOTIFY_NEWS, hours=48)
        if last is None:
            return True
        elapsed = (datetime.now(UTC).replace(tzinfo=None) - last).total_seconds()
        return elapsed >= self.config.runtime.NEWS_COOLDOWN

    async def _send_notification(self, user: str) -> bool:
        """Select mode and execute: check-in, news, or thought candidates."""
        assert self._channel is not None
        try:
            if self._should_checkin(user):
                return await self._send_mode(user, CheckinMode())
            if random.random() < self.NEWS_CHANCE and self._news_cooldown_elapsed():
                return await self._send_mode(user, NewsMode())
            return await self._send_best_candidate(user)
        except Exception:
            logger.exception("Failed to send notification to %s", user)
            return False
        finally:
            self._pending_thought = None

    # ── Mode execution (shared pipeline) ──────────────────────────────

    # Max retries for notification URL validation
    NOTIFY_URL_RETRIES = PennyConstants.NOTIFY_URL_RETRIES

    async def _send_mode(self, user: str, mode: NotificationMode) -> bool:
        """Execute a notification mode: generate candidate, validate, send.

        Handles tools-unavailable fallback for single-shot modes (checkin, news).
        """
        logger.info("Notify %s for %s", mode.__class__.__name__, user)
        self._last_tools_unavailable = None
        candidate = await self._execute_mode(user, mode)
        if not candidate:
            if self._last_tools_unavailable:
                return await self._send_tools_unavailable(user, self._last_tools_unavailable)
            return False
        return await self._send_candidate(user, candidate)

    async def _execute_mode(self, user: str, mode: NotificationMode) -> NotifyCandidate | None:
        """Shared pipeline: prepare, tools, prompt, run, validate, candidate."""
        mode.prepare(self)
        self._install_tools(mode.get_tools(self, user))
        system_prompt = mode.build_system_prompt(self, user)
        response = await self._run_mode(user, mode, system_prompt)
        candidate = self._to_candidate(mode, response)
        mode.cleanup(self)
        return candidate

    async def _run_mode(
        self, user: str, mode: NotificationMode, system_prompt: str
    ) -> ControllerResponse:
        """Run the model — with or without URL validation per mode."""
        if mode.validate_urls:
            return await self._run_with_url_validation(
                prompt=mode.prompt,
                max_steps=mode.get_max_steps(),
                system_prompt=system_prompt,
                extra_source=mode.extra_source(),
            )
        return await self.run(
            prompt=mode.prompt,
            max_steps=mode.get_max_steps(),
            history=mode.get_history(self, user),
            system_prompt=system_prompt,
        )

    def _to_candidate(
        self, mode: NotificationMode, response: ControllerResponse
    ) -> NotifyCandidate | None:
        """Validate response and build a NotifyCandidate."""
        answer = response.answer.strip() if response.answer else None
        if not answer:
            return None
        if self._is_tools_unavailable(answer):
            self._last_tools_unavailable = answer
            return None
        if mode.check_disqualified and self._is_disqualified(answer):
            logger.info("Disqualified candidate: %s", answer[:60])
            return None
        image_prompt = mode.extract_image_prompt(self, response, answer)
        return NotifyCandidate(
            answer=answer,
            thought=self._pending_thought,
            attachments=response.attachments or [],
            image_prompt=image_prompt,
        )

    # ── URL-validated run ────────────────────────────────────────────

    async def _run_with_url_validation(
        self,
        prompt: str,
        max_steps: int,
        system_prompt: str,
        extra_source: str = "",
    ) -> ControllerResponse:
        """Run agentic loop, retrying if the response contains hallucinated URLs."""
        response = ControllerResponse(answer="")
        for attempt in range(1 + self.NOTIFY_URL_RETRIES):
            response = await self.run(
                prompt=prompt, max_steps=max_steps, system_prompt=system_prompt
            )
            answer = response.answer.strip() if response.answer else ""
            if not answer:
                return response
            source_text = self._get_source_text()
            if extra_source:
                source_text = extra_source + "\n" + source_text
            bad_urls = self._find_hallucinated_urls(answer, source_text)
            if not bad_urls:
                return response
            logger.warning(
                "Notify URL validation attempt %d/%d: %d hallucinated URL(s): %s",
                attempt + 1,
                1 + self.NOTIFY_URL_RETRIES,
                len(bad_urls),
                ", ".join(u[:80] for u in bad_urls),
            )
        logger.warning("Notify exhausted URL validation retries, using last attempt")
        return response

    # Prefix of the tools-unavailable response (parameterized, so exact match won't work)
    _TOOLS_UNAVAILABLE_PREFIX = PennyResponse.AGENT_TOOLS_UNAVAILABLE.split("(")[0]

    @classmethod
    def _is_tools_unavailable(cls, answer: str) -> bool:
        """Check if the answer is a tools-unavailable system response."""
        return answer.startswith(cls._TOOLS_UNAVAILABLE_PREFIX)

    async def _send_tools_unavailable(self, user: str, answer: str) -> bool:
        """Send a tools-unavailable message so the user knows to investigate."""
        assert self._channel is not None
        logger.warning("Sending tools-unavailable notification to %s: %s", user, answer)
        await self._channel.send_response(
            user, answer, parent_id=None, image_prompt="", quote_message=None
        )
        return True

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
        if answer.startswith(cls._TOOLS_UNAVAILABLE_PREFIX):
            return True
        return cls._is_refusal(answer)

    # ── Thought candidates ────────────────────────────────────────────

    async def _send_best_candidate(self, user: str) -> bool:
        """Score thoughts by embedding, then generate and send the best."""
        self._last_tools_unavailable: str | None = None
        n = int(self.config.runtime.NOTIFY_CANDIDATES)
        thoughts = self._get_top_thoughts(user, n)
        if not thoughts:
            logger.warning("No viable notification candidates for %s", user)
            return False
        winner = await self._pick_best_thought(user, thoughts)
        candidate = await self._execute_mode(user, ThoughtMode(winner, self.config))
        if not candidate:
            if self._last_tools_unavailable:
                return await self._send_tools_unavailable(user, self._last_tools_unavailable)
            return False
        return await self._send_candidate(user, candidate)

    def _get_top_thoughts(self, user: str, n: int) -> list[Thought]:
        """Select diverse unnotified thoughts with per-topic 24h cooldown.

        Selection algorithm:
        1. Most recent unnotified free-thinking thought (preference_id IS NULL),
           only if no free thought was notified in the last 24h.
        2. For each preference with unnotified thoughts, skip if that preference
           was notified in the last 24h. From the remaining, pick the N-1
           least-recently-notified preferences, taking the most recent unnotified
           thought from each.
        """
        all_unnotified = self.db.thoughts.get_all_unnotified(user)
        if not all_unnotified:
            return []

        last_notified = self._get_topic_last_notified_times(user, all_unnotified)
        cutoff = datetime.now(UTC).replace(tzinfo=None)
        cooldown = PennyConstants.THOUGHT_TOPIC_COOLDOWN_SECONDS
        result: list[Thought] = []

        # 1. Most recent free-thinking thought (if not on cooldown)
        free = [t for t in all_unnotified if t.preference_id is None]
        if free and not self._topic_on_cooldown(None, last_notified, cutoff, cooldown):
            result.append(free[-1])

        # 2. Group seeded thoughts by preference, filter by cooldown
        seeded = [t for t in all_unnotified if t.preference_id is not None]
        by_pref: dict[int, list[Thought]] = {}
        for t in seeded:
            assert t.preference_id is not None
            by_pref.setdefault(t.preference_id, []).append(t)

        eligible = {
            pid: thoughts
            for pid, thoughts in by_pref.items()
            if not self._topic_on_cooldown(pid, last_notified, cutoff, cooldown)
        }
        ranked = sorted(eligible.keys(), key=lambda pid: last_notified.get(pid) or "")
        slots = n - len(result)
        for pref_id in ranked[:slots]:
            result.append(eligible[pref_id][-1])  # most recent unnotified

        return result

    def _get_topic_last_notified_times(
        self, user: str, unnotified: list[Thought]
    ) -> dict[int | None, datetime]:
        """Get the most recent notified_at per topic (preference_id or None for free)."""
        pref_ids = {t.preference_id for t in unnotified}
        all_thoughts = self.db.thoughts.get_recent(user)
        last_notified: dict[int | None, datetime] = {}
        for t in all_thoughts:
            if t.preference_id in pref_ids and t.notified_at is not None:
                ts = t.notified_at
                if t.preference_id not in last_notified or ts > last_notified[t.preference_id]:
                    last_notified[t.preference_id] = ts
        return last_notified

    @staticmethod
    def _topic_on_cooldown(
        topic_id: int | None,
        last_notified: dict[int | None, datetime],
        cutoff: datetime,
        cooldown_seconds: float,
    ) -> bool:
        """Check if a topic (preference_id or None) was notified within cooldown."""
        last = last_notified.get(topic_id)
        if last is None:
            return False
        elapsed = (cutoff - last).total_seconds()
        return elapsed < cooldown_seconds

    # ── Thought context sections ─────────────────────────────────────

    def _pending_thought_section(self) -> str | None:
        """### Your Latest Thought — the thought being shared."""
        if self._pending_thought is not None:
            return f"### Your Latest Thought\n{self._pending_thought.content}"
        return None

    def _notified_thought_section(self, user: str) -> str | None:
        """### Recent Background Thinking — already shared with user."""
        thoughts = self.db.thoughts.get_recent_notified(user, limit=1)
        if not thoughts:
            return None
        return f"### Recent Background Thinking\n{thoughts[0].content}"

    # ── Image prompt extraction ──────────────────────────────────────

    @staticmethod
    def _extract_search_query(tool_calls: list[ToolCallRecord]) -> str | None:
        """Extract the search query from tool calls for use as image prompt."""
        for tc in tool_calls:
            if tc.tool != "search":
                continue
            args = SearchArgs.model_validate(tc.arguments)
            if args.queries:
                return args.queries[0]
        return None

    # Matches **bold text** in markdown (first occurrence)
    _BOLD_PATTERN = re.compile(r"\*\*(.+?)\*\*")

    @classmethod
    def _extract_first_headline(cls, text: str) -> str | None:
        """Extract the first bold headline from response text for image search."""
        match = cls._BOLD_PATTERN.search(text)
        return match.group(1) if match else None

    # ── Candidate scoring ─────────────────────────────────────────────

    async def _pick_best_thought(self, user: str, thoughts: list[Thought]) -> Thought:
        """Score thoughts on novelty + sentiment using cached embeddings."""
        if len(thoughts) == 1 or not self._embedding_model_client:
            return thoughts[0]

        recent_vecs = await self._embed_recent_messages(user)
        likes, dislikes = load_preference_vectors(
            self.db.preferences.get_with_embeddings(user),
            PennyConstants.PreferenceValence.POSITIVE,
            PennyConstants.PreferenceValence.NEGATIVE,
        )
        raw_scores = self._score_thoughts(thoughts, recent_vecs, likes, dislikes)
        if not raw_scores:
            return thoughts[0]
        return self._select_best(raw_scores)

    @staticmethod
    def _score_thoughts(
        thoughts: list[Thought],
        recent_vecs: list[list[float]],
        likes: list[list[float]],
        dislikes: list[list[float]],
    ) -> list[tuple[Thought, float, float]]:
        """Score thoughts using cached embeddings. Skips thoughts without embeddings."""
        results: list[tuple[Thought, float, float]] = []
        for thought in thoughts:
            if not thought.embedding:
                continue
            vec = deserialize_embedding(thought.embedding)
            nov = novelty_score(vec, recent_vecs)
            sent = compute_sentiment_score(vec, likes, dislikes)
            results.append((thought, nov, sent))
        return results

    def _select_best(
        self,
        raw_scores: list[tuple],
    ):
        """Normalize novelty and sentiment to [0,1], apply weights, pick best."""
        novelties = [n for _, n, _ in raw_scores]
        sentiments = [s for _, _, s in raw_scores]
        n_min, n_max = min(novelties), max(novelties)
        s_min, s_max = min(sentiments), max(sentiments)
        n_range = n_max - n_min
        s_range = s_max - s_min

        best: NotifyCandidate | None = None
        best_score = float("-inf")
        for candidate, novelty, sentiment in raw_scores:
            norm_novelty = (novelty - n_min) / n_range if n_range else 0.5
            norm_sentiment = (sentiment - s_min) / s_range if s_range else 0.5
            novelty_weight = float(self.config.runtime.NOVELTY_WEIGHT)
            sentiment_weight = float(self.config.runtime.SENTIMENT_WEIGHT)
            score = novelty_weight * norm_novelty + sentiment_weight * norm_sentiment
            logger.info(
                "Candidate score: %.3f (novelty=%.3f, sentiment=%.3f) %s",
                score,
                norm_novelty,
                norm_sentiment,
                (getattr(candidate, "content", "") or getattr(candidate, "answer", ""))[:60],
            )
            if score > best_score:
                best_score = score
                best = candidate
        assert best is not None
        return best

    async def _embed_recent_messages(self, user: str) -> list[list[float]]:
        """Get cached embeddings of recent outgoing messages for novelty comparison."""
        if not self._embedding_model_client:
            return []
        messages = self.db.messages.get_recent_outgoing(user)
        return [deserialize_embedding(msg.embedding) for msg in messages if msg.embedding]

    # ── Send ──────────────────────────────────────────────────────────

    async def _send_candidate(self, user: str, candidate: NotifyCandidate) -> bool:
        """Send the winning candidate and mark its thought as notified."""
        assert self._channel is not None
        thought_id = candidate.thought.id if candidate.thought else None
        await self._channel.send_response(
            user,
            candidate.answer,
            parent_id=None,
            image_prompt=candidate.image_prompt,
            attachments=candidate.attachments or None,
            quote_message=None,
            thought_id=thought_id,
        )
        if candidate.thought and candidate.thought.id is not None:
            self.db.thoughts.mark_notified(candidate.thought.id)
        logger.info("Notification sent to %s", user)
        return True

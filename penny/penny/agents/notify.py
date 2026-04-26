"""NotifyAgent — Penny's notification outreach.

Sends notifications to users when idle: thought candidates and
periodic check-ins. Runs on a schedule via the BackgroundScheduler.

Each notification mode (checkin, thought) is a NotificationMode
subclass that declares its tools, prompt, context, and image extraction.
NotifyAgent orchestrates the shared pipeline: install tools, build prompt,
run model, validate, extract image, send.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from penny.agents.base import Agent
from penny.agents.models import ControllerResponse
from penny.constants import NotifyPromptType, PennyConstants
from penny.database.models import Thought
from penny.llm.embeddings import deserialize_embedding
from penny.llm.similarity import novelty_score
from penny.prompts import Prompt
from penny.responses import PennyResponse

if TYPE_CHECKING:
    from penny.channels import MessageChannel
    from penny.config import Config

logger = logging.getLogger(__name__)


class NotifyCandidate(BaseModel):
    """A candidate notification message awaiting scoring."""

    answer: str
    thought: Thought | None = None
    attachments: list[str] = Field(default_factory=list)


# ── Notification modes ─────────────────────────────────────────────────


class NotificationMode:
    """Declares what varies per notification mode.

    Each mode specifies its tools, system prompt, user prompt, and run
    parameters.  The agent orchestrates the shared pipeline around
    these declarations.

    Subclasses must override: build_system_prompt and prompt.  Tools
    are not a per-mode concern — every notify run uses the agent's
    default tool surface.  Other methods have sensible defaults.
    """

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

    def prepare(self, agent: NotifyAgent) -> None:
        """Hook called before generation (e.g., set pending thought)."""
        pass

    def cleanup(self, agent: NotifyAgent) -> None:
        """Hook called after generation."""
        pass


class CheckinMode(NotificationMode):
    """Check-in: slim context, single step, no URL validation.

    Tools are still installed (every agent gets every tool); the prompt
    drives whether the model uses them.
    """

    def build_system_prompt(self, agent: NotifyAgent, user: str) -> str:
        return "\n\n".join(
            s
            for s in [
                agent._identity_section(),
                agent._context_block(
                    agent._profile_section(user),
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

    def prepare(self, agent: NotifyAgent) -> None:
        agent._pending_thought = None


class ThoughtMode(NotificationMode):
    """Thought candidate: profile + thought context, all tools, URL validation."""

    def __init__(self, thought: Thought | None, config: Config) -> None:
        self._thought = thought
        self._config = config

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

    def prepare(self, agent: NotifyAgent) -> None:
        agent._pending_thought = self._thought

    def cleanup(self, agent: NotifyAgent) -> None:
        agent._pending_thought = None


# ── Agent ──────────────────────────────────────────────────────────────


class NotifyAgent(Agent):
    """Notification outreach agent — sends thoughts and check-ins.

    Uses the template method pattern: each NotificationMode declares what
    varies (tools, prompt, context, image extraction) and _execute_mode()
    orchestrates the shared pipeline.

    Context matrix — each mode gets tailored context:

        Mode     | History | Thought    | Turns | Tools | Steps
        -------- | ------- | ---------- | ----- | ----- | -----
        Checkin  | 7d      | 1 notified | yes   | none  | 1
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
        run_id = uuid.uuid4().hex
        return await self._send_notification(user, run_id)

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

    async def _send_notification(self, user: str, run_id: str) -> bool:
        """Select mode and execute: check-in or thought candidates."""
        assert self._channel is not None
        try:
            if self._should_checkin(user):
                return await self._send_mode(user, CheckinMode(), run_id, NotifyPromptType.CHECKIN)
            return await self._send_best_candidate(user, run_id)
        except Exception:
            logger.exception("Failed to send notification to %s", user)
            return False
        finally:
            self._pending_thought = None

    # ── Mode execution (shared pipeline) ──────────────────────────────

    # Max retries for notification URL validation
    NOTIFY_URL_RETRIES = PennyConstants.NOTIFY_URL_RETRIES

    async def _send_mode(
        self, user: str, mode: NotificationMode, run_id: str, prompt_type: str
    ) -> bool:
        """Execute a notification mode: generate candidate, validate, send."""
        logger.info("Notify %s for %s", mode.__class__.__name__, user)
        self._last_tools_unavailable = None
        candidate = await self._execute_mode(user, mode, run_id, prompt_type)
        if not candidate:
            if self._last_tools_unavailable:
                return await self._send_tools_unavailable(user, self._last_tools_unavailable)
            return False
        return await self._send_candidate(user, candidate)

    async def _execute_mode(
        self, user: str, mode: NotificationMode, run_id: str, prompt_type: str
    ) -> NotifyCandidate | None:
        """Shared pipeline: prepare, tools, prompt, run, validate, candidate."""
        mode.prepare(self)
        self._install_tools(self.get_tools(user))
        system_prompt = mode.build_system_prompt(self, user)
        response = await self._run_mode(user, mode, system_prompt, run_id, prompt_type)
        candidate = self._to_candidate(mode, response)
        mode.cleanup(self)
        return candidate

    async def _run_mode(
        self, user: str, mode: NotificationMode, system_prompt: str, run_id: str, prompt_type: str
    ) -> ControllerResponse:
        """Run the model — with or without URL validation per mode."""
        if mode.validate_urls:
            return await self._run_with_url_validation(
                prompt=mode.prompt,
                max_steps=mode.get_max_steps(),
                system_prompt=system_prompt,
                extra_source=mode.extra_source(),
                run_id=run_id,
                prompt_type=prompt_type,
            )
        return await self.run(
            prompt=mode.prompt,
            max_steps=mode.get_max_steps(),
            history=mode.get_history(self, user),
            system_prompt=system_prompt,
            run_id=run_id,
            prompt_type=prompt_type,
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
        attachments = response.attachments or []
        if not attachments and self._pending_thought and self._pending_thought.image:
            attachments = [self._pending_thought.image]
        return NotifyCandidate(
            answer=answer,
            thought=self._pending_thought,
            attachments=attachments,
        )

    # ── URL-validated run ────────────────────────────────────────────

    async def _run_with_url_validation(
        self,
        prompt: str,
        max_steps: int,
        system_prompt: str,
        extra_source: str = "",
        run_id: str | None = None,
        prompt_type: str | None = None,
    ) -> ControllerResponse:
        """Run agentic loop, retrying if the response contains hallucinated URLs."""
        response = ControllerResponse(answer="")
        for attempt in range(1 + self.NOTIFY_URL_RETRIES):
            response = await self.run(
                prompt=prompt,
                max_steps=max_steps,
                system_prompt=system_prompt,
                run_id=run_id,
                prompt_type=prompt_type,
            )
            answer = response.answer.strip() if response.answer else ""
            if not answer:
                return response
            source_text = self._get_source_text()
            if extra_source:
                source_text = f"{extra_source}\n{source_text}"
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
            user, answer, parent_id=None, author=self.name, quote_message=None
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

    async def _send_best_candidate(self, user: str, run_id: str) -> bool:
        """Score thoughts by embedding, pick the best, and send it directly.

        Thoughts are already user-facing quality from the thinking agent's
        summary step — no agentic loop or LLM rewrite needed.
        """
        n = int(self.config.runtime.NOTIFY_CANDIDATES)
        thoughts = self._get_top_thoughts(user, n)
        if not thoughts:
            logger.warning("No viable notification candidates for %s", user)
            return False
        winner = await self._pick_best_thought(user, thoughts)
        attachments = [winner.image] if winner.image else []
        candidate = NotifyCandidate(
            answer=winner.content,
            thought=winner,
            attachments=attachments,
        )
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
        ranked = sorted(eligible.keys(), key=lambda pid: last_notified.get(pid) or datetime.min)
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

    # ── Candidate scoring ─────────────────────────────────────────────

    async def _pick_best_thought(self, user: str, thoughts: list[Thought]) -> Thought:
        """Score thoughts by novelty against recent messages. Pick most novel."""
        if len(thoughts) == 1 or not self._embedding_model_client:
            return thoughts[0]

        recent_vecs = await self._embed_recent_messages(user)
        scored = self._score_thoughts_novelty(thoughts, recent_vecs)
        if not scored:
            return thoughts[0]
        return self._select_most_novel(scored)

    @staticmethod
    def _score_thoughts_novelty(
        thoughts: list[Thought],
        recent_vecs: list[list[float]],
    ) -> list[tuple[Thought, float]]:
        """Score thoughts by novelty. Skips thoughts without embeddings."""
        results: list[tuple[Thought, float]] = []
        for thought in thoughts:
            if not thought.embedding:
                continue
            vec = deserialize_embedding(thought.embedding)
            nov = novelty_score(vec, recent_vecs)
            results.append((thought, nov))
        return results

    @staticmethod
    def _select_most_novel(scored: list[tuple[Thought, float]]) -> Thought:
        """Pick the thought with the highest novelty score."""
        best, best_score = scored[0]
        for candidate, score in scored[1:]:
            logger.info(
                "Candidate novelty: %.3f — %s",
                score,
                (candidate.content or "")[:60],
            )
            if score > best_score:
                best_score = score
                best = candidate
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
            author=self.name,
            attachments=candidate.attachments or None,
            quote_message=None,
            thought_id=thought_id,
        )
        if candidate.thought and candidate.thought.id is not None:
            self.db.thoughts.mark_notified(candidate.thought.id)
        logger.info("Notification sent to %s", user)
        return True

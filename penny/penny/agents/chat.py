"""ChatAgent — Penny's conversation mode.

Handles incoming user messages with tools for search and news.
Context is injected automatically via the Agent base class.
Also runs on a schedule to proactively share thoughts when users are idle.
"""

from __future__ import annotations

import logging
import random
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from penny.agents.base import Agent
from penny.agents.models import ControllerResponse, ToolCallRecord
from penny.database.models import Thought
from penny.ollama.embeddings import cosine_similarity
from penny.ollama.similarity import embed_text
from penny.prompts import Prompt
from penny.responses import PennyResponse

if TYPE_CHECKING:
    from penny.channels import MessageChannel

logger = logging.getLogger(__name__)


class ProactiveCandidate(BaseModel):
    """A candidate proactive message awaiting scoring."""

    answer: str
    thought: Thought | None = None
    attachments: list[str] = Field(default_factory=list)
    image_prompt: str | None = None


class ChatAgent(Agent):
    """Conversation-mode agent — handles user messages and proactive outreach.

    Context matrix — each entry point gets tailored context:

        Mode     | Entities | History | Thought    | Turns | Tools | Steps
        -------- | -------- | ------- | ---------- | ----- | ----- | -----
        User Msg | msg      | 7d      | 1 notified | yes   | all   | 5
        Vision   | msg      | 7d      | 1 notified | yes   | none  | 1
        Checkin  | -        | 7d      | 1 notified | yes   | none  | 1
        News     | -        | 7d      | -          | -     | all   | 5
        Thought  | thought  | 7d      | the thought| -     | all   | 5

    All modes include profile (user name). Entity anchor column
    shows what drives the embedding similarity search.

    Conv turns only include messages since the last history rollup's
    period_end, so rolled-up content isn't duplicated as raw turns.

    Agentic loop: model responds with tool calls or text. Tool results
    are appended and the loop continues. Final step removes tools to
    force text output. Text response = done.
    """

    name: str = "chat"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._boot_time = datetime.now(UTC).replace(tzinfo=None)

    def set_channel(self, channel: MessageChannel) -> None:
        """Set the channel for sending proactive messages."""
        self._channel: MessageChannel | None = channel

    # ── Proactive messaging ────────────────────────────────────────────

    async def execute_for_user(self, user: str) -> bool:
        """Scheduled cycle: send a proactive message if the user has been idle."""
        if not self._should_send_proactive(user):
            return False
        return await self._send_proactive(user)

    def _should_send_proactive(self, user: str) -> bool:
        """Python-space eligibility checks for proactive messaging."""
        if not getattr(self, "_channel", None):
            return False
        if self.db.users.is_muted(user):
            return False
        if not self._has_recent_thoughts(user):
            return False
        return self._cooldown_elapsed(user)

    # Check-in requires user activity within this window (seconds)
    CHECKIN_ACTIVE_WINDOW = 1800  # 30 minutes

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
            self.config.runtime.PROACTIVE_COOLDOWN_MIN * (2 ** max(count - 1, 0)),
            self.config.runtime.PROACTIVE_COOLDOWN_MAX,
        )
        return elapsed >= cooldown

    async def _get_top_thoughts(self, user: str, n: int) -> list[Thought]:
        """Rank un-notified thoughts by preference affinity and return top N."""
        hours = int(self.config.runtime.THOUGHT_FRESHNESS_HOURS)
        thoughts = self.db.thoughts.get_all_unnotified(user, freshness_hours=hours)
        if len(thoughts) <= n:
            return thoughts
        return await self._rank_thoughts(user, thoughts, n)

    async def _rank_thoughts(self, user: str, thoughts: list[Thought], n: int) -> list[Thought]:
        """Score thoughts by preference affinity and return the top N."""
        likes, dislikes = self._load_preference_vectors(user)
        if (not likes and not dislikes) or not self._embedding_model_client:
            return random.sample(thoughts, n)

        scored: list[tuple[Thought, float]] = []
        for thought in thoughts:
            vec = await embed_text(self._embedding_model_client, thought.content)
            if vec is None:
                continue
            score = self._compute_sentiment_score(vec, likes, dislikes)
            scored.append((thought, score))

        if not scored:
            return random.sample(thoughts, n)
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [t for t, _ in scored[:n]]

    # ── Check-in gating ───────────────────────────────────────────────

    def _should_checkin(self, user: str) -> bool:
        """Check-in if 24h cooldown elapsed AND user was active in last 30m."""
        return self._checkin_cooldown_elapsed() and self._user_recently_active(user)

    def _checkin_cooldown_elapsed(self) -> bool:
        """At most one check-in per 24 hours (rolling window)."""
        last = self.db.messages.get_last_checkin_time(Prompt.PROACTIVE_CHECKIN, hours=24)
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

    # ── Proactive pipeline ────────────────────────────────────────────

    # 1-in-3 chance of sending news instead of thought candidates
    NEWS_CHANCE = 1 / 3

    async def _send_proactive(self, user: str) -> bool:
        """Check-in if eligible, then coin-flip news, otherwise thought candidates."""
        assert self._channel is not None
        try:
            if self._should_checkin(user):
                return await self._send_checkin(user)
            if random.random() < self.NEWS_CHANCE:
                return await self._send_news(user)
            return await self._send_best_candidate(user)
        except Exception:
            logger.exception("Failed to send proactive message to %s", user)
            return False
        finally:
            self._proactive_thought = None

    async def _send_checkin(self, user: str) -> bool:
        """Send a check-in message — slim context, no tools, single step."""
        logger.info("Proactive check-in for %s", user)
        self._proactive_thought = None
        context = self._build_checkin_context(user)
        self._install_tools([])
        response = await self.run(
            prompt=Prompt.PROACTIVE_CHECKIN,
            history=self._build_conversation(user),
            context=context,
            max_steps=1,
        )
        answer = response.answer.strip() if response.answer else None
        if not answer:
            return False
        return await self._send_candidate(
            user, ProactiveCandidate(answer=answer, attachments=response.attachments or [])
        )

    def _build_checkin_context(self, user: str) -> str:
        """Checkin context: profile + history rollups + last thought. No entities."""
        sections: list[str | None] = [
            self._build_profile_context(user, None),
            self._build_history_context(user),
            self._build_thought_context(user),
        ]
        return "\n\n".join(s for s in sections if s)

    async def _send_news(self, user: str) -> bool:
        """Send a news message — profile + history only, tools enabled."""
        logger.info("Proactive news for %s", user)
        context = self._build_news_context(user)
        self._install_tools(self.get_tools(user))
        response = await self.run(
            prompt=Prompt.PROACTIVE_NEWS,
            context=context,
        )
        answer = response.answer.strip() if response.answer else None
        if not answer:
            return False
        image_prompt = self._extract_image_prompt(response.tool_calls)
        return await self._send_candidate(
            user,
            ProactiveCandidate(
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
    def _extract_image_prompt(tool_calls: list[ToolCallRecord]) -> str | None:
        """Extract an image search query from tool calls (news topic or search query)."""
        for tc in tool_calls:
            if tc.tool == "fetch_news" and tc.arguments.get("topic"):
                return tc.arguments["topic"]
            if tc.tool == "search" and tc.arguments.get("query"):
                return tc.arguments["query"]
        return None

    async def _send_best_candidate(self, user: str) -> bool:
        """Generate thought candidates, score, send the best."""
        n = int(self.config.runtime.PROACTIVE_CANDIDATES)
        candidates = await self._generate_thought_candidates(user, n)
        if not candidates:
            logger.warning("No viable proactive candidates for %s", user)
            return False
        winner = await self._pick_best_candidate(user, candidates)
        return await self._send_candidate(user, winner)

    async def _generate_thought_candidates(self, user: str, n: int) -> list[ProactiveCandidate]:
        """Generate N thought candidates ranked by preference affinity."""
        candidates: list[ProactiveCandidate] = []
        thoughts = await self._get_top_thoughts(user, n)
        for i, thought in enumerate(thoughts):
            candidate = await self._generate_one_candidate(
                user, Prompt.PROACTIVE_PROMPT, thought=thought
            )
            if candidate:
                logger.info(
                    "Candidate thought %d/%d: %s", i + 1, len(thoughts), candidate.answer[:60]
                )
                candidates.append(candidate)
        return candidates

    async def _generate_one_candidate(
        self, user: str, prompt: str, thought: Thought | None
    ) -> ProactiveCandidate | None:
        """Generate a single proactive candidate via the agentic loop.

        Uses thought-specific context (profile + history rollups + thought)
        without conversation turns, so the model focuses on the thought
        rather than continuing the conversation.
        """
        self._proactive_thought = thought
        context = self._build_thought_candidate_context(user)
        self._install_tools(self.get_tools(user))
        response = await self.run(prompt=prompt, context=context)
        self._proactive_thought = None
        answer = response.answer.strip() if response.answer else None
        if not answer:
            return None
        image_prompt = self._extract_image_prompt(response.tool_calls)
        return ProactiveCandidate(
            answer=answer,
            thought=thought,
            attachments=response.attachments or [],
            image_prompt=image_prompt,
        )

    def _build_thought_candidate_context(self, user: str) -> str:
        """Thought candidate context: profile + thought only. No history or conv turns."""
        sections: list[str | None] = [
            self._build_profile_context(user, None),
            self._build_thought_context(user),
        ]
        return "\n\n".join(s for s in sections if s)

    async def _pick_best_candidate(
        self, user: str, candidates: list[ProactiveCandidate]
    ) -> ProactiveCandidate:
        """Score candidates on novelty + sentiment and return the best."""
        if len(candidates) == 1 or not self._embedding_model_client:
            return candidates[0]

        recent_vecs = await self._embed_recent_messages(user)
        likes, dislikes = self._load_preference_vectors(user)
        best: ProactiveCandidate | None = None
        best_score = float("-inf")

        for candidate in candidates:
            vec = await embed_text(self._embedding_model_client, candidate.answer)
            if vec is None:
                continue
            novelty = self._novelty_score(vec, recent_vecs)
            sentiment = self._compute_sentiment_score(vec, likes, dislikes)
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

    @staticmethod
    def _novelty_score(vec: list[float], recent_vecs: list[list[float]]) -> float:
        """1 - max similarity to any recent message. Higher = more novel."""
        if not recent_vecs:
            return 1.0
        max_sim = max(cosine_similarity(vec, rv) for rv in recent_vecs)
        return 1.0 - max_sim

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

    async def _send_candidate(self, user: str, candidate: ProactiveCandidate) -> bool:
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
        logger.info("Proactive message sent to %s", user)
        return True

    # ── Message handling ───────────────────────────────────────────────

    async def handle(
        self,
        content: str,
        sender: str,
        images: list[str] | None = None,
        entity_anchor: str | None = None,
    ) -> ControllerResponse:
        """Handle an incoming message — summary method.

        Builds context, processes images, runs agentic loop.
        entity_anchor overrides what drives entity similarity search
        (e.g. thought content for proactive messages).
        """
        self._current_user = sender
        self._pending_content = entity_anchor or content
        try:
            content, has_images = await self._process_images(content, images)
            context_text = await self.get_context(sender)
            history = self.get_history(sender)

            if has_images:
                logger.info("Handling vision message from %s", sender)
                self._install_tools([])
                return await self.run(
                    prompt=content,
                    history=history,
                    max_steps=1,
                    system_prompt=Prompt.VISION_RESPONSE_PROMPT,
                    context=context_text,
                )

            logger.info("Handling message from %s (conversation mode)", sender)
            self._install_tools(self.get_tools(sender))
            return await self.run(
                prompt=content,
                history=history,
                context=context_text,
            )
        finally:
            self._current_user = None
            self._pending_content = None

    # ── Hooks ─────────────────────────────────────────────────────────────

    async def get_context(self, user: str) -> str:
        """Full context — profile, history, thought.

        Entity context disabled while evaluating usefulness.
        """
        content = self._pending_content
        sections: list[str | None] = [
            self._build_profile_context(user, content),
            # TODO: entity context disabled while evaluating usefulness
            # await self._build_entity_context(user, content),
            self._build_history_context(user),
            self._build_thought_context(user),
        ]
        return "\n\n".join(s for s in sections if s)

    def _build_thought_context(self, sender: str) -> str | None:
        """Build thought context — single thought for proactive, notified thoughts for chat.

        Only thoughts Penny has actually shared with the user appear in chat context.
        Thinking agent uses the base version (all thoughts) to avoid repetition.
        """
        thought = getattr(self, "_proactive_thought", None)
        if thought is not None:
            return f"## Your Latest Thought\n{thought.content}"
        hours = int(self.config.runtime.THOUGHT_FRESHNESS_HOURS)
        thoughts = self.db.thoughts.get_recent_notified(sender, freshness_hours=hours, limit=1)
        if not thoughts:
            return None
        return f"## Recent Background Thinking\n{thoughts[0].content}"

    def get_history(self, user: str) -> list[tuple[str, str]] | None:
        """Recent conversation messages for chat continuity."""
        return self._build_conversation(user)

    # ── Image processing ──────────────────────────────────────────────────

    async def _process_images(self, content: str, images: list[str] | None) -> tuple[str, bool]:
        """Caption images with vision model and build combined text prompt."""
        if not images:
            return content, False

        captions = [await self.caption_image(img) for img in images]
        caption = ", ".join(captions)
        if content:
            content = PennyResponse.VISION_IMAGE_CONTEXT.format(user_text=content, caption=caption)
        else:
            content = PennyResponse.VISION_IMAGE_ONLY_CONTEXT.format(caption=caption)
        logger.info("Built vision prompt: %s", content[:200])
        return content, True

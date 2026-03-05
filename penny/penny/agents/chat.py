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

from penny.agents.base import Agent
from penny.agents.models import ControllerResponse
from penny.database.models import Thought
from penny.ollama.similarity import embed_text
from penny.prompts import Prompt
from penny.responses import PennyResponse

if TYPE_CHECKING:
    from penny.channels import MessageChannel

logger = logging.getLogger(__name__)


class ChatAgent(Agent):
    """Conversation-mode agent — handles user messages with tools.

    System message::

        [system]
        <current datetime>
        <Prompt.PENNY_IDENTITY>
        <user profile — name from db.users.get_info, skipped if none>

        ## Relevant Knowledge
        - <top K entities by embedding similarity to user message,
          with up to ENTITY_CONTEXT_MAX_FACTS per entity, from db.entities + db.facts>

        ## Recent Discussion Topics
        - <daily topic summaries from db.history, up to HISTORY_CONTEXT_LIMIT>

        ## Recent Background Thinking
        <today's thought summaries from db.thoughts, up to THOUGHT_CONTEXT_LIMIT>

        <Prompt.CONVERSATION_PROMPT with {tools} listing>

    Conversation history::

        [user]      <message>
        [assistant]  <response>
        ...
        <today's messages from db.messages, up to MESSAGE_CONTEXT_LIMIT turns>

    Current message::

        [user]  <the incoming message being handled>

    Agentic loop::

        [assistant]  <tool call or text response>
        [tool]       <tool result>
        ...
        <repeats until text response or MESSAGE_MAX_STEPS reached.
        Final step removes tools to force text output.>

    Tools:
        search     — web search (Perplexity + Serper images)
        fetch_news — news search (TheNewsAPI.com)

    Exit condition:
        Text response (no tool calls) → returned to user.

    Vision path:
        Images captioned via vision model, single no-tool call
        with VISION_RESPONSE_PROMPT. No tools, no agentic loop.
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

    _THOUGHT_VARIANTS: list[str] = [
        Prompt.PROACTIVE_PROMPT,
        Prompt.PROACTIVE_NEWS,
        Prompt.PROACTIVE_FOLLOWUP,
    ]

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

    async def _get_next_thought(self, user: str) -> Thought | None:
        """Pick the best un-notified thought by preference affinity.

        Scores all today's un-notified thoughts against user preferences
        and randomly selects from the top pool.
        """
        hours = int(self.config.runtime.THOUGHT_FRESHNESS_HOURS)
        thoughts = self.db.thoughts.get_all_unnotified(user, freshness_hours=hours)
        if not thoughts:
            return None
        if len(thoughts) == 1:
            return thoughts[0]
        return await self._pick_preferred_thought(user, thoughts)

    async def _pick_preferred_thought(self, user: str, thoughts: list[Thought]) -> Thought:
        """Score thoughts by preference affinity and pick from the top pool."""
        likes, dislikes = self._load_preference_vectors(user)
        if (not likes and not dislikes) or not self._embedding_model_client:
            return random.choice(thoughts)

        scored = await self._score_thoughts(thoughts, likes, dislikes)
        if not scored:
            return random.choice(thoughts)

        scored.sort(key=lambda pair: pair[1], reverse=True)
        pool = scored[: self.PREFERRED_POOL_SIZE]
        return random.choice(pool)[0]

    async def _score_thoughts(
        self,
        thoughts: list[Thought],
        likes: list[list[float]],
        dislikes: list[list[float]],
    ) -> list[tuple[Thought, float]]:
        """Embed each thought and compute preference sentiment score."""
        scored: list[tuple[Thought, float]] = []
        for thought in thoughts:
            vec = await embed_text(self._embedding_model_client, thought.content)
            if vec is None:
                continue
            score = self._compute_sentiment_score(vec, likes, dislikes)
            scored.append((thought, score))
        return scored

    def _pick_thought_variant(self) -> str:
        """Randomly select a thought-sharing prompt variant."""
        return random.choice(self._THOUGHT_VARIANTS)

    async def _pick_proactive_mode(self, user: str) -> tuple[str, Thought | None]:
        """Choose between thought-sharing variants and check-in.

        Returns (prompt, thought). ~1/6 chance of check-in.
        All thought-sharing variants anchor entity search to the thought content.
        """
        if random.random() < 1 / 6:
            logger.info("Proactive check-in for %s", user)
            return Prompt.PROACTIVE_CHECKIN, None
        thought = await self._get_next_thought(user)
        variant = self._pick_thought_variant()
        logger.info("Proactive thought-sharing (%s) for %s", variant[:40], user)
        return variant, thought

    async def _send_proactive(self, user: str) -> bool:
        """Generate a proactive message — thought-sharing or check-in."""
        assert self._channel is not None
        try:
            prompt, thought = await self._pick_proactive_mode(user)
            self._proactive_thought = thought
            anchor = thought.content if thought else None
            response = await self.handle(
                content=prompt,
                sender=user,
                entity_anchor=anchor,
            )
            answer = response.answer.strip() if response.answer else None
            if not answer:
                logger.warning("Proactive message produced empty response for %s", user)
                return False

            await self._channel.send_response(
                user,
                answer,
                parent_id=None,
                attachments=response.attachments or None,
                quote_message=None,
            )
            if thought and thought.id is not None:
                self.db.thoughts.mark_notified(thought.id)
            logger.info("Proactive message sent to %s", user)
            return True
        except Exception:
            logger.exception("Failed to send proactive message to %s", user)
            return False
        finally:
            self._proactive_thought = None

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
        """Full context with entity similarity anchored to message content."""
        content = self._pending_content
        sections: list[str | None] = [
            self._build_profile_context(user, content),
            await self._build_entity_context(user, content),
            self._build_history_context(user),
            self._build_thought_context(user),
            self._build_dislike_context(user),
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
        thoughts = self.db.thoughts.get_recent_notified(
            sender, freshness_hours=hours, limit=self.THOUGHT_CONTEXT_LIMIT
        )
        if not thoughts:
            return None
        lines = [t.content for t in thoughts]
        return "## Recent Background Thinking\n" + "\n\n".join(lines)

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

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

    def _has_recent_thoughts(self, user: str) -> bool:
        """Check if user has thoughts since midnight today."""
        midnight = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
        thoughts = self.db.thoughts.get_recent(user)
        return any(t.created_at >= midnight for t in thoughts)

    def _cooldown_elapsed(self, user: str) -> bool:
        """Check if enough time since last autonomous outgoing message.

        Uses exponential backoff: cooldown doubles with each consecutive
        autonomous message since the user's last incoming message.
        """
        latest = self.db.messages.get_latest_autonomous_outgoing_time(user)
        if latest is None:
            return True
        elapsed = (datetime.now(UTC).replace(tzinfo=None) - latest).total_seconds()
        count = self.db.messages.count_autonomous_since_last_incoming(user)
        cooldown = min(
            self.config.runtime.PROACTIVE_COOLDOWN_MIN * (2 ** max(count - 1, 0)),
            self.config.runtime.PROACTIVE_COOLDOWN_MAX,
        )
        return elapsed >= cooldown

    def _get_latest_thought_content(self, user: str) -> str | None:
        """Get the most recent thought content for entity context anchoring."""
        thoughts = self.db.thoughts.get_recent(user, limit=1)
        return thoughts[0].content if thoughts else None

    def _pick_proactive_mode(self, user: str) -> tuple[str, str | None]:
        """Choose between thought-sharing and check-in prompt.

        Returns (prompt, entity_anchor). ~1/6 chance of check-in.
        """
        if random.random() < 1 / 6:
            logger.info("Proactive check-in for %s", user)
            return Prompt.PROACTIVE_CHECKIN, None
        return Prompt.PROACTIVE_PROMPT, self._get_latest_thought_content(user)

    async def _send_proactive(self, user: str) -> bool:
        """Generate a proactive message — thought-sharing or check-in."""
        assert self._channel is not None
        try:
            prompt, anchor = self._pick_proactive_mode(user)
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
            logger.info("Proactive message sent to %s", user)
            return True
        except Exception:
            logger.exception("Failed to send proactive message to %s", user)
            return False

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
        ]
        return "\n\n".join(s for s in sections if s)

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

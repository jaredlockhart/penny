"""ChatAgent — Penny's conversation mode.

Handles incoming user messages with tools for search and news.
Context is injected automatically via the Agent base class.
"""

from __future__ import annotations

import logging

from penny.agents.base import Agent
from penny.agents.models import ControllerResponse
from penny.prompts import Prompt
from penny.responses import PennyResponse

logger = logging.getLogger(__name__)


class ChatAgent(Agent):
    """Conversation-mode agent — handles user messages.

    Context matrix:

        Mode     | Entities | History | Thought    | Turns | Tools | Steps
        -------- | -------- | ------- | ---------- | ----- | ----- | -----
        User Msg | msg      | 7d      | 1 notified | yes   | all   | 5
        Vision   | msg      | 7d      | 1 notified | yes   | none  | 1

    All modes include profile (user name). Entity anchor column
    shows what drives the embedding similarity search.

    Conv turns only include messages since the last history rollup's
    period_end, so rolled-up content isn't duplicated as raw turns.

    Agentic loop: model responds with tool calls or text. Tool results
    are appended and the loop continues. Final step removes tools to
    force text output. Text response = done.
    """

    name: str = "chat"

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
        (e.g. thought content for notifications).
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
        """Build thought context — only thoughts Penny has shared with the user.

        Only notified thoughts appear in chat context so the model
        doesn't reference thoughts the user hasn't seen yet.
        """
        hours = int(self.config.runtime.THOUGHT_FRESHNESS_HOURS)
        thoughts = self.db.thoughts.get_recent_notified(sender, freshness_hours=hours, limit=1)
        if not thoughts:
            return None
        return f"## Recent Background Thinking\n{thoughts[0].content}"

    def get_history(self, user: str) -> list[tuple[str, str]] | None:
        """Recent conversation messages for chat continuity."""
        return self._build_conversation(user)

    # ── Vision ────────────────────────────────────────────────────────────

    async def caption_image(self, image_b64: str) -> str:
        """Caption an image using the vision model."""
        messages = [
            {"role": "user", "content": Prompt.VISION_AUTO_DESCRIBE_PROMPT, "images": [image_b64]},
        ]
        assert self._vision_model_client is not None
        response = await self._vision_model_client.chat(messages=messages)
        return response.content.strip()

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

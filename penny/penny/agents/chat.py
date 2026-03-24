"""ChatAgent — Penny's conversation mode.

Handles incoming user messages with tools for search and news.
Context is injected automatically via the Agent base class.
"""

from __future__ import annotations

import logging

from penny.agents.base import Agent
from penny.agents.models import ControllerResponse
from penny.constants import PennyConstants
from penny.prompts import Prompt
from penny.responses import PennyResponse

logger = logging.getLogger(__name__)


class ChatAgent(Agent):
    """Conversation-mode agent — handles user messages.

    Context matrix:

        Mode     | History | Thought    | Turns | Tools | Steps
        -------- | ------- | ---------- | ----- | ----- | -----
        User Msg | 7d      | 1 notified | yes   | all   | 5
        Vision   | 7d      | 1 notified | yes   | none  | 1

    All modes include profile (user name).

    Conv turns only include messages since the last history rollup's
    period_end, so rolled-up content isn't duplicated as raw turns.

    Agentic loop: model responds with tool calls or text. Tool results
    are appended and the loop continues. Final step removes tools to
    force text output. Text response = done.
    """

    name: str = "chat"

    def get_max_steps(self) -> int:
        """Read from config each call so /config changes take effect immediately."""
        return int(self.config.runtime.MESSAGE_MAX_STEPS)

    # ── Message handling ───────────────────────────────────────────────

    async def handle(
        self,
        content: str,
        sender: str,
        images: list[str] | None = None,
    ) -> ControllerResponse:
        """Handle an incoming message — summary method.

        Builds context, processes images, runs agentic loop.
        """
        self._current_user = sender
        try:
            content, has_images = await self._process_images(content, images)
            history = self.get_history(sender)

            if has_images:
                logger.info("Handling vision message from %s", sender)
                self._install_tools([])
                system_prompt = await self._build_system_prompt(
                    sender, content, instructions=Prompt.VISION_RESPONSE_PROMPT
                )
                return await self.run(
                    prompt=content,
                    history=history,
                    max_steps=PennyConstants.VISION_MAX_STEPS,
                    system_prompt=system_prompt,
                )

            logger.info("Handling message from %s (conversation mode)", sender)
            self._install_tools(self.get_tools(sender))
            system_prompt = await self._build_system_prompt(sender, content)
            return await self.run(
                prompt=content,
                max_steps=self.get_max_steps(),
                history=history,
                system_prompt=system_prompt,
            )
        finally:
            self._current_user = None

    # ── System prompt ──────────────────────────────────────────────────────

    async def _build_system_prompt(
        self, user: str, content: str | None = None, instructions: str | None = None
    ) -> str:
        """Identity + profile + history + thought + instructions."""
        return "\n\n".join(
            s
            for s in [
                self._identity_section(),
                self._context_block(
                    self._profile_section(user, content),
                    self._history_section(user),
                    self._thought_section(user),
                ),
                self._instructions_section(instructions),
            ]
            if s
        )

    def _thought_section(self, sender: str) -> str | None:
        """Build thought context — only thoughts Penny has shared with the user.

        Only notified thoughts appear in chat context so the model
        doesn't reference thoughts the user hasn't seen yet.
        """
        thoughts = self.db.thoughts.get_recent_notified(sender, limit=1)
        if not thoughts:
            return None
        return f"### Recent Background Thinking\n{thoughts[0].content}"

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

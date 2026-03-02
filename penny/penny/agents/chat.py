"""ChatAgent — conversation mode of Penny's mind.

Same tools and thought stream as the thinking agent, but oriented toward
real-time conversation with the user. Context is injected automatically
via the shared PennyAgent base class.
"""

from __future__ import annotations

import logging

from penny.agents.models import ControllerResponse, MessageRole
from penny.agents.penny_agent import PennyAgent
from penny.prompts import Prompt
from penny.responses import PennyResponse

logger = logging.getLogger(__name__)


class ChatAgent(PennyAgent):
    """Conversation-mode agent — same mind as inner monologue, user-facing."""

    async def handle(
        self,
        content: str,
        sender: str,
        quoted_text: str | None = None,
        images: list[str] | None = None,
    ) -> ControllerResponse:
        """Handle an incoming message — summary method.

        Builds shared context, processes images, runs agentic loop, logs thought.
        """
        history = await self._build_context(sender, content)
        if quoted_text:
            history = self._inject_quoted_context(quoted_text, history)
        content, has_images = await self._process_images(content, images)

        if has_images:
            logger.info("Handling vision message from %s", sender)
            response = await self.run(
                prompt=content,
                history=history,
                use_tools=False,
                max_steps=1,
                system_prompt=Prompt.VISION_RESPONSE_PROMPT,
            )
        else:
            logger.info("Handling message from %s (conversation mode)", sender)
            self._install_tools(self._build_tools(sender))
            response = await self.run(
                prompt=content,
                history=history,
                use_tools=True,
                system_prompt=Prompt.CONVERSATION_PROMPT,
            )

        self._log_conversation_thought(sender, content, response.answer)
        return response

    # ── Quote context ─────────────────────────────────────────────────────

    def _inject_quoted_context(
        self,
        quoted_text: str,
        history: list[tuple[str, str]] | None,
    ) -> list[tuple[str, str]] | None:
        """Inject quoted message as context when user is reply-quoting."""
        context = f'The user is reply-quoting your previous message: "{quoted_text}"'
        history = history or []
        return [*history, (MessageRole.SYSTEM.value, context)]

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

    # ── Post-response ─────────────────────────────────────────────────────

    def _log_conversation_thought(
        self, sender: str, user_message: str, response: str | None
    ) -> None:
        """Log a thought summarizing the conversation for inner monologue continuity."""
        try:
            reply = (response or "")[:150]
            msg = user_message[:150]
            thought = f'Conversation: user said "{msg}" — I replied "{reply}"'
            self.db.thoughts.add(sender, thought)
            logger.info("[conversation_thought] %s", thought[:200])
        except Exception:
            logger.warning("Failed to log conversation thought")

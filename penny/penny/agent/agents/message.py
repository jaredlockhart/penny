"""MessageAgent for handling incoming user messages."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from penny.agent.base import Agent
from penny.agent.models import ControllerResponse, MessageRole

if TYPE_CHECKING:
    from penny.profile import ProfilePromptHandler

logger = logging.getLogger(__name__)


class MessageAgent(Agent):
    """Agent for handling incoming user messages."""

    def __init__(self, **kwargs: object):
        """Initialize MessageAgent."""
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._profile_handler: ProfilePromptHandler | None = None

    def set_profile_handler(self, handler: ProfilePromptHandler) -> None:
        """Set the profile prompt handler for collecting user info."""
        self._profile_handler = handler

    async def handle(
        self,
        content: str,
        sender: str,
        quoted_text: str | None = None,
    ) -> tuple[int | None, ControllerResponse]:
        """
        Handle an incoming message by preparing context and running the agent.

        Args:
            content: The message content from the user
            sender: The sender identifier
            quoted_text: Optional quoted text if this is a reply

        Returns:
            Tuple of (parent_id for thread linking, ControllerResponse with answer)
        """
        # Check if we need to collect user profile first
        if self._profile_handler and self._profile_handler.needs_profile(sender):
            logger.info("User %s needs to provide profile info", sender)
            response_text, success = await self._profile_handler.collect_profile(sender, content)

            # If collection failed, return error message
            if not success:
                return None, ControllerResponse(answer=response_text)

            # Profile collected successfully - now process the original message
            logger.info("Profile collected for %s, processing original message", sender)
            # Fall through to normal message processing

        # Get thread context if quoted
        parent_id = None
        history = None
        if quoted_text:
            parent_id, history = self.db.get_thread_context(quoted_text)

        # Inject user profile context if available
        if self._profile_handler:
            profile_summary = self._profile_handler.get_profile_summary(sender)
            if profile_summary:
                # Prepend profile context to history
                history = history or []
                history = [(MessageRole.SYSTEM.value, profile_summary), *history]
                logger.debug("Injected profile context for %s", sender)

        # Run agent
        response = await self.run(prompt=content, history=history)

        return parent_id, response

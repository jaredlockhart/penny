"""MessageAgent for handling incoming user messages."""

from __future__ import annotations

import logging

from penny.agent.base import Agent
from penny.agent.models import ControllerResponse, MessageRole
from penny.tools.builtin import SearchTool

logger = logging.getLogger(__name__)


class MessageAgent(Agent):
    """Agent for handling incoming user messages."""

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
        # Get thread context if quoted
        parent_id = None
        history = None
        if quoted_text:
            parent_id, history = self.db.get_thread_context(quoted_text)

        # Inject user profile context if available
        try:
            user_info = self.db.get_user_info(sender)
            if user_info:
                profile_summary = (
                    f"User context: {user_info.name}, {user_info.location} ({user_info.timezone})"
                )
                # Prepend profile context to history
                history = history or []
                history = [(MessageRole.SYSTEM.value, profile_summary), *history]
                logger.debug("Injected profile context for %s", sender)

                # Redact user name from outbound search queries
                search_tool = self._tool_registry.get("search")
                if isinstance(search_tool, SearchTool):
                    search_tool.redact_terms = [user_info.name]
        except Exception:
            # Silently skip if userinfo table doesn't exist (e.g., in test mode)
            pass

        # Run agent
        response = await self.run(prompt=content, history=history)

        return parent_id, response

"""MessageAgent for handling incoming user messages."""

import logging

from penny.agent.base import Agent
from penny.agent.models import ControllerResponse

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
            sender: The sender identifier (unused in this version)
            quoted_text: Optional quoted text if this is a reply

        Returns:
            Tuple of (parent_id for thread linking, ControllerResponse with answer)
        """
        # Get thread context if quoted
        parent_id = None
        history = None
        if quoted_text:
            parent_id, history = self.db.get_thread_context(quoted_text)

        # Run agent
        response = await self.run(prompt=content, history=history)

        return parent_id, response

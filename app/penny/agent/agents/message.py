"""MessageAgent for handling incoming user messages."""

from penny.agent.base import Agent
from penny.agent.models import ControllerResponse, MessageRole


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
            sender: The sender identifier for profile lookup
            quoted_text: Optional quoted text if this is a reply

        Returns:
            Tuple of (parent_id for thread linking, ControllerResponse with answer)
        """
        parent_id = None
        history = None
        if quoted_text:
            parent_id, history = self.db.get_thread_context(quoted_text)

        # Inject user profile if available
        history = self._inject_profile(sender, history)

        response = await self.run(prompt=content, history=history)
        return parent_id, response

    def _inject_profile(
        self, sender: str, history: list[tuple[str, str]] | None
    ) -> list[tuple[str, str]] | None:
        """Inject user profile at the start of history if available."""
        profile = self.db.get_user_profile(sender)
        if not profile:
            return history

        profile_entry = (
            MessageRole.SYSTEM.value,
            f"User profile for this conversation:\n{profile.profile_text}",
        )

        if history is None:
            return [profile_entry]
        else:
            return [profile_entry, *history]

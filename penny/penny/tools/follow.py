"""Follow tool — manage ongoing news monitoring subscriptions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from penny.tools.base import Tool

if TYPE_CHECKING:
    from penny.commands.models import CommandContext

logger = logging.getLogger(__name__)


class FollowTool(Tool):
    """Manage ongoing news monitoring subscriptions."""

    name = "follow"
    description = (
        "Manage news monitoring. Actions: "
        "'status' to see current follows, "
        "'create' to start following a topic (include timing like 'daily AI news'), "
        "'cancel' to stop following a topic by name."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "create", "cancel"],
                "description": "What to do",
            },
            "topic": {
                "type": "string",
                "description": "Topic with optional timing (for 'create') or name (for 'cancel')",
            },
        },
        "required": ["action"],
    }

    def __init__(self, command_context: CommandContext):
        self._context = command_context

    async def execute(self, **kwargs: Any) -> str:
        """Route to status, create, or cancel."""
        action: str = kwargs["action"]
        if action == "status":
            return self._status()
        if action == "create":
            return await self._create(kwargs.get("topic", ""))
        if action == "cancel":
            return self._cancel(kwargs.get("topic", ""))
        return f"Unknown action: {action}"

    def _status(self) -> str:
        """Show active follow subscriptions."""
        logger.info("[inner_monologue] follow: status")
        follows = self._context.db.follow_prompts.get_active(self._context.user)
        if not follows:
            return "No active follow subscriptions."
        lines = []
        for fp in follows:
            date = fp.created_at.strftime("%Y-%m-%d")
            lines.append(f"- {fp.prompt_text} ({fp.timing_description}, since {date})")
        return "\n".join(lines)

    async def _create(self, topic: str) -> str:
        """Create a follow prompt via the standard command flow."""
        if not topic:
            return "Topic is required for 'create' action."
        logger.info("[inner_monologue] follow: create '%s'", topic)
        from penny.commands.follow import FollowCommand

        command = FollowCommand()
        result = await command.execute(topic, self._context)
        return result.text

    def _cancel(self, topic: str) -> str:
        """Cancel a follow by matching topic name."""
        if not topic:
            return "Topic name is required for 'cancel' action."
        logger.info("[inner_monologue] follow: cancel '%s'", topic)
        follows = self._context.db.follow_prompts.get_active(self._context.user)
        topic_lower = topic.lower()
        match = next((fp for fp in follows if topic_lower in fp.prompt_text.lower()), None)
        if not match or match.id is None:
            return f"No active follow matching '{topic}'."
        self._context.db.events.delete_for_follow_prompt(match.id)
        self._context.db.follow_prompts.cancel(match.id)
        return f"Cancelled follow for '{match.prompt_text}'."

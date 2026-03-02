"""Learn tool — manage background research on topics."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from penny.tools.base import Tool

if TYPE_CHECKING:
    from penny.database import Database

logger = logging.getLogger(__name__)


class LearnTool(Tool):
    """Manage background research on topics."""

    name = "learn"
    description = (
        "Manage background research. Actions: "
        "'status' to see current learn prompts, "
        "'create' to start researching a new topic."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "create"],
                "description": "What to do",
            },
            "topic": {
                "type": "string",
                "description": "Topic to research (required for 'create')",
            },
        },
        "required": ["action"],
    }

    def __init__(self, db: Database, user: str, searches_remaining: int):
        self._db = db
        self._user = user
        self._searches_remaining = searches_remaining

    async def execute(self, **kwargs: Any) -> str:
        """Route to status or create."""
        action: str = kwargs["action"]
        if action == "status":
            return self._status()
        if action == "create":
            return self._create(kwargs.get("topic", ""))
        return f"Unknown action: {action}"

    def _status(self) -> str:
        """Show current learn prompts."""
        logger.info("[inner_monologue] learn: status")
        prompts = self._db.learn_prompts.get_for_user(self._user)
        prompts = [lp for lp in prompts if lp.announced_at is None]
        if not prompts:
            return "No active learn prompts."
        lines = []
        for lp in prompts:
            status = "searching" if lp.status == "active" else lp.status
            lines.append(f"- {lp.prompt_text} ({status}, {lp.searches_remaining} left)")
        return "\n".join(lines)

    def _create(self, topic: str) -> str:
        """Create a new learn prompt."""
        if not topic:
            return "Topic is required for 'create' action."
        logger.info("[inner_monologue] learn: create '%s'", topic)
        self._db.learn_prompts.create(
            user=self._user,
            prompt_text=topic,
            searches_remaining=self._searches_remaining,
        )
        return f"Learn prompt created for '{topic}'."

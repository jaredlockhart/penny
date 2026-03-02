"""Think tool — record a persistent inner thought."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from penny.tools.base import Tool

if TYPE_CHECKING:
    from penny.database import Database

logger = logging.getLogger(__name__)


class ThinkTool(Tool):
    """Record a persistent inner thought."""

    name = "think"
    description = (
        "Record an internal thought. Use this to note observations, intentions, or reflections."
    )
    parameters = {
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": "Your thought",
            }
        },
        "required": ["thought"],
    }

    def __init__(self, db: Database, user: str):
        self._db = db
        self._user = user

    async def execute(self, **kwargs: Any) -> str:
        """Write a thought to the persistent log."""
        thought: str = kwargs["thought"]
        self._db.thoughts.add(self._user, thought)
        logger.info("[inner_monologue] think: %s", thought)
        return "Thought recorded."

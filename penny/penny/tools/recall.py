"""Recall tool — read from Penny's internal memory stores."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from penny.tools.base import Tool

if TYPE_CHECKING:
    from penny.database import Database

logger = logging.getLogger(__name__)


class RecallTool(Tool):
    """Read from Penny's internal memory stores."""

    name = "recall"
    description = (
        "Read from your memory. Choose what to recall: "
        "'messages' (recent conversations), "
        "'knowledge' (entities and facts you know about), "
        "or 'events' (recent news events)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "what": {
                "type": "string",
                "enum": ["messages", "knowledge", "events"],
                "description": "What to recall",
            }
        },
        "required": ["what"],
    }

    def __init__(self, db: Database, user: str):
        self._db = db
        self._user = user

    async def execute(self, **kwargs: Any) -> str:
        """Retrieve formatted data from the requested memory store."""
        what: str = kwargs["what"]
        logger.info("[inner_monologue] recall: %s", what)
        if what == "messages":
            return self._recall_messages()
        if what == "knowledge":
            return self._recall_knowledge()
        if what == "events":
            return self._recall_events()
        return f"Unknown recall type: {what}"

    def _recall_messages(self) -> str:
        """Format recent conversation messages (both directions) as readable text."""
        messages = self._db.messages.get_conversation(self._user, limit=20)
        if not messages:
            return "No recent messages."
        lines = []
        for msg in messages:
            ts = msg.timestamp.strftime("%Y-%m-%d %H:%M")
            direction = "User" if msg.direction == "incoming" else "Penny"
            lines.append(f"[{ts}] {direction}: {msg.content}")
        return "\n".join(lines)

    def _recall_knowledge(self) -> str:
        """Format top entities by recency with their facts."""
        entities = self._db.entities.get_for_user(self._user)
        if not entities:
            return "No known entities."
        sorted_entities = sorted(entities, key=lambda e: e.created_at, reverse=True)[:20]
        lines = []
        for entity in sorted_entities:
            assert entity.id is not None
            label = f"{entity.name} ({entity.tagline})" if entity.tagline else entity.name
            facts = self._db.facts.get_for_entity(entity.id)
            fact_text = "; ".join(f.content for f in facts[:5])
            lines.append(f"- {label}: {fact_text}" if fact_text else f"- {label}")
        return "\n".join(lines)

    def _recall_events(self) -> str:
        """Format recent events as readable text."""
        events = self._db.events.get_recent(self._user, days=7)
        if not events:
            return "No recent events."
        lines = []
        for event in events[:15]:
            ts = event.occurred_at.strftime("%Y-%m-%d")
            line = f"[{ts}] {event.headline}"
            if event.summary:
                line += f" — {event.summary[:100]}"
            if event.source_url:
                line += f" ({event.source_url})"
            lines.append(line)
        return "\n".join(lines)

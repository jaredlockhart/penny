"""Read emails tool — read full email content by ID."""

from __future__ import annotations

import logging
from typing import Any

from penny.email.protocol import EmailClient
from penny.tools.base import Tool
from penny.tools.ollama import OllamaClient
from penny.tools.ollama import OllamaClient

logger = logging.getLogger(__name__)

NO_EMAILS_TO_READ = "No email IDs provided."


class ReadEmailsTool(Tool):
    """Read the full body of one or more emails by ID."""

    name = "read_emails"
    description = (
        "Read the full content of one or more emails by their IDs. "
        "Use this after search_emails to get the complete bodies of relevant emails."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "email_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of email IDs from search_emails results",
            },
        },
        "required": ["email_ids"],
    }

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Reading emails"

    def __init__(
        self,
        email_client: EmailClient,
        ollama_client: OllamaClient,
        user_query: str,
    ) -> None:
        self._client = email_client
        self._ollama_client = ollama_client
        self._user_query = user_query

    async def execute(self, **kwargs: Any) -> str:
        """Read emails and return their content."""
        email_ids = kwargs["email_ids"]
        """Read emails and return their content."""
        email_ids = kwargs["email_ids"]
        if not email_ids:
            return NO_EMAILS_TO_READ
        emails = await self._client.read_emails(email_ids)
        if not emails:
            return NO_EMAILS_TO_READ

        # Summarize emails using Ollama
        summary = await self._ollama_client.summarize_emails(emails, self._user_query)
        return summary

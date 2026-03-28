"""Read emails tool — read full email content by ID."""

from __future__ import annotations

import logging
from typing import Any

from penny.email.protocol import EmailClient
from penny.tools.base import Tool

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

    def __init__(self, email_client: EmailClient) -> None:
        self._client = email_client

    async def execute(self, **kwargs: Any) -> str:
        """Read emails and return their content."""
        email_ids = kwargs["email_ids"]
        if not email_ids:
            return NO_EMAILS_TO_READ
        emails = await self._client.read_emails(email_ids)
        if not emails:
            return NO_EMAILS_TO_READ

        return "\n\n---\n\n".join(str(e) for e in emails)

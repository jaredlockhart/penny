"""Email search tools using Fastmail JMAP."""

from __future__ import annotations

import logging
from typing import Any

from penny.jmap.client import JmapClient
from penny.ollama.client import OllamaClient
from penny.prompts import Prompt
from penny.tools.base import Tool

logger = logging.getLogger(__name__)

# Tool response constants
NO_EMAILS_FOUND = "No emails found matching that query."
NO_EMAILS_TO_READ = "No email IDs provided."


class SearchEmailsTool(Tool):
    """Search emails by text, sender, subject, or date range."""

    name = "search_emails"
    description = (
        "Search the user's email inbox. Returns a list of matching email summaries "
        "with IDs, subjects, senders, dates, and previews. "
        "Use this to find relevant emails, then use read_emails to get full details."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Full-text search query across subject, body, and sender",
            },
            "from_addr": {
                "type": "string",
                "description": "Filter by sender email address or name",
            },
            "subject": {
                "type": "string",
                "description": "Filter by subject line text",
            },
            "after": {
                "type": "string",
                "description": (
                    "Only emails after this date (ISO 8601, e.g., 2026-01-01T00:00:00Z)"
                ),
            },
            "before": {
                "type": "string",
                "description": "Only emails before this date (ISO 8601)",
            },
        },
        "required": [],
    }

    def __init__(self, jmap_client: JmapClient) -> None:
        self._jmap = jmap_client

    async def execute(self, **kwargs: Any) -> str:
        """Search emails and return formatted summaries."""
        results = await self._jmap.search_emails(**kwargs)
        if not results:
            return NO_EMAILS_FOUND
        header = f"Found {len(results)} email(s):\n\n"
        return header + "\n\n".join(str(r) for r in results)


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

    def __init__(
        self,
        jmap_client: JmapClient,
        ollama_client: OllamaClient,
        user_query: str,
    ) -> None:
        self._jmap = jmap_client
        self._ollama = ollama_client
        self._user_query = user_query

    async def execute(self, **kwargs: Any) -> str:
        """Read emails and summarize relevant content."""
        email_ids = kwargs["email_ids"]
        if not email_ids:
            return NO_EMAILS_TO_READ
        emails = await self._jmap.read_emails(email_ids)
        if not emails:
            return NO_EMAILS_TO_READ

        raw_content = "\n\n---\n\n".join(str(e) for e in emails)
        prompt = Prompt.EMAIL_SUMMARIZE_PROMPT.format(
            query=self._user_query,
            emails=raw_content,
        )
        response = await self._ollama.chat([{"role": "user", "content": prompt}])
        return response.content or raw_content

"""Zoho Mail tools — LLM-callable tools for the Zoho plugin."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from penny.tools.base import Tool
from penny.tools.models import DraftEmailArgs, ListEmailsArgs

if TYPE_CHECKING:
    from penny.plugins.zoho.client import ZohoClient

logger = logging.getLogger(__name__)

NO_EMAILS_FOUND = "No emails found in that folder."


class ListEmailsTool(Tool):
    """List emails from a specific folder."""

    name = "list_emails"
    description = (
        "List emails from a specific folder in the user's mailbox. "
        "Available folders include: Inbox, Sent, Drafts, Trash, Spam. "
        "Returns email summaries with IDs, subjects, senders, dates, and previews. "
        "Use this to browse a folder, then use read_emails to get full details."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "folder": {
                "type": "string",
                "description": (
                    "Name of the folder to list emails from. "
                    "Common folders: Inbox, Sent, Drafts, Trash, Spam. "
                    "Defaults to Inbox if not specified."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of emails to return (default: 10, max: 50)",
            },
        },
        "required": [],
    }

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Listing emails"

    def __init__(self, zoho_client: ZohoClient) -> None:
        self._client = zoho_client

    async def execute(self, **kwargs: Any) -> str:
        """List emails from a folder and return formatted summaries."""
        args = ListEmailsArgs(**kwargs)
        folder = args.folder
        limit = min(args.limit, 50)

        results = await self._client.list_emails(folder_name=folder, limit=limit)
        if not results:
            return NO_EMAILS_FOUND

        folder_name = folder or "Inbox"
        header = f"Found {len(results)} email(s) in {folder_name}:\n\n"
        return header + "\n\n".join(str(r) for r in results)


class ListFoldersTool(Tool):
    """List available email folders."""

    name = "list_folders"
    description = (
        "List all available email folders in the user's mailbox. "
        "Returns folder names and types (Inbox, Sent, Drafts, etc.). "
        "Use this to discover what folders exist before listing emails from them."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Listing folders"

    def __init__(self, zoho_client: ZohoClient) -> None:
        self._client = zoho_client

    async def execute(self, **kwargs: Any) -> str:
        """List all folders and return formatted list."""
        folders = await self._client.get_folders()
        if not folders:
            return "No folders found."

        lines = [f"Found {len(folders)} folder(s):\n"]
        for folder in folders:
            lines.append(f"- {folder.folder_name} ({folder.folder_type})")
        return "\n".join(lines)


class DraftEmailTool(Tool):
    """Compose and save an email draft for user review."""

    name = "draft_email"
    description = (
        "Compose an email and save it as a draft for the user to review before sending. "
        "The draft will be saved to the Drafts folder where the user can edit and send it. "
        "Use this after reading emails to compose responses."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "to": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of recipient email addresses",
            },
            "subject": {
                "type": "string",
                "description": "Email subject line",
            },
            "body": {
                "type": "string",
                "description": "Email body content (plain text)",
            },
            "cc": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of CC recipient email addresses",
            },
        },
        "required": ["to", "subject", "body"],
    }

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Drafting email"

    def __init__(self, zoho_client: ZohoClient) -> None:
        self._client = zoho_client

    async def execute(self, **kwargs: Any) -> str:
        """Save an email draft and return confirmation."""
        args = DraftEmailArgs(**kwargs)
        try:
            message_id = await self._client.draft_response(
                to_addresses=args.to,
                subject=args.subject,
                content=args.body,
                cc_addresses=args.cc,
            )
            if message_id:
                recipients = ", ".join(args.to)
                return (
                    f"Draft saved successfully!\n\n"
                    f"To: {recipients}\n"
                    f"Subject: {args.subject}\n\n"
                    f"The draft has been saved to your Drafts folder for review before sending."
                )
            return "Draft was saved but could not confirm the message ID."
        except Exception as e:
            logger.exception("Failed to save draft")
            return f"Error saving draft: {e}"

"""Zoho Mail tools — LLM-callable tools for the Zoho plugin."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from penny.tools.base import Tool
from penny.tools.models import DraftEmailArgs, ListEmailsArgs

if TYPE_CHECKING:
    from sqlmodel import Session as DBSession

    from penny.plugins.zoho.client import ZohoClient

logger = logging.getLogger(__name__)

NO_EMAILS_FOUND = "No emails found in that folder."


class MoveEmailsArgs(BaseModel):
    """Arguments for moving emails to a folder."""

    message_ids: list[str] = Field(description="List of email message IDs to move")
    folder_path: str = Field(description="Destination folder path (e.g., 'Clients/John Smith')")
    create_if_missing: bool = Field(default=True, description="Create folder if it doesn't exist")


class CreateFolderArgs(BaseModel):
    """Arguments for creating an email folder."""

    folder_path: str = Field(description="Folder path to create (e.g., 'Accounting/Expenses/AWS')")


class ApplyLabelArgs(BaseModel):
    """Arguments for applying a label to emails."""

    message_ids: list[str] = Field(description="List of email message IDs to label")
    label_name: str = Field(description="Label name to apply (e.g., 'completed')")
    create_if_missing: bool = Field(default=True, description="Create label if it doesn't exist")


class CreateEmailRuleArgs(BaseModel):
    """Arguments for creating an email rule."""

    name: str = Field(description="Human-readable rule name")
    condition: dict = Field(description="Rule condition (from, subject_contains, etc.)")
    action: dict = Field(description="Rule action (move_to, label, etc.)")


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


class MoveEmailsTool(Tool):
    """Move emails to a folder, creating the folder if needed."""

    name = "move_emails"
    description = (
        "Move one or more emails to a destination folder. "
        "The folder path can include nested folders like 'Clients/John Smith' or "
        "'Accounting/Expenses/AWS'. Folders will be created if they don't exist. "
        "Use this after reading emails to organize them into appropriate folders."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "message_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of email message IDs to move",
            },
            "folder_path": {
                "type": "string",
                "description": (
                    "Destination folder path. Can be nested like 'Clients/John Smith' "
                    "or 'Accounting/Expenses/AWS'"
                ),
            },
            "create_if_missing": {
                "type": "boolean",
                "description": "Create the folder if it doesn't exist (default: true)",
            },
        },
        "required": ["message_ids", "folder_path"],
    }

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Moving emails"

    def __init__(self, zoho_client: ZohoClient) -> None:
        self._client = zoho_client

    async def execute(self, **kwargs: Any) -> str:
        """Move emails to a folder."""
        args = MoveEmailsArgs(**kwargs)

        if not args.message_ids:
            return "No message IDs provided."

        # Find or create the destination folder
        folder = await self._client.get_folder_by_name(args.folder_path.split("/")[-1])

        if not folder and args.create_if_missing:
            folder = await self._client.create_nested_folder(args.folder_path)
            if not folder:
                return f"Failed to create folder: {args.folder_path}"

        if not folder:
            return f"Folder not found: {args.folder_path}"

        # Move the messages
        success = await self._client.move_messages(args.message_ids, folder.folder_id)
        if success:
            return f"Successfully moved {len(args.message_ids)} email(s) to '{args.folder_path}'"
        return f"Failed to move emails to '{args.folder_path}'"


class CreateFolderTool(Tool):
    """Create an email folder with optional nesting."""

    name = "create_folder"
    description = (
        "Create a new email folder. Supports nested folder paths like "
        "'Clients/John Smith' or 'Accounting/Expenses/AWS'. "
        "Parent folders will be created automatically if they don't exist."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "folder_path": {
                "type": "string",
                "description": ("Folder path to create. Can be nested like 'Clients/John Smith'"),
            },
        },
        "required": ["folder_path"],
    }

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Creating folder"

    def __init__(self, zoho_client: ZohoClient) -> None:
        self._client = zoho_client

    async def execute(self, **kwargs: Any) -> str:
        """Create a folder."""
        args = CreateFolderArgs(**kwargs)

        folder = await self._client.create_nested_folder(args.folder_path)
        if folder:
            return f"Successfully created folder: {args.folder_path}"
        return f"Failed to create folder: {args.folder_path}"


class ApplyLabelTool(Tool):
    """Apply a label to emails."""

    name = "apply_label"
    description = (
        "Apply a label to one or more emails. Labels help categorize emails "
        "without moving them. Common labels include 'completed', 'pending', "
        "'urgent', etc. The label will be created if it doesn't exist."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "message_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of email message IDs to label",
            },
            "label_name": {
                "type": "string",
                "description": "Label name to apply (e.g., 'completed', 'pending')",
            },
            "create_if_missing": {
                "type": "boolean",
                "description": "Create the label if it doesn't exist (default: true)",
            },
        },
        "required": ["message_ids", "label_name"],
    }

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Applying label"

    def __init__(self, zoho_client: ZohoClient) -> None:
        self._client = zoho_client

    async def execute(self, **kwargs: Any) -> str:
        """Apply a label to emails."""
        args = ApplyLabelArgs(**kwargs)

        if not args.message_ids:
            return "No message IDs provided."

        # Find or create the label
        label = await self._client.get_label_by_name(args.label_name)

        if not label and args.create_if_missing:
            label = await self._client.create_label(args.label_name)
            if not label:
                return f"Failed to create label: {args.label_name}"

        if not label:
            return f"Label not found: {args.label_name}"

        label_id = label.get("labelId", "")
        success = await self._client.apply_label(args.message_ids, label_id)
        if success:
            return (
                f"Successfully applied label '{args.label_name}' to "
                f"{len(args.message_ids)} email(s)"
            )
        return f"Failed to apply label '{args.label_name}'"


class ListLabelsTool(Tool):
    """List available email labels."""

    name = "list_labels"
    description = (
        "List all available email labels in the user's mailbox. "
        "Returns label names and colors. Use this to see what labels exist "
        "before applying them to emails."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Listing labels"

    def __init__(self, zoho_client: ZohoClient) -> None:
        self._client = zoho_client

    async def execute(self, **kwargs: Any) -> str:
        """List all labels."""
        labels = await self._client.get_labels()
        if not labels:
            return "No labels found."

        lines = [f"Found {len(labels)} label(s):\n"]
        for label in labels:
            name = label.get("displayName", "Unknown")
            color = label.get("color", "")
            lines.append(f"- {name} ({color})")
        return "\n".join(lines)


class CreateEmailRuleTool(Tool):
    """Create a persistent email rule for automatic organization."""

    name = "create_email_rule"
    description = (
        "Create a persistent email rule that will be automatically applied "
        "during scheduled email checks. Rules can match emails by sender, "
        "subject, or content, and can move emails to folders or apply labels. "
        "Example: Create a rule to move all emails from AWS to Accounting/Expenses/AWS."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Human-readable rule name (e.g., 'AWS invoices to expenses')",
            },
            "condition": {
                "type": "object",
                "description": (
                    "Rule condition. Supported fields: 'from' (sender email/domain), "
                    "'subject_contains' (text in subject), 'body_contains' (text in body)"
                ),
            },
            "action": {
                "type": "object",
                "description": (
                    "Rule action. Supported fields: 'move_to' (folder path), "
                    "'label' (label name to apply)"
                ),
            },
        },
        "required": ["name", "condition", "action"],
    }

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Creating email rule"

    def __init__(self, db_session: DBSession, user_id: str) -> None:
        self._db = db_session
        self._user_id = user_id

    async def execute(self, **kwargs: Any) -> str:
        """Create an email rule."""
        from datetime import UTC, datetime

        from penny.database.models import EmailRule

        args = CreateEmailRuleArgs(**kwargs)

        rule = EmailRule(
            user_id=self._user_id,
            provider="zoho",
            name=args.name,
            condition=json.dumps(args.condition),
            action=json.dumps(args.action),
            enabled=True,
            created_at=datetime.now(UTC),
        )
        self._db.add(rule)
        self._db.commit()

        return (
            f"Email rule '{args.name}' created successfully.\n\n"
            f"Condition: {args.condition}\n"
            f"Action: {args.action}\n\n"
            "This rule will be applied automatically during scheduled email checks."
        )


class ListEmailRulesTool(Tool):
    """List all active email rules."""

    name = "list_email_rules"
    description = (
        "List all active email rules that are applied during scheduled email checks. "
        "Shows rule names, conditions, and actions."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Listing email rules"

    def __init__(self, db_session: DBSession, user_id: str) -> None:
        self._db = db_session
        self._user_id = user_id

    async def execute(self, **kwargs: Any) -> str:
        """List all email rules."""
        from sqlmodel import select

        from penny.database.models import EmailRule

        rules = list(
            self._db.exec(
                select(EmailRule)
                .where(EmailRule.user_id == self._user_id)
                .where(EmailRule.provider == "zoho")
                .where(EmailRule.enabled == True)  # noqa: E712
            )
        )

        if not rules:
            return "No email rules configured."

        lines = [f"Found {len(rules)} active email rule(s):\n"]
        for idx, rule in enumerate(rules, start=1):
            condition = rule.get_condition()
            action = rule.get_action()
            lines.append(f"{idx}. **{rule.name}**")
            lines.append(f"   Condition: {condition}")
            lines.append(f"   Action: {action}")
            lines.append("")

        return "\n".join(lines)

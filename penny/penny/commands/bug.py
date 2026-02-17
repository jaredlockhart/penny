"""GitHub bug filing command."""

import logging
from datetime import UTC, datetime

from github_api.api import GitHubAPI

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.responses import BUG_ERROR, BUG_FILED, BUG_USAGE

logger = logging.getLogger(__name__)


class BugCommand(Command):
    """File a bug report on GitHub."""

    name = "bug"
    description = "File a bug report on GitHub"
    help_text = (
        "Usage: /bug <description>\n\n"
        "Files a bug report in the penny repository. "
        "The first ~60 characters will be used as the title."
    )

    def __init__(self, github_api: GitHubAPI):
        self._github_api = github_api

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute the bug command."""
        description = args.strip()

        if not description:
            return CommandResult(text=BUG_USAGE)

        # Generate title: first 60 chars at word boundary
        title = self._generate_title(description)

        # Generate body with metadata footer
        body = self._generate_body(description, context)

        try:
            issue_url = self._github_api.create_issue(
                title=title,
                body=body,
                labels=["bug"],
            )
            return CommandResult(text=BUG_FILED.format(issue_url=issue_url))

        except Exception as e:
            logger.exception("Failed to create GitHub issue")
            return CommandResult(text=BUG_ERROR.format(error=e))

    def _generate_title(self, description: str) -> str:
        """Extract title from description (first 60 chars at word boundary)."""
        if len(description) <= 60:
            return description

        # Truncate at last word boundary before 60 chars
        truncated = description[:60]
        last_space = truncated.rfind(" ")
        if last_space > 0:
            return f"{truncated[:last_space]}..."
        return f"{truncated}..."

    def _generate_body(self, description: str, context: CommandContext) -> str:
        """Generate issue body with description and metadata footer."""
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Build footer with filing metadata
        footer_parts = [
            "\n\n---",
            f"Filed via {context.channel_type} at {timestamp}",
        ]

        # If this is a quote-reply, include metadata about the quoted message
        if context.message and context.message.quoted_text:
            quoted_msg = context.db.find_outgoing_by_content(context.message.quoted_text)
            if quoted_msg:
                quoted_timestamp = quoted_msg.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
                footer_parts.append(f"Refers to message sent at {quoted_timestamp}")

        footer = "\n".join(footer_parts)
        return description + footer

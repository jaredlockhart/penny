"""Base class for commands that file GitHub issues."""

import logging
from abc import abstractmethod
from datetime import UTC, datetime

from github_api.api import GitHubAPI

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult

logger = logging.getLogger(__name__)


class GitHubIssueCommand(Command):
    """Base class for commands that create GitHub issues with a specific label."""

    @property
    @abstractmethod
    def labels(self) -> list[str]:
        """GitHub labels to apply to the created issue."""
        pass

    @property
    @abstractmethod
    def usage_response(self) -> str:
        """Response string when no description is provided."""
        pass

    @property
    @abstractmethod
    def filed_response(self) -> str:
        """Response string template on success (must contain {issue_url})."""
        pass

    @property
    @abstractmethod
    def error_response(self) -> str:
        """Response string template on failure (must contain {error})."""
        pass

    def __init__(self, github_api: GitHubAPI):
        self._github_api = github_api

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute the command: create a GitHub issue from the description."""
        description = args.strip()

        if not description:
            return CommandResult(text=self.usage_response)

        title = self._generate_title(description)
        body = self._generate_body(description, context)

        try:
            issue_url = self._github_api.create_issue(
                title=title,
                body=body,
                labels=self.labels,
            )
            return CommandResult(text=self.filed_response.format(issue_url=issue_url))

        except Exception as e:
            logger.exception("Failed to create GitHub issue")
            return CommandResult(text=self.error_response.format(error=e))

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

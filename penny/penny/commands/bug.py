"""GitHub bug filing command."""

import logging
from datetime import UTC, datetime

from github_api.api import GitHubAPI

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult

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
            return CommandResult(text="Please provide a bug description. Usage: /bug <description>")

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
            return CommandResult(text=f"Bug filed! {issue_url}")

        except Exception as e:
            logger.exception("Failed to create GitHub issue")
            return CommandResult(text=f"Failed to create issue: {e!s}")

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
        footer = f"\n\n---\nFiled by {context.user} via {context.channel_type} at {timestamp}"
        return description + footer

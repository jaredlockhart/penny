"""GitHub bug filing command."""

from penny.commands.github_issue import GitHubIssueCommand
from penny.responses import PennyResponse


class BugCommand(GitHubIssueCommand):
    """File a bug report on GitHub."""

    name = "bug"
    description = "File a bug report on GitHub"
    help_text = (
        "Usage: /bug <description>\n\n"
        "Files a bug report in the penny repository. "
        "The first ~60 characters will be used as the title."
    )
    labels = ["bug"]
    usage_response = PennyResponse.BUG_USAGE
    filed_response = PennyResponse.BUG_FILED
    error_response = PennyResponse.BUG_ERROR

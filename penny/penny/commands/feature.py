"""GitHub feature request filing command."""

from penny.commands.github_issue import GitHubIssueCommand
from penny.responses import PennyResponse


class FeatureCommand(GitHubIssueCommand):
    """File a feature request on GitHub."""

    name = "feature"
    description = "File a feature request on GitHub"
    help_text = (
        "Usage: /feature <description>\n\n"
        "Files a feature request in the penny repository. "
        "The first ~60 characters will be used as the title."
    )
    labels = ["requirements"]
    usage_response = PennyResponse.FEATURE_USAGE
    filed_response = PennyResponse.FEATURE_FILED
    error_response = PennyResponse.FEATURE_ERROR

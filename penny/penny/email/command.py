"""Multi-provider /email command with plugin routing.

Routes /email queries to whichever email plugin(s) are enabled:
- Single email plugin:  /email <query>
- Multiple plugins:     /email <provider> <query>   e.g. /email zoho <query>

Future: /email <provider>:<account> <query>  e.g. /email zoho:business <query>
"""

from __future__ import annotations

import logging

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.plugins import Plugin
from penny.responses import PennyResponse

logger = logging.getLogger(__name__)


class EmailCommand(Command):
    """Unified email command that routes to the appropriate email plugin."""

    name = "email"
    description = "Search your email and answer questions"
    help_text = (
        "Usage: /email <question>\n\n"
        "Ask a question about your email and Penny will search and read "
        "relevant messages to find the answer.\n\n"
        "If multiple email providers are configured, specify the provider:\n"
        "  /email zoho <question>\n"
        "  /email fastmail <question>\n\n"
        "Examples:\n"
        "• /email what packages am I expecting\n"
        "• /email zoho when is my dentist appointment\n"
        "• /email fastmail any emails from mom this week"
    )

    def __init__(self, email_plugins: list[Plugin]) -> None:
        self._email_plugins = email_plugins

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Route the query to the appropriate email plugin."""
        query = args.strip()
        if not query:
            return CommandResult(text=PennyResponse.EMAIL_NO_QUERY_TEXT)

        plugin, query = self._resolve_provider(query)
        if plugin is None:
            return CommandResult(text=self._multi_provider_error())

        return await plugin.get_commands()[0].execute(query, context)

    def _resolve_provider(self, query: str) -> tuple[Plugin | None, str]:
        """Determine which plugin handles the query and strip the provider prefix.

        Returns (plugin, cleaned_query). Returns (None, query) when provider
        disambiguation is required but not supplied.
        """
        if len(self._email_plugins) == 1:
            return self._email_plugins[0], query

        first_word, _, rest = query.partition(" ")
        provider_name = first_word.split(":")[0].lower()

        for plugin in self._email_plugins:
            if plugin.name == provider_name:
                return plugin, rest.strip()

        return None, query

    def _multi_provider_error(self) -> str:
        """Build the disambiguation error message."""
        provider_names = [p.name for p in self._email_plugins]
        examples = "  ".join(f"/email {n} <question>" for n in provider_names)
        return (
            f"Multiple email providers are enabled: {', '.join(provider_names)}.\n\n"
            f"Please specify the provider:\n  {examples}"
        )

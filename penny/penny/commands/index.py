"""The /commands command — lists all commands or shows help for a specific one."""

from __future__ import annotations

from typing import TYPE_CHECKING

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.responses import PennyResponse

if TYPE_CHECKING:
    from penny.commands.base import CommandRegistry


class IndexCommand(Command):
    """Lists all commands or shows help for a specific command."""

    name = "commands"
    description = "List all commands or get help for a specific command"
    help_text = (
        "Lists all available commands with their descriptions, or shows detailed help "
        "for a specific command.\n\n"
        "**Usage**:\n"
        "- `/commands` — List all available commands\n"
        "- `/commands <command>` — Show detailed help for a specific command"
    )

    def __init__(self, registry: CommandRegistry):
        """Initialize with reference to command registry."""
        self._registry = registry

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute commands command."""
        args = args.strip()

        # If no args, list all commands
        if not args:
            commands = self._registry.list_all()
            lines = [PennyResponse.COMMANDS_HEADER, ""]
            for cmd in sorted(commands, key=lambda c: c.name):
                lines.append(f"- **/{cmd.name}** — {cmd.description}")
            return CommandResult(text="\n".join(lines))

        # Otherwise, show help for specific command
        cmd = self._registry.get(args)
        if not cmd:
            return CommandResult(text=PennyResponse.COMMANDS_UNKNOWN.format(name=args))

        lines = [
            PennyResponse.COMMANDS_HELP_HEADER.format(name=cmd.name),
            "",
            cmd.help_text,
        ]
        return CommandResult(text="\n".join(lines))

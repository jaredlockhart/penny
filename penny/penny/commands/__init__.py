"""Command system for Penny."""

from penny.commands.base import Command, CommandRegistry
from penny.commands.builtin import CommandsCommand, DebugCommand
from penny.commands.models import CommandContext, CommandError, CommandResult

__all__ = [
    "Command",
    "CommandRegistry",
    "CommandContext",
    "CommandResult",
    "CommandError",
    "create_command_registry",
]


def create_command_registry() -> CommandRegistry:
    """
    Factory to create registry with builtin commands.

    Returns:
        CommandRegistry with all builtin commands registered
    """
    registry = CommandRegistry()

    # Register CommandsCommand with self-reference for listing commands
    commands_cmd = CommandsCommand(registry)
    registry.register(commands_cmd)

    # Register other builtin commands
    registry.register(DebugCommand())

    return registry

"""Command system for Penny."""

from collections.abc import Callable

from penny.commands.base import Command, CommandRegistry
from penny.commands.config import ConfigCommand
from penny.commands.debug import DebugCommand
from penny.commands.index import IndexCommand
from penny.commands.models import CommandContext, CommandError, CommandResult
from penny.commands.preferences import DislikeCommand, LikeCommand, UndislikeCommand, UnlikeCommand
from penny.commands.profile import ProfileCommand
from penny.commands.schedule import ScheduleCommand
from penny.commands.test import TestCommand

__all__ = [
    "Command",
    "CommandRegistry",
    "CommandContext",
    "CommandResult",
    "CommandError",
    "create_command_registry",
]


def create_command_registry(
    message_agent_factory: Callable | None = None,
) -> CommandRegistry:
    """
    Factory to create registry with builtin commands.

    Args:
        message_agent_factory: Optional factory for creating MessageAgent instances
                              (required for test command)

    Returns:
        CommandRegistry with all builtin commands registered
    """
    registry = CommandRegistry()

    # Register IndexCommand with self-reference for listing commands
    commands_cmd = IndexCommand(registry)
    registry.register(commands_cmd)

    # Register other builtin commands
    registry.register(DebugCommand())
    registry.register(ConfigCommand())
    registry.register(ProfileCommand())
    registry.register(ScheduleCommand())
    registry.register(LikeCommand())
    registry.register(DislikeCommand())
    registry.register(UnlikeCommand())
    registry.register(UndislikeCommand())

    # Register test command if factory provided
    if message_agent_factory:
        registry.register(TestCommand(message_agent_factory))

    return registry

"""Command system for Penny."""

from collections.abc import Callable
from typing import TYPE_CHECKING

from penny.commands.base import Command, CommandRegistry
from penny.commands.config import ConfigCommand
from penny.commands.debug import DebugCommand
from penny.commands.index import IndexCommand
from penny.commands.models import CommandContext, CommandError, CommandResult
from penny.commands.preferences import DislikeCommand, LikeCommand, UndislikeCommand, UnlikeCommand
from penny.commands.profile import ProfileCommand
from penny.commands.research import ResearchCommand
from penny.commands.schedule import ScheduleCommand
from penny.commands.test import TestCommand

if TYPE_CHECKING:
    from github_api.api import GitHubAPI

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
    github_api: "GitHubAPI | None" = None,
) -> CommandRegistry:
    """
    Factory to create registry with builtin commands.

    Args:
        message_agent_factory: Optional factory for creating MessageAgent instances
                              (required for test command)
        github_api: Optional GitHub API client (required for bug command)

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
    registry.register(ResearchCommand())
    registry.register(LikeCommand())
    registry.register(DislikeCommand())
    registry.register(UnlikeCommand())
    registry.register(UndislikeCommand())

    # Register test command if factory provided
    if message_agent_factory:
        registry.register(TestCommand(message_agent_factory))

    # Register bug command if GitHub API is configured
    if github_api:
        from penny.commands.bug import BugCommand

        registry.register(BugCommand(github_api))

    return registry

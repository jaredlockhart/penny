"""Command system for Penny."""

from collections.abc import Callable
from typing import TYPE_CHECKING

from penny.commands.base import Command, CommandRegistry
from penny.commands.config import ConfigCommand
from penny.commands.debug import DebugCommand
from penny.commands.index import IndexCommand
from penny.commands.interests import InterestsCommand
from penny.commands.learn import LearnCommand
from penny.commands.memory import MemoryCommand
from penny.commands.models import CommandContext, CommandError, CommandResult
from penny.commands.preferences import DislikeCommand, LikeCommand, UndislikeCommand, UnlikeCommand
from penny.commands.profile import ProfileCommand
from penny.commands.schedule import ScheduleCommand
from penny.commands.test import TestCommand

if TYPE_CHECKING:
    from github_api.api import GitHubAPI

    from penny.tools import Tool

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
    github_api: GitHubAPI | None = None,
    ollama_image_model: str | None = None,
    fastmail_api_token: str | None = None,
    search_tool: Tool | None = None,
) -> CommandRegistry:
    """
    Factory to create registry with builtin commands.

    Args:
        message_agent_factory: Optional factory for creating MessageAgent instances
                              (required for test command)
        github_api: Optional GitHub API client (required for bug command)
        ollama_image_model: Optional image generation model name (required for draw command)

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
    registry.register(MemoryCommand())
    registry.register(InterestsCommand())
    registry.register(LikeCommand())
    registry.register(DislikeCommand())
    registry.register(UnlikeCommand())
    registry.register(UndislikeCommand())
    registry.register(LearnCommand(search_tool))

    # Register test command if factory provided
    if message_agent_factory:
        registry.register(TestCommand(message_agent_factory))

    # Register bug and feature commands if GitHub API is configured
    if github_api:
        from penny.commands.bug import BugCommand
        from penny.commands.feature import FeatureCommand

        registry.register(BugCommand(github_api))
        registry.register(FeatureCommand(github_api))

    # Register draw command if image model is configured
    if ollama_image_model:
        from penny.commands.draw import DrawCommand

        registry.register(DrawCommand(ollama_image_model))

    # Register email command if Fastmail API token is configured
    if fastmail_api_token:
        from penny.commands.email import EmailCommand

        registry.register(EmailCommand(fastmail_api_token))

    return registry

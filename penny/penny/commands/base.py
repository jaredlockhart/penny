"""Base command abstraction and registry."""

from abc import ABC, abstractmethod

from penny.penny.commands.models import CommandContext, CommandResult


class Command(ABC):
    """Abstract base class for commands."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Command name (without / prefix)."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """One-sentence description for /commands list."""
        pass

    @property
    @abstractmethod
    def help_text(self) -> str:
        """Detailed help for /commands <name>."""
        pass

    @abstractmethod
    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """
        Execute command with arguments.

        Args:
            args: Command arguments (everything after the command name)
            context: Runtime context (db, config, user, etc.)

        Returns:
            CommandResult with response text
        """
        pass


class CommandRegistry:
    """Registry for all available commands."""

    def __init__(self):
        """Initialize empty registry."""
        self._commands: dict[str, Command] = {}

    def register(self, command: Command) -> None:
        """
        Register a command.

        Args:
            command: Command instance to register

        Raises:
            AssertionError: If command name is already registered
        """
        name = command.name.lower()
        assert name not in self._commands, f"Command '{name}' already registered"
        self._commands[name] = command

    def get(self, name: str) -> Command | None:
        """
        Get command by name (case-insensitive).

        Args:
            name: Command name to look up

        Returns:
            Command instance or None if not found
        """
        return self._commands.get(name.lower())

    def list_all(self) -> list[Command]:
        """
        Get all registered commands.

        Returns:
            List of all registered commands
        """
        return list(self._commands.values())

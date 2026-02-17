"""The /test command — executes a prompt in test mode with isolated database."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.constants import TEST_DB_PATH
from penny.responses import (
    TEST_ERROR,
    TEST_MODE_PREFIX,
    TEST_NESTED_ERROR,
    TEST_NO_RESPONSE,
    TEST_USAGE,
)

if TYPE_CHECKING:
    from penny.agents.message import MessageAgent

logger = logging.getLogger(__name__)


class TestCommand(Command):
    """Execute a prompt in test mode using an isolated test database."""

    name = "test"
    description = "Execute a prompt in test mode using an isolated test database"
    help_text = (
        "Executes a prompt through the full Penny flow (search, LLM, response) using "
        "an isolated test database snapshot. All database writes go to the test db only, "
        "leaving production conversation history untouched.\n\n"
        "**Usage**: `/test <prompt>`\n\n"
        "**Limitations**:\n"
        "- Threading/quote-replies are not supported\n"
        "- Nested commands (e.g., `/test /debug`) are not supported\n"
        "- Test database is snapshotted at startup and persists until container restart"
    )

    def __init__(self, message_agent_factory: Any) -> None:
        """
        Initialize with message agent factory.

        Args:
            message_agent_factory: Callable that creates a MessageAgent with a given database
        """
        self._message_agent_factory = message_agent_factory

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute test command."""
        prompt = args.strip()

        # Validate args
        if not prompt:
            return CommandResult(text=TEST_USAGE)

        # Reject nested commands
        if prompt.startswith("/"):
            return CommandResult(text=TEST_NESTED_ERROR)

        # Create test database instance
        from penny.database import Database

        test_db_path = Path(context.db.db_path).parent / TEST_DB_PATH.name
        test_db = Database(str(test_db_path))

        # Create message agent with test database
        test_agent: MessageAgent = self._message_agent_factory(test_db)

        try:
            # Execute agent with test database (no threading support)
            _, response = await test_agent.handle(
                content=prompt,
                sender=context.user,
                quoted_text=None,
            )

            # Prepend [TEST] to response
            answer = response.answer.strip() if response.answer else TEST_NO_RESPONSE
            return CommandResult(text=f"{TEST_MODE_PREFIX}{answer}")

        except Exception as e:
            logger.exception("Error executing test command: %s", e)
            return CommandResult(text=TEST_ERROR.format(error=e))

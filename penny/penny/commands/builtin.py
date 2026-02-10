"""Builtin command implementations."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    psutil: Any = None
    HAS_PSUTIL = False

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.constants import TEST_DB_PATH, TEST_MODE_PREFIX

if TYPE_CHECKING:
    from penny.agent.agents.message import MessageAgent
    from penny.commands.base import CommandRegistry

logger = logging.getLogger(__name__)


class CommandsCommand(Command):
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
            lines = ["**Available Commands**", ""]
            for cmd in sorted(commands, key=lambda c: c.name):
                lines.append(f"- **/{cmd.name}** — {cmd.description}")
            return CommandResult(text="\n".join(lines))

        # Otherwise, show help for specific command
        cmd = self._registry.get(args)
        if not cmd:
            return CommandResult(
                text=f"Unknown command: /{args}. Use /commands to see available commands."
            )

        lines = [
            f"**Command: /{cmd.name}**",
            "",
            cmd.help_text,
        ]
        return CommandResult(text="\n".join(lines))


class DebugCommand(Command):
    """Shows diagnostic information about Penny's current state."""

    name = "debug"
    description = "Show diagnostic information about Penny's current state"
    help_text = (
        "Shows diagnostic information including git commit, uptime, active channel, "
        "database stats, model versions, agent status, and memory usage.\n\n"
        "**Usage**: `/debug`"
    )

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute debug command."""
        # Git commit - try environment variable first (set at build time), then git command
        commit = os.environ.get("GIT_COMMIT")
        if not commit:
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                commit = result.stdout.strip() if result.returncode == 0 else "unknown"
            except Exception as e:
                logger.debug("Failed to get git commit: %s", e)
                commit = "unknown"

        # Uptime
        uptime = datetime.now() - context.start_time
        days = uptime.days
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        uptime_str = f"{days} days, {hours} hours, {minutes} minutes"

        # Database stats
        total_messages = context.db.count_messages()
        active_threads = context.db.count_active_threads()

        # Agent status (read from orchestrator state files)
        agent_status = self._get_agent_status()

        # Memory
        if not HAS_PSUTIL or psutil is None:
            mem_str = "unknown (psutil not installed)"
        else:
            try:
                process = psutil.Process()
                mem_mb = process.memory_info().rss / 1024 / 1024
                mem_percent = process.memory_percent()
                mem_str = f"{mem_mb:.0f} MB ({mem_percent:.1f}%)"
            except Exception as e:
                logger.warning("Failed to get memory info: %s", e)
                mem_str = "unknown"

        fg_model = context.config.ollama_foreground_model
        bg_model = context.config.ollama_background_model

        response = f"""**Debug Information**

**Git Commit**: {commit}
**Uptime**: {uptime_str}
**Channel**: {context.channel_type.title()}
**Database**: {total_messages:,} messages, {active_threads} active threads
**Models**: {fg_model} (foreground), {bg_model} (background)
**Agents**: {agent_status}
**Memory**: {mem_str}
"""
        return CommandResult(text=response)

    def _get_agent_status(self) -> str:
        """Get agent run status from state files."""
        # Agent state files are in /penny/data/penny-team (mounted from host data/)
        agents_dir = Path("/penny/data/penny-team")
        if not agents_dir.exists():
            return "unknown (no state directory)"

        agent_names = ["product-manager", "architect", "worker"]
        status_parts = []

        for agent_name in agent_names:
            state_file = agents_dir / f"{agent_name}.state.json"
            if not state_file.exists():
                status_parts.append(f"{agent_name}: never run")
                continue

            try:
                with open(state_file) as f:
                    state = json.load(f)
                last_run = state.get("last_run")
                if last_run:
                    # Parse ISO timestamp
                    last_run_dt = datetime.fromisoformat(last_run.replace("Z", "+00:00"))
                    ago = datetime.now(last_run_dt.tzinfo) - last_run_dt
                    if ago.total_seconds() < 60:
                        ago_str = f"{int(ago.total_seconds())}s ago"
                    elif ago.total_seconds() < 3600:
                        ago_str = f"{int(ago.total_seconds() / 60)}m ago"
                    else:
                        ago_str = f"{int(ago.total_seconds() / 3600)}h ago"
                    status_parts.append(f"{agent_name}: {ago_str}")
                else:
                    status_parts.append(f"{agent_name}: never run")
            except Exception as e:
                logger.warning("Failed to read state for %s: %s", agent_name, e)
                status_parts.append(f"{agent_name}: error")

        return ", ".join(status_parts)


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
            return CommandResult(text="Usage: /test <prompt>")

        # Reject nested commands
        if prompt.startswith("/"):
            return CommandResult(text="Nested commands are not supported in test mode.")

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
            answer = response.answer.strip() if response.answer else "No response generated."
            return CommandResult(text=f"{TEST_MODE_PREFIX}{answer}")

        except Exception as e:
            logger.exception("Error executing test command: %s", e)
            return CommandResult(text=f"Test mode error: {e!s}")

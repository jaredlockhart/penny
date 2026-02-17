"""The /debug command — shows diagnostic information about Penny."""

from __future__ import annotations

import logging
import os
import subprocess
from datetime import datetime
from typing import Any

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    psutil: Any = None
    HAS_PSUTIL = False

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.responses import PennyResponse

logger = logging.getLogger(__name__)


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
        uptime_str = PennyResponse.DEBUG_UPTIME.format(days=days, hours=hours, minutes=minutes)

        # Database stats
        total_messages = context.db.count_messages()
        active_threads = context.db.count_active_threads()

        # Background task status
        task_status = self._get_task_status(context)

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

        response = PennyResponse.DEBUG_TEMPLATE.format(
            commit=commit,
            uptime=uptime_str,
            channel=context.channel_type.title(),
            messages=total_messages,
            threads=active_threads,
            fg_model=fg_model,
            bg_model=bg_model,
            task_status=task_status,
            memory=mem_str,
        )
        return CommandResult(text=response)

    def _get_task_status(self, context: CommandContext) -> str:
        """Get background task run status from scheduler."""
        if context.scheduler is None:
            return PennyResponse.DEBUG_NO_SCHEDULER

        agent_status = context.scheduler.get_agent_status()
        status_parts = []

        for agent_name, seconds_ago in agent_status.items():
            if seconds_ago is None:
                status_parts.append(PennyResponse.DEBUG_TASK_NEVER.format(name=agent_name))
            elif seconds_ago < 60:
                status_parts.append(
                    PennyResponse.DEBUG_TASK_SECONDS.format(
                        name=agent_name, seconds=int(seconds_ago)
                    )
                )
            elif seconds_ago < 3600:
                status_parts.append(
                    PennyResponse.DEBUG_TASK_MINUTES.format(
                        name=agent_name, minutes=int(seconds_ago / 60)
                    )
                )
            else:
                status_parts.append(
                    PennyResponse.DEBUG_TASK_HOURS.format(
                        name=agent_name, hours=int(seconds_ago / 3600)
                    )
                )

        return ", ".join(status_parts)

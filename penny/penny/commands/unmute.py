"""The /unmute command — re-enable proactive notifications."""

from __future__ import annotations

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.responses import PennyResponse


class UnmuteCommand(Command):
    """Unmute proactive notifications."""

    name = "unmute"
    description = "Unmute proactive notifications"
    help_text = (
        "Re-enable proactive notifications after muting them with /mute.\n\n"
        "**Usage**:\n"
        "• `/unmute` — Unmute notifications"
    )

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute unmute command."""
        if not context.db.users.is_muted(context.user):
            return CommandResult(text=PennyResponse.UNMUTE_ALREADY)

        context.db.users.set_unmuted(context.user)
        return CommandResult(text=PennyResponse.UNMUTE_ENABLED)

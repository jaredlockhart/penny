"""The /mute command — silence proactive notifications."""

from __future__ import annotations

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.responses import PennyResponse


class MuteCommand(Command):
    """Mute proactive notifications."""

    name = "mute"
    description = "Mute proactive notifications"
    help_text = (
        "Mute proactive notifications like fact discoveries and learn completion "
        "announcements. Scheduled tasks and replies to your messages are not affected.\n\n"
        "**Usage**:\n"
        "• `/mute` — Mute notifications\n"
        "• `/unmute` — Unmute notifications"
    )

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute mute command."""
        if context.db.users.is_muted(context.user):
            return CommandResult(text=PennyResponse.MUTE_ALREADY)

        context.db.users.set_muted(context.user)
        return CommandResult(text=PennyResponse.MUTE_ENABLED)

"""The /style command — view and manage adaptive speaking style."""

from __future__ import annotations

from penny.agents.style import AdaptiveStyleAgent
from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult


class StyleCommand(Command):
    """View and manage adaptive speaking style."""

    name = "style"
    description = "View and manage adaptive speaking style"
    help_text = (
        "View and manage how Penny adapts to match your speaking style.\n\n"
        "**Usage**:\n"
        "- `/style` — View your current style profile\n"
        "- `/style reset` — Clear and regenerate your style profile from current messages\n"
        "- `/style off` — Disable adaptive style\n"
        "- `/style on` — Re-enable adaptive style"
    )

    def __init__(self, style_agent: AdaptiveStyleAgent):
        """
        Initialize style command.

        Args:
            style_agent: The adaptive style agent instance
        """
        self._style_agent = style_agent

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute style command."""
        subcommand = args.strip().lower()

        # Case 1: View current style profile
        if not subcommand:
            profile = context.db.get_user_style_profile(context.user)
            if not profile:
                total_messages = context.db.count_user_messages(context.user)
                if total_messages < 20:
                    return CommandResult(
                        text=(
                            f"No style profile yet. I need at least 20 messages to "
                            f"learn your style (you've sent {total_messages} so far)."
                        )
                    )
                return CommandResult(
                    text=(
                        "No style profile yet, but you have enough messages. "
                        "I'll analyze your style soon!"
                    )
                )

            status = "enabled" if profile.enabled else "disabled"
            lines = [
                "**Your Speaking Style**",
                "",
                profile.style_prompt,
                "",
                f"Status: {status}",
                f"Based on {profile.message_count} messages",
            ]
            return CommandResult(text="\n".join(lines))

        # Case 2: Reset style profile
        if subcommand == "reset":
            total_messages = context.db.count_user_messages(context.user)
            if total_messages < 20:
                return CommandResult(
                    text=(
                        f"I need at least 20 messages to analyze your style "
                        f"(you've sent {total_messages} so far)."
                    )
                )

            # Get recent messages
            messages = context.db.get_user_messages_for_style(context.user, limit=100)
            if not messages:
                return CommandResult(text="No messages found to analyze.")

            # Analyze style
            style_prompt = await self._style_agent._analyze_style(context.user, messages)
            if not style_prompt:
                return CommandResult(text="Failed to analyze your style. Please try again.")

            # Upsert profile
            context.db.upsert_user_style_profile(
                user_id=context.user,
                style_prompt=style_prompt,
                message_count=len(messages),
            )

            return CommandResult(
                text=f"Style profile reset and regenerated from {len(messages)} messages."
            )

        # Case 3: Disable adaptive style
        if subcommand == "off":
            profile = context.db.get_user_style_profile(context.user)
            if not profile:
                return CommandResult(text="No style profile to disable.")

            if not profile.enabled:
                return CommandResult(text="Adaptive style is already disabled.")

            context.db.update_style_profile_enabled(context.user, False)
            return CommandResult(text="Adaptive style disabled.")

        # Case 4: Enable adaptive style
        if subcommand == "on":
            profile = context.db.get_user_style_profile(context.user)
            if not profile:
                return CommandResult(
                    text=(
                        "No style profile yet. I'll create one once you've "
                        "sent at least 20 messages."
                    )
                )

            if profile.enabled:
                return CommandResult(text="Adaptive style is already enabled.")

            context.db.update_style_profile_enabled(context.user, True)
            return CommandResult(text="Adaptive style enabled.")

        # Unknown subcommand
        return CommandResult(
            text=(
                f"Unknown subcommand: {subcommand}\n"
                f"Use /style, /style reset, /style off, or /style on."
            )
        )

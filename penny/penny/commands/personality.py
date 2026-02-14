"""Personality command - /personality."""

from __future__ import annotations

import logging

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.constants import (
    PERSONALITY_NO_CUSTOM,
    PERSONALITY_RESET_NOT_SET,
    PERSONALITY_RESET_SUCCESS,
    PERSONALITY_UPDATE_SUCCESS,
)

logger = logging.getLogger(__name__)


class PersonalityCommand(Command):
    """View, set, or reset custom personality prompt."""

    name = "personality"
    description = "Customize Penny's personality and tone"
    help_text = (
        "Set a custom personality that affects how Penny responds.\n\n"
        "**Usage**:\n"
        "- `/personality` — View current personality prompt\n"
        "- `/personality <prompt>` — Set custom personality\n"
        "- `/personality reset` — Clear custom personality and revert to default\n\n"
        "**Examples**:\n"
        "- `/personality you are a pirate who speaks in nautical metaphors`\n"
        "- `/personality be extremely concise, never use more than 2 sentences`\n"
        "- `/personality you are a sarcastic but helpful friend who loves dad jokes`\n"
        "- `/personality reset`"
    )

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute personality command."""
        args = args.strip()

        # No args - show current personality
        if not args:
            personality = context.db.get_personality_prompt(context.user)
            if not personality:
                return CommandResult(text=PERSONALITY_NO_CUSTOM)

            return CommandResult(text=f"Current personality: {personality.prompt_text}")

        # Reset personality
        if args.lower() == "reset":
            removed = context.db.remove_personality_prompt(context.user)
            if removed:
                return CommandResult(text=PERSONALITY_RESET_SUCCESS)
            else:
                return CommandResult(text=PERSONALITY_RESET_NOT_SET)

        # Set new personality
        prompt_text = args
        context.db.set_personality_prompt(context.user, prompt_text)
        return CommandResult(text=PERSONALITY_UPDATE_SUCCESS)

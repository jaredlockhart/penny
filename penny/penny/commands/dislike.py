"""/dislike command — list negative preferences or add a new one."""

from __future__ import annotations

from penny.commands.preference_base import NEGATIVE_CONFIG, PreferenceAddCommand


class DislikeCommand(PreferenceAddCommand):
    """List negative preferences or add a new one."""

    name = "dislike"
    description = "Show or add dislikes"
    help_text = (
        "Show your disliked preferences or add a new one.\n\n"
        "**Usage**:\n"
        "• `/dislike` — Show numbered list of dislikes\n"
        "• `/dislike <text>` — Add a new dislike\n\n"
        "**Examples**:\n"
        "• `/dislike`\n"
        "• `/dislike cold weather`"
    )
    valence_config = NEGATIVE_CONFIG

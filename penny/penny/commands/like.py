"""/like command — list positive preferences or add a new one."""

from __future__ import annotations

from penny.commands.preference_base import POSITIVE_CONFIG, PreferenceAddCommand


class LikeCommand(PreferenceAddCommand):
    """List positive preferences or add a new one."""

    name = "like"
    description = "Show or add likes"
    help_text = (
        "Show your liked preferences or add a new one.\n\n"
        "**Usage**:\n"
        "• `/like` — Show numbered list of likes\n"
        "• `/like <text>` — Add a new like\n\n"
        "**Examples**:\n"
        "• `/like`\n"
        "• `/like dark roast coffee`"
    )
    valence_config = POSITIVE_CONFIG

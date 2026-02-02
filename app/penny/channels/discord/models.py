"""Pydantic models for Discord API structures (template for contribution)."""

from pydantic import BaseModel


# TODO: Add Discord-specific models here
# Examples:
# - DiscordMessage
# - DiscordUser
# - DiscordChannel
# - DiscordEmbed
# etc.


class DiscordMessagePayload(BaseModel):
    """Placeholder for Discord message payload."""

    content: str
    channel_id: str

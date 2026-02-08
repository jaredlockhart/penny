"""Pydantic models for Discord API structures."""

from pydantic import BaseModel, ConfigDict, Field


class DiscordUser(BaseModel):
    """Discord user structure."""

    id: str
    username: str
    discriminator: str = ""
    bot: bool = False
    global_name: str | None = None


class DiscordMessage(BaseModel):
    """Discord message structure from gateway events."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    channel_id: str = Field(alias="channel_id")
    author: DiscordUser
    content: str
    timestamp: str
    guild_id: str | None = Field(default=None, alias="guild_id")

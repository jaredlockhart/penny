"""Channel abstraction for communication platforms."""

from __future__ import annotations

from typing import TYPE_CHECKING

from penny.channels.base import IncomingMessage, MessageChannel
from penny.channels.discord import DiscordChannel
from penny.channels.signal import SignalChannel
from penny.config import Config

if TYPE_CHECKING:
    from penny.agents import MessageAgent
    from penny.commands import CommandRegistry
    from penny.database import Database

# Channel type constants
CHANNEL_TYPE_SIGNAL = "signal"
CHANNEL_TYPE_DISCORD = "discord"


def create_channel(
    config: Config,
    message_agent: MessageAgent,
    db: Database,
    command_registry: CommandRegistry | None = None,
) -> MessageChannel:
    """
    Create the appropriate channel based on configuration.

    Args:
        config: Application configuration
        message_agent: Agent for processing incoming messages
        db: Database for logging messages
        command_registry: Optional command registry for handling commands

    Returns:
        Configured MessageChannel instance

    Raises:
        ValueError: If channel type is unknown or required config is missing
    """
    if config.channel_type == CHANNEL_TYPE_DISCORD:
        if not config.discord_bot_token or not config.discord_channel_id:
            raise ValueError("Discord requires DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID")
        return DiscordChannel(
            token=config.discord_bot_token,
            channel_id=config.discord_channel_id,
            message_agent=message_agent,
            db=db,
            command_registry=command_registry,
        )
    elif config.channel_type == CHANNEL_TYPE_SIGNAL:
        if not config.signal_number:
            raise ValueError("Signal requires SIGNAL_NUMBER")
        return SignalChannel(
            api_url=config.signal_api_url,
            phone_number=config.signal_number,
            message_agent=message_agent,
            db=db,
            command_registry=command_registry,
            max_retries=config.ollama_max_retries,
            retry_delay=config.ollama_retry_delay,
        )
    else:
        raise ValueError(f"Unknown channel type: {config.channel_type}")


__all__ = [
    "MessageChannel",
    "IncomingMessage",
    "SignalChannel",
    "DiscordChannel",
    "create_channel",
    "CHANNEL_TYPE_SIGNAL",
    "CHANNEL_TYPE_DISCORD",
]

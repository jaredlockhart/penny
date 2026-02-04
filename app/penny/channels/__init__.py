"""Channel abstraction for communication platforms."""

from penny.channels.base import IncomingMessage, MessageCallback, MessageChannel
from penny.channels.discord import DiscordChannel
from penny.channels.signal import SignalChannel
from penny.config import Config

# Channel type constants
CHANNEL_TYPE_SIGNAL = "signal"
CHANNEL_TYPE_DISCORD = "discord"


def create_channel(config: Config, on_message: MessageCallback) -> MessageChannel:
    """
    Create the appropriate channel based on configuration.

    Args:
        config: Application configuration
        on_message: Callback for incoming messages

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
            on_message=on_message,
        )
    elif config.channel_type == CHANNEL_TYPE_SIGNAL:
        if not config.signal_number:
            raise ValueError("Signal requires SIGNAL_NUMBER")
        return SignalChannel(
            api_url=config.signal_api_url,
            phone_number=config.signal_number,
            on_message=on_message,
        )
    else:
        raise ValueError(f"Unknown channel type: {config.channel_type}")


__all__ = [
    "MessageChannel",
    "MessageCallback",
    "IncomingMessage",
    "SignalChannel",
    "DiscordChannel",
    "create_channel",
    "CHANNEL_TYPE_SIGNAL",
    "CHANNEL_TYPE_DISCORD",
]

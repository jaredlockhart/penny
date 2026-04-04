"""Channel abstraction for communication platforms."""

from __future__ import annotations

from typing import TYPE_CHECKING

from penny.channels.base import IncomingMessage, MessageChannel
from penny.channels.discord import DiscordChannel
from penny.channels.manager import ChannelManager
from penny.channels.signal import SignalChannel
from penny.config import Config
from penny.constants import ChannelType

if TYPE_CHECKING:
    from penny.agents import ChatAgent
    from penny.commands import CommandRegistry
    from penny.database import Database

# Channel type constants (aliases for backward compat)
CHANNEL_TYPE_SIGNAL = ChannelType.SIGNAL
CHANNEL_TYPE_DISCORD = ChannelType.DISCORD
CHANNEL_TYPE_BROWSER = ChannelType.BROWSER


def create_channel_manager(
    config: Config,
    message_agent: ChatAgent,
    db: Database,
    command_registry: CommandRegistry | None = None,
) -> ChannelManager:
    """Create a ChannelManager with all configured channels registered."""
    manager = ChannelManager(
        message_agent=message_agent,
        db=db,
        command_registry=command_registry,
    )

    _register_primary_channel(config, message_agent, db, command_registry, manager)

    if config.browser_enabled:
        _register_browser_channel(config, message_agent, db, command_registry, manager)

    return manager


def _register_primary_channel(
    config: Config,
    message_agent: ChatAgent,
    db: Database,
    command_registry: CommandRegistry | None,
    manager: ChannelManager,
) -> None:
    """Create and register the primary channel (Signal or Discord)."""
    if config.channel_type == ChannelType.DISCORD:
        if not config.discord_bot_token or not config.discord_channel_id:
            raise ValueError("Discord requires DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID")
        channel = DiscordChannel(
            token=config.discord_bot_token,
            channel_id=config.discord_channel_id,
            message_agent=message_agent,
            db=db,
            command_registry=command_registry,
        )
        manager.register_channel(ChannelType.DISCORD, channel)
    elif config.channel_type == ChannelType.SIGNAL:
        if not config.signal_number:
            raise ValueError("Signal requires SIGNAL_NUMBER")
        channel = SignalChannel(
            api_url=config.signal_api_url,
            phone_number=config.signal_number,
            message_agent=message_agent,
            db=db,
            command_registry=command_registry,
            max_retries=config.llm_max_retries,
            retry_delay=config.llm_retry_delay,
        )
        manager.register_channel(ChannelType.SIGNAL, channel)
        # Seed the Signal device
        db.devices.register(ChannelType.SIGNAL, config.signal_number, "Signal", is_default=True)
    else:
        raise ValueError(f"Unknown channel type: {config.channel_type}")


def _register_browser_channel(
    config: Config,
    message_agent: ChatAgent,
    db: Database,
    command_registry: CommandRegistry | None,
    manager: ChannelManager,
) -> None:
    """Create and register the browser channel if enabled."""
    from penny.channels.browser import BrowserChannel

    channel = BrowserChannel(
        host=config.browser_host,
        port=config.browser_port,
        message_agent=message_agent,
        db=db,
        command_registry=command_registry,
    )
    manager.register_channel(ChannelType.BROWSER, channel)


def create_channel(
    config: Config,
    message_agent: ChatAgent,
    db: Database,
    command_registry: CommandRegistry | None = None,
) -> MessageChannel:
    """Create a single channel (backward compat for tests)."""
    manager = create_channel_manager(config, message_agent, db, command_registry)
    return manager


__all__ = [
    "MessageChannel",
    "IncomingMessage",
    "SignalChannel",
    "DiscordChannel",
    "ChannelManager",
    "create_channel",
    "create_channel_manager",
    "CHANNEL_TYPE_SIGNAL",
    "CHANNEL_TYPE_DISCORD",
    "CHANNEL_TYPE_BROWSER",
]

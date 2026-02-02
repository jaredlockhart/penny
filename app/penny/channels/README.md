# Channels Module

This module provides an abstraction layer for communication channels, allowing Penny to work with different messaging platforms.

## Architecture

The `MessageChannel` abstract base class defines the interface that all channel implementations must follow. This allows the agent to work with any messaging platform without being tightly coupled to a specific implementation.

## Directory Structure

Each channel implementation follows this structure:

```
penny/channels/
├── base.py                 # Abstract MessageChannel interface
├── signal/                 # Signal implementation
│   ├── __init__.py
│   ├── channel.py         # SignalChannel class
│   └── models.py          # Signal-specific Pydantic models
└── discord/                # Discord template
    ├── __init__.py
    ├── channel.py         # DiscordChannel class
    └── models.py          # Discord-specific Pydantic models
```

## Creating a New Channel

To add support for a new platform (e.g., Slack, Telegram):

1. Create a new subdirectory (e.g., `slack/`)
2. Create `channel.py` and implement the `MessageChannel` interface:

```python
# slack/channel.py
from penny.channels.base import MessageChannel, IncomingMessage

class SlackChannel(MessageChannel):
    async def send_message(self, recipient: str, message: str) -> bool:
        """Send a message to a recipient."""
        # Implementation here
        pass

    async def send_typing(self, recipient: str, typing: bool) -> bool:
        """Send typing indicator."""
        # Implementation here
        pass

    def get_connection_url(self) -> str:
        """Get connection URL/identifier."""
        # Return connection string
        pass

    def extract_message(self, raw_data: dict) -> IncomingMessage | None:
        """Extract message from platform-specific data."""
        # Parse platform data and return IncomingMessage
        pass

    async def close(self) -> None:
        """Cleanup resources."""
        # Close connections
        pass
```

3. Create `models.py` for platform-specific Pydantic models:

```python
# slack/models.py
from pydantic import BaseModel

class SlackMessage(BaseModel):
    """Slack message structure."""
    channel: str
    user: str
    text: str
```

4. Create `__init__.py` to export your channel:

```python
# slack/__init__.py
from penny.channels.slack.channel import SlackChannel
from penny.channels.slack.models import SlackMessage

__all__ = ["SlackChannel", "SlackMessage"]
```

5. Optionally add to main `channels/__init__.py` for convenience:

```python
from penny.channels.slack import SlackChannel
```

6. Use it in the agent:

```python
from penny.channels import SlackChannel

channel = SlackChannel(...)
agent = PennyAgent(config, channel=channel)
```

## Reference Implementation: Signal

See the [`signal/`](./signal/) directory for a complete reference implementation:
- [`signal/channel.py`](./signal/channel.py) - SignalChannel implementation
- [`signal/models.py`](./signal/models.py) - Signal-specific Pydantic models
- [`signal/__init__.py`](./signal/__init__.py) - Module exports

## Template: Discord

See the [`discord/`](./discord/) directory for a starter template with TODOs for Discord integration.

## Notes

- The `recipient` parameter is platform-specific (could be phone number, user ID, channel ID, etc.)
- The `IncomingMessage` model is generic with just `sender` and `content` fields
- Platform-specific models and logic should be kept in their respective `models.py` files
- Each channel implementation is self-contained in its own subdirectory

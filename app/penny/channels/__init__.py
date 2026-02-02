"""Channel abstraction for communication platforms."""

from penny.channels.base import IncomingMessage, MessageChannel
from penny.channels.signal import SignalChannel

__all__ = ["MessageChannel", "IncomingMessage", "SignalChannel"]

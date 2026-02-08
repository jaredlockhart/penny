"""Models for command system."""

from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel

from penny.config import Config
from penny.database import Database


@dataclass
class CommandContext:
    """Runtime context passed to command handlers."""

    db: Database
    config: Config
    user: str  # Signal number or Discord user ID
    channel_type: str  # "signal" or "discord"
    start_time: datetime  # Penny startup time for uptime calculation


class CommandResult(BaseModel):
    """Result from executing a command."""

    text: str  # Response text to send to user


class CommandError(BaseModel):
    """Error from executing a command."""

    message: str  # Error message to send to user

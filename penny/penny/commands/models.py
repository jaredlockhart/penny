"""Models for command system."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel

from penny.config import Config
from penny.database import Database
from penny.ollama import OllamaClient

if TYPE_CHECKING:
    from penny.channels.base import IncomingMessage
    from penny.scheduler import BackgroundScheduler


@dataclass
class CommandContext:
    """Runtime context passed to command handlers."""

    db: Database
    config: Config
    ollama_client: OllamaClient
    user: str  # Signal number or Discord user ID
    channel_type: str  # "signal" or "discord"
    start_time: datetime  # Penny startup time for uptime calculation
    scheduler: BackgroundScheduler | None = None  # Background task scheduler
    message: IncomingMessage | None = None  # The incoming message (for quote-reply metadata)


class CommandResult(BaseModel):
    """Result from executing a command."""

    text: str  # Response text to send to user


class CommandError(BaseModel):
    """Error from executing a command."""

    message: str  # Error message to send to user

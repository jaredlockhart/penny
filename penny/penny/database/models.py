"""SQLModel models for Penny's memory."""

import json
from datetime import UTC, datetime

from sqlmodel import Field, SQLModel


class PromptLog(SQLModel, table=True):
    """Log of every prompt sent to Ollama and its response."""

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), index=True)
    model: str
    messages: str  # JSON-serialized list of message dicts
    tools: str | None = None  # JSON-serialized tool definitions
    response: str  # JSON-serialized response dict
    thinking: str | None = None  # Model's thinking/reasoning trace
    duration_ms: int | None = None  # How long the call took

    def get_messages(self) -> list[dict]:
        return json.loads(self.messages)

    def get_response(self) -> dict:
        return json.loads(self.response)


class SearchLog(SQLModel, table=True):
    """Log of every Perplexity search call and its response."""

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), index=True)
    query: str = Field(index=True)
    response: str
    duration_ms: int | None = None
    trigger: str = Field(default="user_message", index=True)  # SearchTrigger enum value


class MessageLog(SQLModel, table=True):
    """Log of every user message and agent response."""

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), index=True)
    direction: str = Field(index=True)  # "incoming" or "outgoing"
    sender: str = Field(index=True)
    content: str
    parent_id: int | None = Field(default=None, foreign_key="messagelog.id", index=True)
    signal_timestamp: int | None = Field(default=None)  # Original Signal timestamp (ms since epoch)
    recipient: str | None = Field(default=None, index=True)  # Who the message was sent to
    external_id: str | None = Field(default=None, index=True)  # Platform-specific message ID
    is_reaction: bool = Field(default=False, index=True)  # True if this is a reaction message
    processed: bool = Field(
        default=False
    )  # True if this message has been processed by extraction pipeline
    thought_id: int | None = Field(
        default=None, foreign_key="thought.id", index=True
    )  # FK to thought that triggered this notification
    embedding: bytes | None = None  # Serialized float32 embedding vector


class UserInfo(SQLModel, table=True):
    """Basic user information collected on first interaction."""

    __tablename__ = "userinfo"

    id: int | None = Field(default=None, primary_key=True)
    sender: str = Field(unique=True, index=True)
    name: str
    location: str
    timezone: str  # IANA timezone (e.g., "America/Los_Angeles")
    date_of_birth: str  # YYYY-MM-DD format
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CommandLog(SQLModel, table=True):
    """Log of every command invocation and its response."""

    __tablename__ = "command_logs"

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), index=True)
    user: str = Field(index=True)  # Signal number or Discord user ID
    channel_type: str  # "signal" or "discord"
    command_name: str = Field(index=True)  # e.g., "debug"
    command_args: str  # e.g., "" or "debug" (for /commands debug)
    response: str  # Full response text sent to user
    error: str | None = None  # Error message if command failed


class RuntimeConfig(SQLModel, table=True):
    """User-configurable runtime settings stored in database."""

    __tablename__ = "runtime_config"

    key: str = Field(primary_key=True)
    value: str  # Store as string, parse to correct type on load
    description: str  # Human-readable description with units
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Schedule(SQLModel, table=True):
    """User-created scheduled background tasks."""

    id: int | None = Field(default=None, primary_key=True)
    user_id: str = Field(index=True)  # Signal number or Discord user ID
    user_timezone: str = Field(default="UTC")  # IANA timezone (e.g., "America/Los_Angeles")
    cron_expression: str  # Cron format for recurring execution
    prompt_text: str  # Prompt to execute when schedule fires
    timing_description: str  # Original human description for display (e.g., "daily 9am")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MuteState(SQLModel, table=True):
    """Per-user mute state for notifications.

    Row exists = muted. Delete row = unmuted.
    """

    user: str = Field(primary_key=True)
    muted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Thought(SQLModel, table=True):
    """A persistent inner monologue entry — Penny's stream of consciousness."""

    id: int | None = Field(default=None, primary_key=True)
    user: str = Field(index=True)
    content: str
    preference_id: int | None = Field(default=None, foreign_key="preference.id", index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), index=True)
    notified_at: datetime | None = None  # When this thought was shared with the user
    embedding: bytes | None = None  # Serialized float32 embedding vector


class Preference(SQLModel, table=True):
    """A user preference extracted from conversation sentiment or emoji reactions."""

    id: int | None = Field(default=None, primary_key=True)
    user: str = Field(index=True)
    content: str  # The preference topic (e.g., "dark roast coffee", "cold weather")
    valence: str = Field(index=True)  # PreferenceValence enum value
    embedding: bytes | None = None  # Serialized float32 embedding vector
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), index=True)
    last_thought_at: datetime | None = None  # When this preference was last used as a thinking seed
    mention_count: int = Field(default=1)  # Times this topic was mentioned in conversation
    source: str = Field(default="extracted", index=True)  # PreferenceSource enum value


class ConversationHistory(SQLModel, table=True):
    """A topic summary for a conversation period (daily, weekly, monthly)."""

    __tablename__ = "conversationhistory"

    id: int | None = Field(default=None, primary_key=True)
    user: str = Field(index=True)
    period_start: datetime = Field(index=True)
    period_end: datetime
    duration: str  # PennyConstants.HistoryDuration enum value
    topics: str  # Bullet-point list of topics discussed
    embedding: bytes | None = None  # Serialized float32 embedding vector
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

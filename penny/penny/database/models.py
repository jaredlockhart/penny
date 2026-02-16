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


class MessageLog(SQLModel, table=True):
    """Log of every user message and agent response."""

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), index=True)
    direction: str = Field(index=True)  # "incoming" or "outgoing"
    sender: str = Field(index=True)
    content: str
    parent_id: int | None = Field(default=None, foreign_key="messagelog.id", index=True)
    signal_timestamp: int | None = Field(default=None)  # Original Signal timestamp (ms since epoch)
    external_id: str | None = Field(default=None, index=True)  # Platform-specific message ID
    is_reaction: bool = Field(default=False, index=True)  # True if this is a reaction message
    processed: bool = Field(
        default=False
    )  # True if this message has been processed by PreferenceAgent


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


class Preference(SQLModel, table=True):
    """User preferences (likes/dislikes) for discovery."""

    id: int | None = Field(default=None, primary_key=True)
    user: str = Field(index=True)  # Signal number or Discord user ID
    topic: str  # Arbitrary natural language phrase
    type: str  # "like" or "dislike"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ResearchTask(SQLModel, table=True):
    """Research tasks initiated by users via /research command."""

    __tablename__ = "research_tasks"

    id: int | None = Field(default=None, primary_key=True)
    thread_id: str = Field(index=True)  # Signal thread ID or Discord channel ID
    message_id: str | None = Field(default=None)  # ID of the report message, set when posted
    parent_task_id: int | None = Field(
        default=None, foreign_key="research_tasks.id"
    )  # For continuations
    topic: str  # User's research request
    status: str = Field(index=True)  # "awaiting_focus", "in_progress", "completed", "failed"
    focus: str | None = None  # User's focus/direction for research (set after clarification)
    options: str | None = None  # Raw numbered options shown to user (for resolving number replies)
    max_iterations: int  # Snapshot of config at creation time
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None


class ResearchIteration(SQLModel, table=True):
    """Individual search iterations within a research task."""

    __tablename__ = "research_iterations"

    id: int | None = Field(default=None, primary_key=True)
    research_task_id: int = Field(foreign_key="research_tasks.id", index=True)
    iteration_num: int
    query: str
    findings: str  # JSON blob: extracted insights from this iteration
    sources: str  # JSON array: URLs discovered
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Schedule(SQLModel, table=True):
    """User-created scheduled background tasks."""

    id: int | None = Field(default=None, primary_key=True)
    user_id: str = Field(index=True)  # Signal number or Discord user ID
    user_timezone: str = Field(default="UTC")  # IANA timezone (e.g., "America/Los_Angeles")
    cron_expression: str  # Cron format for recurring execution
    prompt_text: str  # Prompt to execute when schedule fires
    timing_description: str  # Original human description for display (e.g., "daily 9am")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PersonalityPrompt(SQLModel, table=True):
    """User custom personality prompts that shape Penny's tone and behavior."""

    __tablename__ = "personalityprompt"

    user_id: str = Field(primary_key=True)  # Signal number or Discord user ID
    prompt_text: str  # The custom personality prompt
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

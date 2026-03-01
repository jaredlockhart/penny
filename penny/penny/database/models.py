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
    extracted: bool = Field(default=False)  # True after entity extraction processing
    trigger: str = Field(default="user_message", index=True)  # SearchTrigger enum value
    learn_prompt_id: int | None = Field(default=None, foreign_key="learnprompt.id", index=True)


class LearnPrompt(SQLModel, table=True):
    """A user-initiated learning prompt with lifecycle tracking."""

    id: int | None = Field(default=None, primary_key=True)
    user: str = Field(index=True)  # Signal number or Discord user ID
    prompt_text: str  # Original user text (e.g., "find me stuff about speakers")
    status: str = Field(default="active", index=True)  # LearnPromptStatus enum value
    searches_remaining: int = Field(default=0)  # Searches left to execute
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    announced_at: datetime | None = Field(default=None)  # When completion announcement was sent


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
    )  # True if this message has been processed by extraction pipeline


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


class Entity(SQLModel, table=True):
    """A named entity (product, person, place, concept)."""

    id: int | None = Field(default=None, primary_key=True)
    user: str = Field(index=True)  # Signal number or Discord user ID
    name: str  # Lowercased canonical name (e.g., "kef ls50 meta")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    embedding: bytes | None = None  # Serialized float32 embedding vector
    tagline: str | None = None  # Short disambiguating summary (e.g., "british prog rock band")
    last_enriched_at: datetime | None = None  # When this entity was last enriched
    last_notified_at: datetime | None = None  # When this entity was last included in a notification
    heat: float = Field(default=0.0)  # Persistent interest score (thermodynamic heat model)
    heat_cooldown: int = Field(default=0)  # Notification cycles remaining before eligible


class Engagement(SQLModel, table=True):
    """A user engagement event recording interest in an entity."""

    id: int | None = Field(default=None, primary_key=True)
    user: str = Field(index=True)  # Signal number or Discord user ID
    entity_id: int | None = Field(default=None, foreign_key="entity.id", index=True)
    engagement_type: str = Field(index=True)  # EngagementType enum value
    valence: str  # EngagementValence enum value
    strength: float  # Weight 0.0-1.0
    source_message_id: int | None = Field(default=None, foreign_key="messagelog.id", index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), index=True)


class MuteState(SQLModel, table=True):
    """Per-user mute state for proactive notifications.

    Row exists = muted. Delete row = unmuted.
    """

    user: str = Field(primary_key=True)
    muted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Fact(SQLModel, table=True):
    """An individual fact about an entity with provenance tracking."""

    id: int | None = Field(default=None, primary_key=True)
    entity_id: int = Field(foreign_key="entity.id", index=True)
    content: str  # The fact text (e.g., "Costs $1,599 per pair")
    source_url: str | None = None  # URL where the fact was found
    source_search_log_id: int | None = Field(default=None, foreign_key="searchlog.id", index=True)
    source_message_id: int | None = Field(default=None, foreign_key="messagelog.id", index=True)
    learned_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    notified_at: datetime | None = None  # When this fact was communicated to user
    embedding: bytes | None = None  # Serialized float32 embedding vector


class Event(SQLModel, table=True):
    """A time-stamped occurrence that can involve multiple entities."""

    id: int | None = Field(default=None, primary_key=True)
    user: str = Field(index=True)  # Signal number or Discord user ID
    headline: str  # Short event title
    summary: str  # Event description/body
    occurred_at: datetime = Field(index=True)  # When the event happened
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_url: str | None = None  # Article URL
    source_type: str = Field(index=True)  # EventSourceType enum value
    external_id: str | None = Field(default=None, index=True)  # Article URL for dedup
    notified_at: datetime | None = None  # When user was told
    embedding: bytes | None = None  # Serialized float32 embedding vector
    follow_prompt_id: int | None = Field(
        default=None, foreign_key="followprompt.id", index=True
    )  # Which follow prompt generated this event


class FollowPrompt(SQLModel, table=True):
    """An ongoing monitoring subscription for event tracking."""

    __tablename__ = "followprompt"

    id: int | None = Field(default=None, primary_key=True)
    user: str = Field(index=True)  # Signal number or Discord user ID
    prompt_text: str  # User's natural language topic (e.g., "AI news")
    status: str = Field(default="active", index=True)  # FollowPromptStatus enum value
    query_terms: str = ""  # LLM-generated search terms (JSON list)
    cadence: str = Field(default="daily")  # Legacy — kept for backward compat
    cron_expression: str = Field(default="0 9 * * *")  # 5-field cron schedule
    timing_description: str = Field(default="daily")  # Human-readable timing
    user_timezone: str = Field(default="UTC")  # IANA timezone for cron evaluation
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_polled_at: datetime | None = None  # When this subscription was last checked
    last_notified_at: datetime | None = None  # When this prompt last triggered a notification

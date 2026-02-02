"""Configuration models for agentic components."""

from pydantic import BaseModel, Field


class ControllerConfig(BaseModel):
    """Configuration for AgenticController."""

    max_steps: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of agentic loop iterations",
    )


class AgentConfig(BaseModel):
    """Configuration for PennyAgent."""

    message_max_steps: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Max steps for message handler",
    )
    task_max_steps: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Max steps for task processor",
    )
    idle_timeout_seconds: float = Field(
        default=5.0,
        ge=1.0,
        le=60.0,
        description="Seconds to wait before processing tasks",
    )
    task_check_interval: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Seconds between task queue checks",
    )
    conversation_history_limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of messages in conversation context",
    )

    class Config:
        """Pydantic config."""

        frozen = True  # Make immutable

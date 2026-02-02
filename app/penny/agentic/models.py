"""Pydantic models and enums for agentic loop."""

from enum import Enum

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Valid message roles in chat conversations."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """A message in a chat conversation."""

    role: MessageRole
    content: str

    def to_dict(self) -> dict:
        """Convert to dict for Ollama API."""
        return {"role": self.role.value, "content": self.content}


class ControllerResponse(BaseModel):
    """Response from the agentic controller."""

    answer: str = Field(description="The final answer from the controller")
    thinking: str | None = Field(
        default=None, description="Optional thinking/reasoning trace from the model"
    )


class MessageClassification(str, Enum):
    """Classification of user messages for routing."""

    TASK = "task"
    IMMEDIATE = "immediate"


class ClassificationResult(BaseModel):
    """Result of message classification."""

    classification: MessageClassification = Field(
        description="Whether the message is a task or immediate question"
    )
    acknowledgment: str | None = Field(
        default=None, description="Acknowledgment message for tasks, None for immediate"
    )

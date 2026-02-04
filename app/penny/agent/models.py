"""Pydantic models and enums for agent loop."""

from enum import StrEnum

from pydantic import BaseModel, Field


class MessageRole(StrEnum):
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
    attachments: list[str] = Field(
        default_factory=list, description="Base64-encoded image attachments"
    )

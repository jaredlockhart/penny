"""Pydantic models for Ollama API structures."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    """Types of chunks in streaming responses."""

    THINKING = "thinking"
    RESPONSE = "response"


class GenerateRequest(BaseModel):
    """Request to generate a completion."""

    model: str
    prompt: str
    stream: bool = False
    options: dict[str, Any] | None = None


class GenerateResponse(BaseModel):
    """Response from a generation request."""

    model: str
    created_at: str = Field(alias="created_at")
    response: str = ""
    thinking: str = ""  # For thinking models
    done: bool
    tool_calls: list[dict[str, Any]] | None = None  # Ollama native tool calling
    context: list[int] | None = None
    total_duration: int | None = Field(default=None, alias="total_duration")
    load_duration: int | None = Field(default=None, alias="load_duration")
    prompt_eval_count: int | None = Field(default=None, alias="prompt_eval_count")
    prompt_eval_duration: int | None = Field(default=None, alias="prompt_eval_duration")
    eval_count: int | None = Field(default=None, alias="eval_count")
    eval_duration: int | None = Field(default=None, alias="eval_duration")

    class Config:
        populate_by_name = True


class OllamaStreamResponse(BaseModel):
    """Raw response chunk from Ollama streaming API."""

    model: str | None = None
    created_at: str | None = None
    response: str = ""
    thinking: str = ""
    done: bool = False

    class Config:
        populate_by_name = True


class StreamChunk(BaseModel):
    """A chunk from the streaming API."""

    type: ChunkType
    content: str


class ResponseLine(BaseModel):
    """A complete response line with optional thinking."""

    line: str
    thinking: str | None = None


class ChatResponseMessage(BaseModel):
    """Message object from chat response."""

    role: str
    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    thinking: str | None = None

    class Config:
        populate_by_name = True


class ChatResponse(BaseModel):
    """Response from Ollama chat API."""

    message: ChatResponseMessage
    thinking: str | None = None
    done: bool = True
    model: str | None = None
    created_at: str | None = None

    class Config:
        populate_by_name = True
        extra = "allow"  # Allow additional fields from Ollama

    @property
    def content(self) -> str:
        """Get message content."""
        return self.message.content

    @property
    def has_tool_calls(self) -> bool:
        """Check if response has tool calls."""
        return bool(self.message.tool_calls)

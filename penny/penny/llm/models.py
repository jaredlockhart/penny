"""Pydantic models for LLM client responses.

These are our own types, decoupled from any SDK. The LlmClient
translates provider-specific responses into these models.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

# ── Error types ──────────────────────────────────────────────────────────


class LlmError(Exception):
    """Base error for LLM client operations."""


class LlmNotFoundError(LlmError):
    """Model not found (404). Should not be retried."""


class LlmConnectionError(LlmError):
    """Could not connect to the LLM server."""


class LlmResponseError(LlmError):
    """Server returned an error response."""


class LlmMalformedToolCallError(LlmError):
    """Model generated malformed tool call JSON that the server could not parse.

    Raised when Ollama (or any OpenAI-compatible backend) returns a 500 error
    with an 'error parsing tool call' message. The raw malformed JSON is
    attached for debugging.
    """

    def __init__(self, message: str, raw_json: str | None = None) -> None:
        super().__init__(message)
        self.raw_json = raw_json


# ── Response types ───────────────────────────────────────────────────────


class LlmToolCallFunction(BaseModel):
    """Function details within a tool call."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LlmToolCall(BaseModel):
    """A tool call from the model response."""

    id: str
    function: LlmToolCallFunction


class LlmMessage(BaseModel):
    """Message object from a chat response."""

    role: str
    content: str = ""
    tool_calls: list[LlmToolCall] | None = None
    thinking: str | None = None

    def to_input_message(self) -> dict[str, Any]:
        """Convert to input message format for the next request (excludes thinking)."""
        message: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": json.dumps(tool_call.function.arguments),
                    },
                }
                for tool_call in self.tool_calls
            ]
        return message


class LlmResponse(BaseModel):
    """Response from an LLM chat call."""

    message: LlmMessage
    thinking: str | None = None
    model: str | None = None

    @property
    def content(self) -> str:
        """Get message content."""
        return self.message.content

    @property
    def has_tool_calls(self) -> bool:
        """Check if response has tool calls."""
        return bool(self.message.tool_calls)

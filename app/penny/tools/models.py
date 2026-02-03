"""Pydantic models for tool calling."""

from typing import Any

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Result from a search tool containing text and optional image."""

    text: str
    image_base64: str | None = None
    urls: list[str] = Field(default_factory=list)


class ToolCall(BaseModel):
    """A tool call from the model."""

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None


class ToolResult(BaseModel):
    """Result from executing a tool."""

    tool: str
    result: Any
    error: str | None = None
    id: str | None = None


class ToolDefinition(BaseModel):
    """Definition of a tool for the model."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)

"""Pydantic models for tool calling."""

from typing import Any

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Result from a search tool containing text and optional image."""

    text: str
    image_base64: str | None = None
    urls: list[str] = Field(default_factory=list)

    def __str__(self) -> str:
        image_summary = f"<image {len(self.image_base64)} chars>" if self.image_base64 else "None"
        return f"SearchResult(text={self.text}, urls={self.urls}, image_base64={image_summary})"


class BrowseArgs(BaseModel):
    """Validated arguments for the browse tool."""

    queries: list[str] = Field(default_factory=list)
    reasoning: str | None = None


class SendMessageArgs(BaseModel):
    """Validated arguments for the send_message tool."""

    content: str


class SearchEmailsArgs(BaseModel):
    """Validated arguments for the search_emails tool."""

    text: str | None = None
    from_addr: str | None = None
    subject: str | None = None
    after: str | None = None
    before: str | None = None


class ReadEmailsArgs(BaseModel):
    """Validated arguments for the read_emails tool."""

    email_ids: list[str]


class ListEmailsArgs(BaseModel):
    """Validated arguments for the list_emails tool."""

    limit: int = 10
    folder: str | None = None


class DraftEmailArgs(BaseModel):
    """Validated arguments for the draft_email tool."""

    to: list[str]
    subject: str
    body: str
    cc: list[str] | None = None


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

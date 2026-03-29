"""Pydantic models for tool calling."""

from typing import Any

from pydantic import BaseModel, Field


class SearchArgs(BaseModel):
    """Validated arguments for the search tool."""

    query: str


class SearchResult(BaseModel):
    """Result from a search tool containing text and optional image."""

    text: str
    image_base64: str | None = None
    urls: list[str] = Field(default_factory=list)

    def __str__(self) -> str:
        image_summary = f"<image {len(self.image_base64)} chars>" if self.image_base64 else "None"
        return f"SearchResult(text={self.text}, urls={self.urls}, image_base64={image_summary})"


class FetchNewsArgs(BaseModel):
    """Validated arguments for the fetch_news tool."""

    topic: str = "top news"


class BrowseUrlArgs(BaseModel):
    """Validated arguments for the browse_url tool."""

    url: str


class InnerCall(BaseModel):
    """A single lookup inside a MultiTool call — exactly one key set.

    Uses single-key flat objects: {"search": "query"}, {"browse_url": "url"},
    or {"fetch_news": "topic"}.
    """

    search: str | None = None
    browse_url: str | None = None
    fetch_news: str | None = None

    @property
    def tool_name(self) -> str:
        if self.search is not None:
            return "search"
        if self.browse_url is not None:
            return "browse_url"
        return "fetch_news"

    @property
    def value(self) -> str:
        return self.search or self.browse_url or self.fetch_news or ""


class MultiToolArgs(BaseModel):
    """Validated arguments for the multi-tool."""

    calls: list[InnerCall]
    reasoning: str | None = None


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

    folder: str | None = None
    limit: int = 10


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

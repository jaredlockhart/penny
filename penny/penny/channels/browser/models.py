"""Pydantic models for browser extension WebSocket protocol."""

from __future__ import annotations

from pydantic import BaseModel

# Incoming message types (browser → server)
BROWSER_MSG_TYPE_MESSAGE = "message"
BROWSER_MSG_TYPE_TOOL_RESPONSE = "tool_response"

# Outgoing message types (server → browser)
BROWSER_RESP_TYPE_MESSAGE = "message"
BROWSER_RESP_TYPE_TYPING = "typing"
BROWSER_RESP_TYPE_STATUS = "status"
BROWSER_RESP_TYPE_TOOL_REQUEST = "tool_request"


class BrowserIncoming(BaseModel):
    """A chat message received from the browser extension."""

    type: str
    content: str = ""
    sender: str = "browser-user"


class BrowserToolResponse(BaseModel):
    """A tool execution result from the browser extension."""

    type: str
    request_id: str
    result: str | None = None
    error: str | None = None


class BrowserOutgoing(BaseModel):
    """A message sent to the browser extension."""

    type: str
    content: str = ""
    active: bool | None = None
    connected: bool | None = None


class BrowserToolRequest(BaseModel):
    """A tool execution request sent to the browser extension."""

    type: str = BROWSER_RESP_TYPE_TOOL_REQUEST
    request_id: str
    tool: str
    arguments: dict

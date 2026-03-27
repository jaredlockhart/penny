"""Pydantic models for browser extension WebSocket protocol."""

from __future__ import annotations

from pydantic import BaseModel

# Incoming message types
BROWSER_MSG_TYPE_MESSAGE = "message"

# Outgoing message types
BROWSER_RESP_TYPE_MESSAGE = "message"
BROWSER_RESP_TYPE_TYPING = "typing"
BROWSER_RESP_TYPE_STATUS = "status"


class BrowserIncoming(BaseModel):
    """A message received from the browser extension."""

    type: str
    content: str = ""
    sender: str = "browser-user"


class BrowserOutgoing(BaseModel):
    """A message sent to the browser extension."""

    type: str
    content: str = ""
    active: bool | None = None
    connected: bool | None = None

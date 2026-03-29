"""Pydantic models for browser extension WebSocket protocol."""

from __future__ import annotations

from pydantic import BaseModel

from penny.channels.base import PageContext

# Incoming message types (browser → server)
BROWSER_MSG_TYPE_MESSAGE = "message"
BROWSER_MSG_TYPE_TOOL_RESPONSE = "tool_response"
BROWSER_MSG_TYPE_THOUGHTS_REQUEST = "thoughts_request"
BROWSER_MSG_TYPE_THOUGHT_REACTION = "thought_reaction"
BROWSER_MSG_TYPE_PREFERENCES_REQUEST = "preferences_request"
BROWSER_MSG_TYPE_PREFERENCE_ADD = "preference_add"
BROWSER_MSG_TYPE_PREFERENCE_DELETE = "preference_delete"
BROWSER_MSG_TYPE_HEARTBEAT = "heartbeat"
BROWSER_MSG_TYPE_CONFIG_REQUEST = "config_request"
BROWSER_MSG_TYPE_CONFIG_UPDATE = "config_update"
BROWSER_MSG_TYPE_REGISTER = "register"

# Outgoing message types (server → browser)
BROWSER_RESP_TYPE_MESSAGE = "message"
BROWSER_RESP_TYPE_TYPING = "typing"
BROWSER_RESP_TYPE_STATUS = "status"
BROWSER_RESP_TYPE_TOOL_REQUEST = "tool_request"
BROWSER_RESP_TYPE_THOUGHTS = "thoughts_response"
BROWSER_RESP_TYPE_PREFERENCES = "preferences_response"
BROWSER_RESP_TYPE_CONFIG = "config_response"


class BrowserIncoming(BaseModel):
    """A chat message received from the browser extension."""

    type: str
    content: str = ""
    sender: str = "browser-user"
    page_context: PageContext | None = None


class BrowserToolResponse(BaseModel):
    """A tool execution result from the browser extension."""

    type: str
    request_id: str
    result: str | None = None
    error: str | None = None


class BrowserOutgoing(BaseModel):
    """A message sent to the browser extension."""

    type: str
    content: str | None = None
    active: bool | None = None
    connected: bool | None = None


class BrowserToolRequest(BaseModel):
    """A tool execution request sent to the browser extension."""

    type: str = BROWSER_RESP_TYPE_TOOL_REQUEST
    request_id: str
    tool: str
    arguments: dict


class BrowserPreferencesRequest(BaseModel):
    """A request to list preferences by valence."""

    type: str
    valence: str


class BrowserPreferenceAdd(BaseModel):
    """A request to add a new preference."""

    type: str
    valence: str
    content: str


class BrowserPreferenceDelete(BaseModel):
    """A request to delete a preference by ID."""

    type: str
    preference_id: int


class BrowserConfigUpdate(BaseModel):
    """A request to update a runtime config param."""

    type: str
    key: str
    value: str


class BrowserRegister(BaseModel):
    """Addon registers its device label on connect."""

    type: str
    sender: str

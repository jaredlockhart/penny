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
BROWSER_MSG_TYPE_CAPABILITIES_UPDATE = "capabilities_update"
BROWSER_MSG_TYPE_DOMAIN_UPDATE = "domain_update"
BROWSER_MSG_TYPE_DOMAIN_DELETE = "domain_delete"
BROWSER_MSG_TYPE_PERMISSION_REQUEST = "permission_request"
BROWSER_MSG_TYPE_PERMISSION_DECISION = "permission_decision"
BROWSER_MSG_TYPE_SCHEDULES_REQUEST = "schedules_request"
BROWSER_MSG_TYPE_SCHEDULE_ADD = "schedule_add"
BROWSER_MSG_TYPE_SCHEDULE_UPDATE = "schedule_update"
BROWSER_MSG_TYPE_SCHEDULE_DELETE = "schedule_delete"

# Outgoing message types (server → browser)
BROWSER_RESP_TYPE_MESSAGE = "message"
BROWSER_RESP_TYPE_TYPING = "typing"
BROWSER_RESP_TYPE_STATUS = "status"
BROWSER_RESP_TYPE_TOOL_REQUEST = "tool_request"
BROWSER_RESP_TYPE_THOUGHTS = "thoughts_response"
BROWSER_RESP_TYPE_PREFERENCES = "preferences_response"
BROWSER_RESP_TYPE_CONFIG = "config_response"
BROWSER_RESP_TYPE_DOMAIN_PERMISSIONS = "domain_permissions_sync"
BROWSER_RESP_TYPE_PERMISSION_PROMPT = "permission_prompt"
BROWSER_RESP_TYPE_PERMISSION_DISMISS = "permission_dismiss"
BROWSER_RESP_TYPE_SCHEDULES = "schedules_response"


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
    image: str | None = None


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


class BrowserCapabilitiesUpdate(BaseModel):
    """Addon declares its tool-use capability."""

    type: str
    tool_use_enabled: bool


class BrowserDomainUpdate(BaseModel):
    """A request to add or update a domain permission."""

    type: str
    domain: str
    permission: str


class BrowserDomainDelete(BaseModel):
    """A request to delete a domain permission."""

    type: str
    domain: str


class DomainPermissionRecord(BaseModel):
    """A single domain permission entry for sync payloads."""

    domain: str
    permission: str


class BrowserDomainPermissionsSync(BaseModel):
    """Full domain permissions list sent to all connected addons."""

    type: str = BROWSER_RESP_TYPE_DOMAIN_PERMISSIONS
    permissions: list[DomainPermissionRecord]


class BrowserPermissionRequest(BaseModel):
    """Addon reports it needs a domain permission decision."""

    type: str
    request_id: str
    domain: str
    url: str


class BrowserPermissionDecision(BaseModel):
    """Addon or Signal user decided on a domain permission."""

    type: str
    request_id: str
    allowed: bool


class BrowserPermissionPrompt(BaseModel):
    """Server asks an addon to show a permission dialog."""

    type: str = BROWSER_RESP_TYPE_PERMISSION_PROMPT
    request_id: str
    domain: str
    url: str


class BrowserPermissionDismiss(BaseModel):
    """Server tells addons to close a pending permission dialog."""

    type: str = BROWSER_RESP_TYPE_PERMISSION_DISMISS
    request_id: str


class BrowserScheduleAdd(BaseModel):
    """A request to add a new schedule via natural language."""

    type: str
    command: str


class BrowserScheduleUpdate(BaseModel):
    """A request to update a schedule's prompt text."""

    type: str
    schedule_id: int
    prompt_text: str


class BrowserScheduleDelete(BaseModel):
    """A request to delete a schedule by ID."""

    type: str
    schedule_id: int


class ScheduleRecord(BaseModel):
    """A single schedule entry for response payloads."""

    id: int
    timing_description: str
    prompt_text: str
    cron_expression: str

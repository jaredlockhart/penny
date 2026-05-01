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
BROWSER_MSG_TYPE_PREFERENCE_THOUGHTS_REQUEST = "preference_thoughts_request"
BROWSER_MSG_TYPE_PROMPT_LOGS_REQUEST = "prompt_logs_request"
BROWSER_MSG_TYPE_MEMORIES_REQUEST = "memories_request"
BROWSER_MSG_TYPE_MEMORY_DETAIL_REQUEST = "memory_detail_request"

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
BROWSER_RESP_TYPE_PROMPT_LOGS = "prompt_logs_response"
BROWSER_RESP_TYPE_PREFERENCE_THOUGHTS = "preference_thoughts_response"
BROWSER_RESP_TYPE_PROMPT_LOG_UPDATE = "prompt_log_update"
BROWSER_RESP_TYPE_RUN_OUTCOME = "run_outcome_update"
BROWSER_RESP_TYPE_MEMORIES = "memories_response"
BROWSER_RESP_TYPE_MEMORY_DETAIL = "memory_detail_response"
BROWSER_RESP_TYPE_MEMORY_CHANGED = "memory_changed"


class BrowserIncoming(BaseModel):
    """A chat message received from the browser extension."""

    type: str
    content: str
    sender: str
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


class BrowserThoughtsRequest(BaseModel):
    """A request from the addon for thought feed data.

    `notified_pages` controls how many pages of notified thoughts to return.
    The server owns the page size; the addon only counts pages.
    """

    type: str
    notified_pages: int = 1


class ThoughtCard(BaseModel):
    """A single thought as serialized for the addon feed."""

    id: int
    title: str
    content: str
    image: str
    created_at: str
    notified: bool
    seed_topic: str


class BrowserThoughtsResponse(BaseModel):
    """The thoughts feed payload sent to the addon."""

    type: str = BROWSER_RESP_TYPE_THOUGHTS
    unnotified: list[ThoughtCard]
    notified: list[ThoughtCard]
    notified_has_more: bool


class BrowserPreferenceThoughtsRequest(BaseModel):
    """A request to list thoughts for a specific preference."""

    type: str
    preference_id: int


class BrowserMemoryDetailRequest(BaseModel):
    """A request to load entries + metadata for a single memory."""

    type: str
    name: str


class MemoryRecord(BaseModel):
    """One memory's metadata for the addon's Memories tab list view."""

    name: str
    type: str  # "collection" | "log"
    description: str
    recall: str  # "off" | "recent" | "relevant" | "all"
    archived: bool
    extraction_prompt: str | None
    collector_interval_seconds: int | None
    last_collected_at: str | None
    entry_count: int


class MemoryEntryRecord(BaseModel):
    """One memory entry as serialized for the drill-in view."""

    id: int
    key: str | None
    content: str
    author: str
    created_at: str


class BrowserMemoriesResponse(BaseModel):
    """Full list of memories sent to the addon for the Memories tab."""

    type: str = BROWSER_RESP_TYPE_MEMORIES
    memories: list[MemoryRecord]


class BrowserMemoryDetailResponse(BaseModel):
    """One memory's metadata + entries (newest-first)."""

    type: str = BROWSER_RESP_TYPE_MEMORY_DETAIL
    memory: MemoryRecord
    entries: list[MemoryEntryRecord]


class BrowserMemoryChanged(BaseModel):
    """Push notification: a memory was mutated.  ``name`` is the affected
    memory, or ``None`` for fan-out events not scoped to one memory."""

    type: str = BROWSER_RESP_TYPE_MEMORY_CHANGED
    name: str | None = None

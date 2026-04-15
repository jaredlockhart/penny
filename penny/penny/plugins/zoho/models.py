"""Pydantic models for Zoho API data (Mail, Calendar, Projects)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class ZohoCredentials(BaseModel):
    """Zoho OAuth credentials for API access."""

    client_id: str
    client_secret: str
    refresh_token: str


class ZohoSession(BaseModel):
    """Cached Zoho OAuth session data."""

    access_token: str
    expires_at: float  # Unix timestamp when token expires
    api_domain: str = "https://mail.zoho.com"  # API domain from token response


class ZohoAccount(BaseModel):
    """Zoho Mail account information."""

    account_id: str
    email_address: str
    display_name: str | None = None


class ZohoFolder(BaseModel):
    """Zoho Mail folder information."""

    folder_id: str
    folder_name: str
    folder_type: str  # Inbox, Sent, Drafts, Trash, Spam, etc.
    path: str  # e.g., "/Inbox", "/Sent"
    is_archived: bool = False


class ZohoCalendarInfo(BaseModel):
    """Zoho Calendar metadata."""

    caluid: str
    name: str
    color: str | None = None
    timezone: str | None = None
    is_default: bool = False


class ZohoEvent(BaseModel):
    """Zoho Calendar event."""

    uid: str
    title: str
    start: datetime | None = None
    end: datetime | None = None
    timezone: str | None = None
    description: str | None = None
    location: str | None = None
    is_allday: bool = False
    attendees: list[str] = []
    etag: int | None = None  # Required for updates
    is_recurring: bool = False  # Whether this is a recurring event
    recurrenceid: str | None = None  # For identifying specific occurrence
    rrule: str | None = None  # Recurrence rule for recurring events


class ZohoPortal(BaseModel):
    """Zoho Projects portal information."""

    id: str
    name: str
    is_default: bool = False


class ZohoProject(BaseModel):
    """Zoho Projects project information."""

    id: str
    name: str
    status: str | None = None
    description: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    owner_name: str | None = None


class ZohoTaskList(BaseModel):
    """Zoho Projects task list (milestone)."""

    id: str
    name: str
    status: str | None = None
    flag: str | None = None  # "internal" or "external"


class ZohoTask(BaseModel):
    """Zoho Projects task."""

    id: str
    name: str
    status: str | None = None
    priority: str | None = None  # "none", "low", "medium", "high"
    description: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    completion_percentage: int = 0
    tasklist_id: str | None = None
    tasklist_name: str | None = None
    owners: list[str] = []  # List of owner names/ZPUIDs

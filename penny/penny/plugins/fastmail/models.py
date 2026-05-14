"""Pydantic models specific to the Fastmail JMAP client."""

from __future__ import annotations

from pydantic import BaseModel


class JmapSession(BaseModel):
    """Cached JMAP session data."""

    api_url: str
    account_id: str

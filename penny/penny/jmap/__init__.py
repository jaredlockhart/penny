"""Fastmail JMAP client for email search."""

from penny.jmap.client import JmapClient
from penny.jmap.models import EmailAddress, EmailDetail, EmailSummary, JmapSession

__all__ = [
    "EmailAddress",
    "EmailDetail",
    "EmailSummary",
    "JmapClient",
    "JmapSession",
]

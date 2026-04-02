"""Email client abstraction for multiple providers."""

from penny.email.models import EmailAddress, EmailDetail, EmailSummary
from penny.email.protocol import EmailClient

__all__ = [
    "EmailAddress",
    "EmailClient",
    "EmailDetail",
    "EmailSummary",
]

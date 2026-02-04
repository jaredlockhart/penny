"""Pydantic models for Signal API message structures."""

from enum import StrEnum

from pydantic import BaseModel, Field


class HttpMethod(StrEnum):
    """HTTP methods for API requests."""

    PUT = "PUT"
    DELETE = "DELETE"


class Quote(BaseModel):
    """Quoted/replied message from Signal."""

    id: int
    author: str | None = None
    authorNumber: str | None = Field(default=None, alias="authorNumber")
    authorUuid: str | None = Field(default=None, alias="authorUuid")
    text: str | None = None

    class Config:
        populate_by_name = True


class DataMessage(BaseModel):
    """Data message from Signal."""

    timestamp: int
    message: str
    expiresInSeconds: int = Field(default=0, alias="expiresInSeconds")
    isExpirationUpdate: bool = Field(default=False, alias="isExpirationUpdate")
    viewOnce: bool = Field(default=False, alias="viewOnce")
    quote: Quote | None = None

    class Config:
        populate_by_name = True


class TypingMessage(BaseModel):
    """Typing indicator message."""

    action: str
    timestamp: int


class InnerEnvelope(BaseModel):
    """Inner envelope containing actual message data."""

    source: str
    sourceNumber: str = Field(alias="sourceNumber")
    sourceUuid: str = Field(alias="sourceUuid")
    sourceName: str | None = Field(default=None, alias="sourceName")
    sourceDevice: int = Field(alias="sourceDevice")
    timestamp: int
    serverReceivedTimestamp: int = Field(alias="serverReceivedTimestamp")
    serverDeliveredTimestamp: int = Field(alias="serverDeliveredTimestamp")
    dataMessage: DataMessage | None = Field(default=None, alias="dataMessage")
    typingMessage: TypingMessage | None = Field(default=None, alias="typingMessage")

    class Config:
        populate_by_name = True


class SignalEnvelope(BaseModel):
    """Top-level Signal WebSocket message envelope."""

    envelope: InnerEnvelope
    account: str


class SendMessageRequest(BaseModel):
    """Request to send a Signal message."""

    message: str
    number: str
    recipients: list[str]
    base64_attachments: list[str] | None = None
    text_mode: str | None = "styled"  # Enable markdown-style formatting

    def __str__(self) -> str:
        attachments = (
            [f"<{len(a)} chars>" for a in self.base64_attachments]
            if self.base64_attachments
            else None
        )
        return (
            f"SendMessageRequest(message={self.message!r}, number={self.number}, "
            f"recipients={self.recipients}, base64_attachments={attachments})"
        )


class SendMessageResponse(BaseModel):
    """Response from sending a Signal message."""

    timestamp: int | None = None


class TypingIndicatorRequest(BaseModel):
    """Request to send a typing indicator."""

    recipient: str

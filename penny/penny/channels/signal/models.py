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


class ReactionEmoji(BaseModel):
    """Emoji used in a reaction."""

    value: str  # The emoji unicode character


class Reaction(BaseModel):
    """Reaction message from Signal."""

    emoji: str | ReactionEmoji
    targetAuthor: str = Field(alias="targetAuthor")
    targetAuthorNumber: str = Field(alias="targetAuthorNumber")
    targetSentTimestamp: int = Field(alias="targetSentTimestamp")
    isRemove: bool = Field(default=False, alias="isRemove")

    class Config:
        populate_by_name = True


class SignalAttachment(BaseModel):
    """Attachment metadata from Signal data message."""

    contentType: str = Field(alias="contentType")
    id: str
    size: int | None = None
    filename: str | None = None

    class Config:
        populate_by_name = True


class DataMessage(BaseModel):
    """Data message from Signal."""

    timestamp: int
    message: str | None = None  # None for reactions
    expiresInSeconds: int = Field(default=0, alias="expiresInSeconds")
    isExpirationUpdate: bool = Field(default=False, alias="isExpirationUpdate")
    viewOnce: bool = Field(default=False, alias="viewOnce")
    quote: Quote | None = None
    reaction: Reaction | None = None
    attachments: list[SignalAttachment] | None = None

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
    syncMessage: dict | None = Field(default=None, alias="syncMessage")

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
    # Quote reply fields (optional)
    quote_timestamp: int | None = None
    quote_author: str | None = None
    quote_message: str | None = None

    def __str__(self) -> str:
        attachments = (
            [f"<{len(a)} chars>" for a in self.base64_attachments]
            if self.base64_attachments
            else None
        )
        quote = (
            f"quote_author={self.quote_author}, quote_timestamp={self.quote_timestamp}"
            if self.quote_timestamp
            else None
        )
        return (
            f"SendMessageRequest(message={self.message!r}, number={self.number}, "
            f"recipients={self.recipients}, base64_attachments={attachments}, {quote})"
        )


class SendMessageResponse(BaseModel):
    """Response from sending a Signal message."""

    timestamp: int | None = None


class TypingIndicatorRequest(BaseModel):
    """Request to send a typing indicator."""

    recipient: str

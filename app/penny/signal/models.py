"""Pydantic models for Signal API message structures."""

from enum import Enum

from pydantic import BaseModel, Field


class HttpMethod(str, Enum):
    """HTTP methods for API requests."""

    PUT = "PUT"
    DELETE = "DELETE"


class DataMessage(BaseModel):
    """Data message from Signal."""

    timestamp: int
    message: str
    expiresInSeconds: int = Field(default=0, alias="expiresInSeconds")
    isExpirationUpdate: bool = Field(default=False, alias="isExpirationUpdate")
    viewOnce: bool = Field(default=False, alias="viewOnce")

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


class SendMessageResponse(BaseModel):
    """Response from sending a Signal message."""

    timestamp: int | None = None


class IncomingMessage(BaseModel):
    """Extracted incoming message from Signal."""

    sender: str
    content: str


class TypingIndicatorRequest(BaseModel):
    """Request to send a typing indicator."""

    recipient: str

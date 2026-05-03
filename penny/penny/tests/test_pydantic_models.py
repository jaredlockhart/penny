"""Validation tests for Pydantic models with required fields.

These models had ``str = ""`` defaults that masked missing-field bugs at
the wire boundary. After tightening the fields to required, the model
must reject incomplete payloads with a ``ValidationError`` instead of
silently producing an instance with empty strings.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from penny.channels.base import PageContext
from penny.channels.browser.models import BrowserIncoming
from penny.channels.discord.models import DiscordUser
from penny.llm.models import LlmToolCall, LlmToolCallFunction


class TestPageContextRequiresAllFields:
    def test_full_payload_accepted(self) -> None:
        ctx = PageContext(title="Hello", url="https://example.com", text="body")
        assert ctx.title == "Hello"

    @pytest.mark.parametrize("missing", ["title", "url", "text"])
    def test_missing_field_rejected(self, missing: str) -> None:
        payload = {"title": "Hello", "url": "https://example.com", "text": "body"}
        del payload[missing]
        with pytest.raises(ValidationError, match=missing):
            PageContext(**payload)


class TestBrowserIncomingRequiresContentAndSender:
    def test_full_payload_accepted(self) -> None:
        msg = BrowserIncoming(type="message", content="hi", sender="firefox")
        assert msg.content == "hi"

    def test_missing_content_rejected(self) -> None:
        with pytest.raises(ValidationError, match="content"):
            BrowserIncoming(type="message", sender="firefox")  # ty: ignore[missing-argument]

    def test_missing_sender_rejected(self) -> None:
        with pytest.raises(ValidationError, match="sender"):
            BrowserIncoming(type="message", content="hi")  # ty: ignore[missing-argument]


class TestDiscordUserRequiresDiscriminator:
    def test_missing_discriminator_rejected(self) -> None:
        with pytest.raises(ValidationError, match="discriminator"):
            DiscordUser(id="1", username="alice")  # ty: ignore[missing-argument]


class TestLlmToolCallRequiresIdAndFunctionName:
    def test_missing_function_name_rejected(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            LlmToolCallFunction(arguments={})  # ty: ignore[missing-argument]

    def test_missing_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="id"):
            LlmToolCall(function=LlmToolCallFunction(name="search"))  # ty: ignore[missing-argument]

    def test_missing_function_rejected(self) -> None:
        with pytest.raises(ValidationError, match="function"):
            LlmToolCall(id="call_1")  # ty: ignore[missing-argument]

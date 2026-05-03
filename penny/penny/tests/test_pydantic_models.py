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
from penny.tools.memory_args import CollectionEntrySpec, CollectionWriteArgs


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


class TestCollectionEntrySpecDescriptionAlias:
    """LLMs confuse top-level 'description' with per-entry 'content' — both must be accepted."""

    def test_content_field_accepted(self) -> None:
        entry = CollectionEntrySpec.model_validate({"key": "k", "content": "c"})
        assert entry.content == "c"

    def test_description_alias_accepted(self) -> None:
        entry = CollectionEntrySpec.model_validate({"key": "k", "description": "c"})
        assert entry.content == "c"

    def test_kwargs_construction_still_works(self) -> None:
        entry = CollectionEntrySpec(key="k", content="c")
        assert entry.content == "c"

    def test_collection_write_args_with_description_entries(self) -> None:
        args = CollectionWriteArgs.model_validate(
            {
                "memory": "places",
                "entries": [
                    {"key": "cafe", "description": "Nice rooftop bar"},
                    {"key": "restaurant", "content": "Italian place downtown"},
                ],
            }
        )
        assert args.entries[0].content == "Nice rooftop bar"
        assert args.entries[1].content == "Italian place downtown"


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

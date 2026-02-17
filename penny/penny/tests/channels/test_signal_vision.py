"""Integration tests for Signal vision (image attachment) handling."""

import pytest

from penny.prompts import VISION_AUTO_DESCRIBE_PROMPT, VISION_RESPONSE_PROMPT
from penny.responses import PennyResponse
from penny.tests.conftest import TEST_SENDER, wait_until

# Minimal valid JPEG header bytes for testing
FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100


@pytest.mark.asyncio
async def test_image_with_text_captions_then_forwards(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Image + text: vision model captions, then foreground model responds without search."""
    config = make_config(ollama_vision_model="test-vision-model")

    def handler(request, count):
        if count == 1:
            # Vision model captioning call
            return mock_ollama._make_text_response(request, "a cute orange cat")
        # Foreground model: direct response without tool call
        return mock_ollama._make_text_response(request, "that's a cute cat! 🐱")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config):
        await signal_server.push_image_message(
            sender=TEST_SENDER,
            image_data=FAKE_JPEG,
            text="what's in this photo?",
        )

        response = await signal_server.wait_for_message(timeout=10.0)
        assert "cat" in response["message"].lower()

        # Verify two-step flow: vision model called first, then foreground model
        await wait_until(lambda: len(mock_ollama.requests) >= 2)

        # First call: vision model with images
        caption_request = mock_ollama.requests[0]
        assert caption_request["model"] == "test-vision-model"
        user_msgs = [m for m in caption_request["messages"] if m["role"] == "user"]
        assert any("images" in m for m in user_msgs)
        assert any(VISION_AUTO_DESCRIBE_PROMPT in m.get("content", "") for m in user_msgs)

        # Second call: foreground model with combined text prompt (no images, no tools)
        foreground_request = mock_ollama.requests[1]
        assert foreground_request["model"] == "test-model"
        user_msgs = [m for m in foreground_request["messages"] if m["role"] == "user"]
        assert not any("images" in m for m in user_msgs)
        # Combined prompt should contain user text and caption
        expected = PennyResponse.VISION_IMAGE_CONTEXT.format(
            user_text="what's in this photo?", caption="a cute orange cat"
        )
        assert any(expected in m.get("content", "") for m in user_msgs)
        # Verify no tools were provided (None = tools disabled)
        assert foreground_request.get("tools") is None
        # Verify vision response prompt (not search prompt) was used
        system_msgs = [m for m in foreground_request["messages"] if m["role"] == "system"]
        system_text = system_msgs[0]["content"]
        assert "sent an image" in system_text
        assert VISION_RESPONSE_PROMPT in system_text


@pytest.mark.asyncio
async def test_image_without_text_captions_then_forwards(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Image with no text: vision model captions, forwarded without search."""
    config = make_config(ollama_vision_model="test-vision-model")

    def handler(request, count):
        if count == 1:
            # Vision model captioning
            return mock_ollama._make_text_response(request, "a sunset over the ocean")
        # Foreground model: direct response without tool call
        return mock_ollama._make_text_response(request, "beautiful sunset! 🌅")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config):
        await signal_server.push_image_message(
            sender=TEST_SENDER,
            image_data=FAKE_JPEG,
            text=None,  # No text
        )

        response = await signal_server.wait_for_message(timeout=10.0)
        assert "sunset" in response["message"].lower()

        await wait_until(lambda: len(mock_ollama.requests) >= 2)

        # First call: vision model captioning with describe prompt
        caption_request = mock_ollama.requests[0]
        assert caption_request["model"] == "test-vision-model"

        # Second call: foreground model with image-only context (no images, no tools)
        foreground_request = mock_ollama.requests[1]
        assert foreground_request["model"] == "test-model"
        user_msgs = [m for m in foreground_request["messages"] if m["role"] == "user"]
        assert not any("images" in m for m in user_msgs)
        expected = PennyResponse.VISION_IMAGE_ONLY_CONTEXT.format(caption="a sunset over the ocean")
        assert any(expected in m.get("content", "") for m in user_msgs)
        # Verify no tools were provided (None = tools disabled)
        assert foreground_request.get("tools") is None
        # Verify vision response prompt (not search prompt) was used
        system_msgs = [m for m in foreground_request["messages"] if m["role"] == "system"]
        system_text = system_msgs[0]["content"]
        assert "sent an image" in system_text
        assert VISION_RESPONSE_PROMPT in system_text


@pytest.mark.asyncio
async def test_image_without_vision_model_sends_acknowledgment(
    signal_server,
    mock_ollama,
    _mock_search,
    test_config,
    test_user_info,
    running_penny,
):
    """When vision model is not configured, send acknowledgment message."""
    # test_config has no ollama_vision_model (None by default)
    async with running_penny(test_config):
        await signal_server.push_image_message(
            sender=TEST_SENDER,
            image_data=FAKE_JPEG,
            text="what's this?",
        )

        response = await signal_server.wait_for_message(timeout=10.0)
        assert PennyResponse.VISION_NOT_CONFIGURED_MESSAGE in response["message"]

        # Verify Ollama was NOT called
        assert len(mock_ollama.requests) == 0


@pytest.mark.asyncio
async def test_non_image_attachment_ignored(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Non-image attachments (e.g., PDF) don't trigger vision pipeline."""
    config = make_config(ollama_vision_model="test-vision-model")
    mock_ollama.set_default_flow(
        search_query="document",
        final_response="here's what I found about documents 📄",
    )

    async with running_penny(config):
        await signal_server.push_image_message(
            sender=TEST_SENDER,
            image_data=b"%PDF-1.4...",
            content_type="application/pdf",
            text="check this document",
        )

        await signal_server.wait_for_message(timeout=10.0)

        # Should process as normal text message (no images in request)
        await wait_until(lambda: len(mock_ollama.requests) >= 1)
        first_request = mock_ollama.requests[0]
        user_messages = [m for m in first_request["messages"] if m["role"] == "user"]
        assert not any("images" in m for m in user_messages)

        # Should use the foreground model, not vision model
        assert first_request["model"] == "test-model"

"""Integration tests for /draw command."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from penny.commands.draw import DrawCommand
from penny.commands.models import CommandContext
from penny.config import Config
from penny.tests.conftest import TEST_SENDER

FAKE_IMAGE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
TEST_IMAGE_MODEL = "x/z-image-turbo"


@pytest.fixture
def mock_ollama_client():
    """Create a mock OllamaClient with generate_image support."""
    client = AsyncMock()
    client.generate_image.return_value = FAKE_IMAGE_B64
    return client


@pytest.fixture
def draw_context(mock_ollama_client):
    """Create a CommandContext for draw command tests."""
    config = MagicMock(spec=Config)
    return CommandContext(
        db=MagicMock(),
        config=config,
        ollama_client=mock_ollama_client,
        user=TEST_SENDER,
        channel_type="signal",
        start_time=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_draw_generates_image(mock_ollama_client, draw_context):
    """Test /draw generates an image and returns it as an attachment."""
    cmd = DrawCommand(TEST_IMAGE_MODEL)
    result = await cmd.execute("a cute cat wearing a top hat", draw_context)

    assert result.text == ""
    assert result.attachments is not None
    assert len(result.attachments) == 1
    assert result.attachments[0] == FAKE_IMAGE_B64
    mock_ollama_client.generate_image.assert_called_once_with(
        prompt="a cute cat wearing a top hat", model=TEST_IMAGE_MODEL
    )


@pytest.mark.asyncio
async def test_draw_empty_prompt(draw_context):
    """Test /draw with no prompt shows usage."""
    cmd = DrawCommand(TEST_IMAGE_MODEL)
    result = await cmd.execute("", draw_context)

    assert "Usage: /draw" in result.text
    assert result.attachments is None


@pytest.mark.asyncio
async def test_draw_whitespace_only_prompt(draw_context):
    """Test /draw with whitespace-only prompt shows usage."""
    cmd = DrawCommand(TEST_IMAGE_MODEL)
    result = await cmd.execute("   ", draw_context)

    assert "Usage: /draw" in result.text
    assert result.attachments is None


@pytest.mark.asyncio
async def test_draw_ollama_error(mock_ollama_client, draw_context):
    """Test /draw handles Ollama errors gracefully."""
    mock_ollama_client.generate_image.side_effect = RuntimeError("Model not found")
    cmd = DrawCommand(TEST_IMAGE_MODEL)
    result = await cmd.execute("a sunset", draw_context)

    assert "Failed to generate image" in result.text
    assert "Model not found" in result.text
    assert result.attachments is None

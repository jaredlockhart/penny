"""Integration tests for /email command (multi-provider routing + Fastmail provider)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from penny.commands.models import CommandContext
from penny.config import Config
from penny.email.command import EmailCommand
from penny.email.models import EmailAddress, EmailDetail, EmailSummary
from penny.plugins.fastmail.commands import FastmailEmailCommand
from penny.responses import PennyResponse
from penny.tests.conftest import TEST_SENDER
from penny.tools.read_emails import NO_EMAILS_TO_READ, ReadEmailsTool

FAKE_TOKEN = "fmu1-test-token"

SAMPLE_SUMMARIES = [
    EmailSummary(
        id="M001",
        subject="Your package has shipped!",
        from_addresses=[EmailAddress(name="Amazon", email="ship-confirm@amazon.com")],
        received_at="2026-02-10T14:30:00Z",
        preview="Your order #123-456 has been shipped and is on its way...",
    ),
    EmailSummary(
        id="M002",
        subject="Delivery scheduled for tomorrow",
        from_addresses=[EmailAddress(name="UPS", email="noreply@ups.com")],
        received_at="2026-02-10T10:00:00Z",
        preview="Your package is scheduled for delivery on Feb 11...",
    ),
]

SAMPLE_DETAIL = EmailDetail(
    id="M001",
    subject="Your package has shipped!",
    from_addresses=[EmailAddress(name="Amazon", email="ship-confirm@amazon.com")],
    to_addresses=[EmailAddress(name="Test User", email="test@fastmail.com")],
    received_at="2026-02-10T14:30:00Z",
    text_body=(
        "Your order #123-456 has been shipped!\n\n"
        "Tracking number: 1Z999AA10123456784\n"
        "Estimated delivery: February 12, 2026"
    ),
)


@pytest.fixture
def mock_jmap_client():
    """Create a mock JmapClient."""
    client = AsyncMock()
    client.search_emails.return_value = SAMPLE_SUMMARIES
    client.read_emails.return_value = [SAMPLE_DETAIL]
    client.close.return_value = None
    return client


@pytest.fixture
def email_context():
    """Create a CommandContext for email command tests."""
    config = MagicMock(spec=Config)
    config.email_max_steps = 5
    config.tool_timeout = 60.0
    runtime = MagicMock()
    runtime.JMAP_REQUEST_TIMEOUT = 30.0
    runtime.EMAIL_BODY_MAX_LENGTH = 4000
    config.runtime = runtime
    return CommandContext(
        db=MagicMock(),
        config=config,
        model_client=MagicMock(),
        user=TEST_SENDER,
        channel_type="signal",
        start_time=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_email_empty_prompt(email_context):
    """Test /email with no args returns usage text."""
    cmd = FastmailEmailCommand(FAKE_TOKEN)
    result = await cmd.execute("", email_context)

    assert result.text == PennyResponse.EMAIL_NO_QUERY_TEXT


@pytest.mark.asyncio
async def test_email_whitespace_only_prompt(email_context):
    """Test /email with whitespace-only args returns usage text."""
    cmd = FastmailEmailCommand(FAKE_TOKEN)
    result = await cmd.execute("   ", email_context)

    assert result.text == PennyResponse.EMAIL_NO_QUERY_TEXT


@pytest.mark.asyncio
async def test_email_search_and_answer(mock_jmap_client, email_context):
    """Test FastmailEmailCommand runs the agent loop and returns an answer."""
    mock_response = MagicMock()
    mock_response.answer = "You have 2 packages coming! One from Amazon arriving Feb 12."

    with (
        patch("penny.plugins.fastmail.commands.JmapClient", return_value=mock_jmap_client),
        patch("penny.plugins.fastmail.commands.Agent") as mock_agent_cls,
    ):
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run.return_value = mock_response
        mock_agent_cls.return_value = mock_agent_instance

        cmd = FastmailEmailCommand(FAKE_TOKEN)
        result = await cmd.execute("what packages am I expecting", email_context)

    assert "packages" in result.text.lower()
    mock_agent_instance.run.assert_called_once_with(
        "what packages am I expecting", max_steps=email_context.config.email_max_steps
    )
    mock_agent_instance.close.assert_called_once()
    mock_jmap_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_email_agent_created_with_repeat_tools(mock_jmap_client, email_context):
    """Test that the email agent is created with allow_repeat_tools=True."""
    with (
        patch("penny.plugins.fastmail.commands.JmapClient", return_value=mock_jmap_client),
        patch("penny.plugins.fastmail.commands.Agent") as mock_agent_cls,
    ):
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run.return_value = MagicMock(answer="test")
        mock_agent_cls.return_value = mock_agent_instance

        cmd = FastmailEmailCommand(FAKE_TOKEN)
        await cmd.execute("check my email", email_context)

    call_kwargs = mock_agent_cls.call_args
    assert call_kwargs.kwargs["allow_repeat_tools"] is True


@pytest.mark.asyncio
async def test_email_agent_cleanup_on_error(mock_jmap_client, email_context):
    """Test that agent and JMAP client are cleaned up even on error."""
    with (
        patch("penny.plugins.fastmail.commands.JmapClient", return_value=mock_jmap_client),
        patch("penny.plugins.fastmail.commands.Agent") as mock_agent_cls,
    ):
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run.side_effect = RuntimeError("Ollama down")
        mock_agent_cls.return_value = mock_agent_instance

        cmd = FastmailEmailCommand(FAKE_TOKEN)
        result = await cmd.execute("check my email", email_context)

    assert "Failed to search email" in result.text
    mock_agent_instance.close.assert_called_once()
    mock_jmap_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_email_jmap_client_created_with_token(email_context):
    """Test that JmapClient is created with the configured token."""
    with (
        patch("penny.plugins.fastmail.commands.JmapClient") as mock_jmap_cls,
        patch("penny.plugins.fastmail.commands.Agent") as mock_agent_cls,
    ):
        mock_client = AsyncMock()
        mock_jmap_cls.return_value = mock_client

        mock_agent_instance = AsyncMock()
        mock_agent_instance.run.return_value = MagicMock(answer="test")
        mock_agent_cls.return_value = mock_agent_instance

        cmd = FastmailEmailCommand(FAKE_TOKEN)
        await cmd.execute("anything", email_context)

    mock_jmap_cls.assert_called_once_with(
        FAKE_TOKEN,
        timeout=30.0,
        max_body_length=4000,
    )


@pytest.mark.asyncio
async def test_email_command_routes_to_single_plugin(mock_jmap_client, email_context):
    """Test /email routes to the only active email plugin when one is configured."""
    mock_plugin = MagicMock()
    mock_plugin.name = "fastmail"
    mock_plugin.capabilities = ["email"]

    mock_cmd = AsyncMock()
    mock_cmd.execute.return_value = MagicMock(text="2 packages coming")
    mock_plugin.get_commands.return_value = [mock_cmd]

    email_cmd = EmailCommand([mock_plugin])
    result = await email_cmd.execute("what packages", email_context)

    mock_cmd.execute.assert_called_once_with("what packages", email_context)
    assert result.text == "2 packages coming"


@pytest.mark.asyncio
async def test_email_command_requires_provider_when_multiple(email_context):
    """Test /email returns disambiguation error when multiple providers active."""
    mock_zoho = MagicMock()
    mock_zoho.name = "zoho"
    mock_fastmail = MagicMock()
    mock_fastmail.name = "fastmail"

    email_cmd = EmailCommand([mock_zoho, mock_fastmail])
    result = await email_cmd.execute("what packages", email_context)

    assert "zoho" in result.text
    assert "fastmail" in result.text


@pytest.mark.asyncio
async def test_email_command_routes_by_provider_prefix(mock_jmap_client, email_context):
    """Test /email zoho <query> routes to zoho plugin when multiple active."""
    mock_zoho = MagicMock()
    mock_zoho.name = "zoho"
    mock_fastmail = MagicMock()
    mock_fastmail.name = "fastmail"

    mock_cmd = AsyncMock()
    mock_cmd.execute.return_value = MagicMock(text="zoho result")
    mock_zoho.get_commands.return_value = [mock_cmd]

    email_cmd = EmailCommand([mock_zoho, mock_fastmail])
    result = await email_cmd.execute("zoho what packages", email_context)

    mock_cmd.execute.assert_called_once_with("what packages", email_context)
    assert result.text == "zoho result"


@pytest.mark.asyncio
async def test_read_emails_tool_returns_content():
    """Test that ReadEmailsTool returns email content directly."""
    mock_jmap = AsyncMock()
    mock_jmap.read_emails.return_value = [SAMPLE_DETAIL]

    tool = ReadEmailsTool(mock_jmap)
    result = await tool.execute(email_ids=["M001"])

    assert "Your order #123-456 has been shipped!" in result
    mock_jmap.read_emails.assert_called_once_with(["M001"])


@pytest.mark.asyncio
async def test_read_emails_tool_no_ids():
    """Test that ReadEmailsTool returns early for empty ID list."""
    mock_jmap = AsyncMock()

    tool = ReadEmailsTool(mock_jmap)
    result = await tool.execute(email_ids=[])

    assert result == NO_EMAILS_TO_READ
    mock_jmap.read_emails.assert_not_called()

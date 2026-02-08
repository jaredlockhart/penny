"""Integration tests for command system."""

import pytest

from penny.tests.conftest import TEST_SENDER


@pytest.mark.asyncio
async def test_commands_list(signal_server, test_config, mock_ollama, running_penny):
    """Test /commands lists all available commands."""
    async with running_penny(test_config) as _penny:
        # Send /commands
        await signal_server.push_message(sender=TEST_SENDER, content="/commands")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should list both commands
        assert "**Available Commands**" in response["message"]
        assert "**/commands**" in response["message"]
        assert "**/debug**" in response["message"]
        assert "List all commands" in response["message"]
        assert "diagnostic information" in response["message"]


@pytest.mark.asyncio
async def test_commands_help_specific(signal_server, test_config, mock_ollama, running_penny):
    """Test /commands <name> shows help for specific command."""
    async with running_penny(test_config) as _penny:
        # Send /commands debug
        await signal_server.push_message(sender=TEST_SENDER, content="/commands debug")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should show debug command help
        assert "**Command: /debug**" in response["message"]
        assert "diagnostic information" in response["message"]
        assert "**Usage**: `/debug`" in response["message"]


@pytest.mark.asyncio
async def test_commands_unknown(signal_server, test_config, mock_ollama, running_penny):
    """Test /commands <unknown> shows error."""
    async with running_penny(test_config) as _penny:
        # Send /commands unknown
        await signal_server.push_message(sender=TEST_SENDER, content="/commands unknown")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should show error
        assert "Unknown command: /unknown" in response["message"]
        assert "Use /commands to see available commands" in response["message"]


@pytest.mark.asyncio
async def test_debug_command(signal_server, test_config, mock_ollama, running_penny):
    """Test /debug returns diagnostic information."""
    async with running_penny(test_config) as _penny:
        # Send /debug
        await signal_server.push_message(sender=TEST_SENDER, content="/debug")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should show debug info
        assert "**Debug Information**" in response["message"]
        assert "**Git Commit**:" in response["message"]
        assert "**Uptime**:" in response["message"]
        assert "**Channel**: Signal" in response["message"]
        assert "**Database**:" in response["message"]
        assert "messages" in response["message"]
        assert "**Models**:" in response["message"]
        assert "**Agents**:" in response["message"]
        assert "**Memory**:" in response["message"]


@pytest.mark.asyncio
async def test_unknown_command(signal_server, test_config, mock_ollama, running_penny):
    """Test unknown command shows error."""
    async with running_penny(test_config) as _penny:
        # Send unknown command
        await signal_server.push_message(sender=TEST_SENDER, content="/unknown")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should show error
        assert "Unknown command: /unknown" in response["message"]
        assert "Use /commands to see available commands" in response["message"]


@pytest.mark.asyncio
async def test_command_threading_blocked(signal_server, test_config, mock_ollama, running_penny):
    """Test that thread-replying to a command is blocked."""
    async with running_penny(test_config) as _penny:
        # First, send a command
        await signal_server.push_message(sender=TEST_SENDER, content="/debug")
        response1 = await signal_server.wait_for_message(timeout=5.0)
        assert "**Debug Information**" in response1["message"]

        # Try to thread-reply to the command
        # Build quote dict matching Signal's Quote model
        quote = {"id": 1, "text": "/debug"}
        await signal_server.push_message(
            sender=TEST_SENDER, content="What does this mean?", quote=quote
        )

        # Should get threading not supported message
        response2 = await signal_server.wait_for_message(timeout=5.0)
        assert "Threading is not supported for commands" in response2["message"]


@pytest.mark.asyncio
async def test_command_logging(signal_server, test_config, mock_ollama, running_penny):
    """Test that commands are logged to the database."""
    async with running_penny(test_config) as penny:
        # Send a command
        await signal_server.push_message(sender=TEST_SENDER, content="/commands debug")
        await signal_server.wait_for_message(timeout=5.0)

        # Check database for command log
        from penny.database.models import CommandLog

        with penny.db.get_session() as session:
            from sqlmodel import select

            logs = list(session.exec(select(CommandLog)).all())
            assert len(logs) == 1
            log = logs[0]
            assert log.command_name == "commands"
            assert log.command_args == "debug"
            assert log.user == TEST_SENDER
            assert log.channel_type == "signal"
            assert "**Command: /debug**" in log.response
            assert log.error is None


@pytest.mark.asyncio
async def test_command_not_logged_to_message_table(
    signal_server, test_config, mock_ollama, running_penny
):
    """Test that commands are NOT logged to MessageLog table."""
    async with running_penny(test_config) as penny:
        # Send a command
        await signal_server.push_message(sender=TEST_SENDER, content="/debug")
        await signal_server.wait_for_message(timeout=5.0)

        # Check that no messages were logged
        from penny.database.models import MessageLog

        with penny.db.get_session() as session:
            from sqlmodel import select

            logs = list(session.exec(select(MessageLog)).all())
            assert len(logs) == 0

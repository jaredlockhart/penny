"""Integration tests for command system."""

import os

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
async def test_debug_git_commit_from_env(signal_server, test_config, mock_ollama, running_penny):
    """Test /debug shows git commit from environment variable."""
    # Set GIT_COMMIT environment variable
    old_commit = os.environ.get("GIT_COMMIT")
    os.environ["GIT_COMMIT"] = "abc1234"

    try:
        async with running_penny(test_config) as _penny:
            # Send /debug
            await signal_server.push_message(sender=TEST_SENDER, content="/debug")

            # Wait for response
            response = await signal_server.wait_for_message(timeout=5.0)

            # Should show git commit from environment
            assert "**Git Commit**: abc1234" in response["message"]
    finally:
        # Restore original environment
        if old_commit is None:
            os.environ.pop("GIT_COMMIT", None)
        else:
            os.environ["GIT_COMMIT"] = old_commit


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


@pytest.mark.asyncio
async def test_config_list(signal_server, test_config, mock_ollama, running_penny):
    """Test /config lists all available config parameters."""
    async with running_penny(test_config) as _penny:
        # Send /config
        await signal_server.push_message(sender=TEST_SENDER, content="/config")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should list all config parameters
        assert "**Runtime Configuration**" in response["message"]
        assert "MESSAGE_MAX_STEPS" in response["message"]
        assert "IDLE_SECONDS" in response["message"]
        assert "FOLLOWUP_MIN_SECONDS" in response["message"]
        assert "FOLLOWUP_MAX_SECONDS" in response["message"]
        assert "DISCOVERY_MIN_SECONDS" in response["message"]
        assert "DISCOVERY_MAX_SECONDS" in response["message"]
        assert "Use `/config <key> <value>` to change a setting" in response["message"]


@pytest.mark.asyncio
async def test_config_get_specific(signal_server, test_config, mock_ollama, running_penny):
    """Test /config <key> shows value of specific parameter."""
    async with running_penny(test_config) as _penny:
        # Send /config IDLE_SECONDS
        await signal_server.push_message(sender=TEST_SENDER, content="/config IDLE_SECONDS")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should show IDLE_SECONDS value (test config uses 99999.0)
        assert "**IDLE_SECONDS**:" in response["message"]
        assert "99999.0" in response["message"]
        assert "Global idle threshold in seconds" in response["message"]


@pytest.mark.asyncio
async def test_config_set_valid(signal_server, test_config, mock_ollama, running_penny):
    """Test /config <key> <value> updates a parameter."""
    async with running_penny(test_config) as penny:
        # Send /config IDLE_SECONDS 600
        await signal_server.push_message(sender=TEST_SENDER, content="/config IDLE_SECONDS 600")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should confirm update
        assert "ok, updated IDLE_SECONDS to 600" in response["message"]

        # Verify config was updated
        assert penny.config.idle_seconds == 600.0

        # Verify database was updated
        from penny.database.models import RuntimeConfig

        with penny.db.get_session() as session:
            from sqlmodel import select

            config_row = session.exec(
                select(RuntimeConfig).where(RuntimeConfig.key == "IDLE_SECONDS")
            ).first()
            assert config_row is not None
            assert config_row.value == "600.0"


@pytest.mark.asyncio
async def test_config_set_invalid_key(signal_server, test_config, mock_ollama, running_penny):
    """Test /config with unknown key shows error."""
    async with running_penny(test_config) as _penny:
        # Send /config FAKE_KEY 123
        await signal_server.push_message(sender=TEST_SENDER, content="/config FAKE_KEY 123")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should show error
        assert "unknown config parameter: FAKE_KEY" in response["message"]
        assert "Use /config to see all available parameters" in response["message"]


@pytest.mark.asyncio
async def test_config_set_invalid_value(signal_server, test_config, mock_ollama, running_penny):
    """Test /config with invalid value shows error."""
    async with running_penny(test_config) as _penny:
        # Send /config IDLE_SECONDS -1 (negative not allowed)
        await signal_server.push_message(sender=TEST_SENDER, content="/config IDLE_SECONDS -1")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should show error
        assert "invalid value for IDLE_SECONDS" in response["message"]
        assert "must be a positive number" in response["message"]


@pytest.mark.asyncio
async def test_config_set_non_numeric(signal_server, test_config, mock_ollama, running_penny):
    """Test /config with non-numeric value shows error."""
    async with running_penny(test_config) as _penny:
        # Send /config MESSAGE_MAX_STEPS abc
        await signal_server.push_message(
            sender=TEST_SENDER, content="/config MESSAGE_MAX_STEPS abc"
        )

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should show error
        assert "invalid value for MESSAGE_MAX_STEPS" in response["message"]
        assert "must be a positive integer" in response["message"]


@pytest.mark.asyncio
async def test_config_case_insensitive(signal_server, test_config, mock_ollama, running_penny):
    """Test /config works with lowercase keys."""
    async with running_penny(test_config) as penny:
        # Send /config idle_seconds 450 (lowercase)
        await signal_server.push_message(sender=TEST_SENDER, content="/config idle_seconds 450")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should work (key gets uppercased internally)
        assert "ok, updated IDLE_SECONDS to 450" in response["message"]
        assert penny.config.idle_seconds == 450.0


@pytest.mark.asyncio
async def test_config_persistence(signal_server, test_config, mock_ollama, running_penny):
    """Test config changes persist in database across agent restarts."""
    # First run: set a config value
    async with running_penny(test_config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/config IDLE_SECONDS 800")
        response = await signal_server.wait_for_message(timeout=5.0)
        assert "ok, updated IDLE_SECONDS to 800" in response["message"]

    # Second run: verify the value persists
    async with running_penny(test_config) as penny:
        # Config should load the value from database
        assert penny.config.idle_seconds == 800.0

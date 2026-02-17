"""Integration tests for /config command."""

import pytest

from penny.tests.conftest import TEST_SENDER


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
        assert "Ok, updated IDLE_SECONDS to 600" in response["message"]

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
        assert "Unknown config parameter: FAKE_KEY" in response["message"]
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
        assert "Invalid value for IDLE_SECONDS" in response["message"]
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
        assert "Invalid value for MESSAGE_MAX_STEPS" in response["message"]
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
        assert "Ok, updated IDLE_SECONDS to 450" in response["message"]
        assert penny.config.idle_seconds == 450.0


@pytest.mark.asyncio
async def test_config_persistence(signal_server, test_config, mock_ollama, running_penny):
    """Test config changes persist in database across agent restarts."""
    # First run: set a config value
    async with running_penny(test_config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/config IDLE_SECONDS 800")
        response = await signal_server.wait_for_message(timeout=5.0)
        assert "Ok, updated IDLE_SECONDS to 800" in response["message"]

    # Second run: verify the value persists
    async with running_penny(test_config) as penny:
        # Config should load the value from database
        assert penny.config.idle_seconds == 800.0

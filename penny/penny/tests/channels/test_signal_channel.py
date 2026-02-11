"""Integration tests for Signal channel."""

import pytest

from penny.channels.signal import SignalChannel
from penny.database import Database


@pytest.mark.asyncio
async def test_validate_connectivity_success(signal_server, test_config, mock_ollama):
    """Test that validate_connectivity succeeds with a reachable Signal API."""
    from penny.agents import MessageAgent
    from penny.constants import SYSTEM_PROMPT

    db = Database(test_config.db_path)
    db.create_tables()

    message_agent = MessageAgent(
        system_prompt=SYSTEM_PROMPT,
        model=test_config.ollama_foreground_model,
        ollama_api_url=test_config.ollama_api_url,
        tools=[],
        db=db,
        max_steps=1,
    )

    channel = SignalChannel(
        api_url=test_config.signal_api_url,
        phone_number=test_config.signal_number or "+15551234567",
        message_agent=message_agent,
        db=db,
    )

    # Should not raise
    await channel.validate_connectivity()

    await channel.close()


@pytest.mark.asyncio
async def test_validate_connectivity_dns_failure(test_db, mock_ollama):
    """Test that validate_connectivity raises ConnectionError on DNS failure."""
    from penny.agents import MessageAgent
    from penny.config import Config
    from penny.constants import SYSTEM_PROMPT

    config = Config(
        channel_type="signal",
        signal_number="+15551234567",
        signal_api_url="http://nonexistent-hostname-that-will-never-resolve.invalid:8080",
        discord_bot_token=None,
        discord_channel_id=None,
        ollama_api_url="http://localhost:11434",
        ollama_foreground_model="test-model",
        ollama_background_model="test-model",
        perplexity_api_key=None,
        log_level="DEBUG",
        db_path=test_db,
    )

    db = Database(config.db_path)
    db.create_tables()

    message_agent = MessageAgent(
        system_prompt=SYSTEM_PROMPT,
        model=config.ollama_foreground_model,
        ollama_api_url=config.ollama_api_url,
        tools=[],
        db=db,
        max_steps=1,
    )

    channel = SignalChannel(
        api_url=config.signal_api_url,
        phone_number=config.signal_number or "+15551234567",
        message_agent=message_agent,
        db=db,
    )

    with pytest.raises(ConnectionError) as exc_info:
        await channel.validate_connectivity()

    error_message = str(exc_info.value)
    assert "Cannot resolve Signal API hostname" in error_message
    assert "nonexistent-hostname-that-will-never-resolve.invalid" in error_message
    assert "SIGNAL_API_URL" in error_message

    await channel.close()


@pytest.mark.asyncio
async def test_validate_connectivity_connection_refused(test_db, mock_ollama):
    """Test that validate_connectivity raises ConnectionError when server is unreachable."""
    from penny.agents import MessageAgent
    from penny.config import Config
    from penny.constants import SYSTEM_PROMPT

    # Use localhost on a port that's not listening
    config = Config(
        channel_type="signal",
        signal_number="+15551234567",
        signal_api_url="http://localhost:19999",  # Unlikely to be in use
        discord_bot_token=None,
        discord_channel_id=None,
        ollama_api_url="http://localhost:11434",
        ollama_foreground_model="test-model",
        ollama_background_model="test-model",
        perplexity_api_key=None,
        log_level="DEBUG",
        db_path=test_db,
    )

    db = Database(config.db_path)
    db.create_tables()

    message_agent = MessageAgent(
        system_prompt=SYSTEM_PROMPT,
        model=config.ollama_foreground_model,
        ollama_api_url=config.ollama_api_url,
        tools=[],
        db=db,
        max_steps=1,
    )

    channel = SignalChannel(
        api_url=config.signal_api_url,
        phone_number=config.signal_number or "+15551234567",
        message_agent=message_agent,
        db=db,
    )

    with pytest.raises(ConnectionError) as exc_info:
        await channel.validate_connectivity()

    error_message = str(exc_info.value)
    assert "Cannot connect to Signal API" in error_message
    assert "http://localhost:19999" in error_message

    await channel.close()

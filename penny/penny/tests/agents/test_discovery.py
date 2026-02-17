"""Integration tests for the DiscoveryAgent."""

import pytest

from penny.tests.conftest import TEST_SENDER


@pytest.mark.asyncio
async def test_discovery_excludes_dislikes(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test that DiscoveryAgent excludes disliked topics from searches:
    1. Build a user profile
    2. Add a dislike preference
    3. Manually trigger DiscoveryAgent
    4. Verify the prompt includes dislike exclusions
    """
    config = make_config()
    mock_ollama.set_default_flow(
        search_query="jazz music",
        final_response="check out John Coltrane! 🎷",
    )

    async with running_penny(config) as penny:
        # Send a message to establish an interest
        await signal_server.push_message(sender=TEST_SENDER, content="i love jazz music")
        await signal_server.wait_for_message(timeout=10.0)

        # Add a like preference (required for discovery to run)
        penny.db.add_preference(TEST_SENDER, "jazz music", "like")

        # Add a dislike preference
        penny.db.add_preference(TEST_SENDER, "Kenny G", "dislike")

        # Verify the preference was added
        dislikes = penny.db.get_preferences(TEST_SENDER, "dislike")
        assert len(dislikes) == 1
        assert dislikes[0].topic == "Kenny G"

        # Clear Ollama requests
        mock_ollama.requests.clear()

        # Manually trigger discovery
        from penny.agents import DiscoveryAgent
        from penny.prompts import Prompt

        discovery_agent = DiscoveryAgent(
            system_prompt=Prompt.SEARCH_PROMPT,
            model=penny.message_agent.model,
            ollama_api_url=config.ollama_api_url,
            tools=penny.message_agent.tools,
            db=penny.db,
        )
        discovery_agent.set_channel(penny.channel)

        # Execute the discovery agent
        result = await discovery_agent.execute()

        # Discovery should have run since we have a profile
        assert result is True, "Discovery should have executed"
        assert len(mock_ollama.requests) >= 1, "Discovery should have called Ollama"

        # Check that the prompt includes dislike exclusions
        discovery_request = mock_ollama.requests[-1]
        messages = discovery_request.get("messages", [])
        system_messages = [m for m in messages if m.get("role") == "system"]

        # Should have a system message mentioning dislikes
        exclusion_found = False
        for msg in system_messages:
            content = msg.get("content", "")
            if "don't include" in content.lower() or "avoid" in content.lower():
                assert "kenny g" in content.lower(), "Dislike should be in exclusions"
                exclusion_found = True
                break

        assert exclusion_found, "Discovery prompt should include dislike exclusions"


@pytest.mark.asyncio
async def test_discovery_no_channel(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Test that DiscoveryAgent returns False when no channel is set."""
    config = make_config()

    async with running_penny(config) as penny:
        from penny.agents import DiscoveryAgent
        from penny.prompts import Prompt

        discovery_agent = DiscoveryAgent(
            system_prompt=Prompt.SEARCH_PROMPT,
            model=config.ollama_foreground_model,
            ollama_api_url=config.ollama_api_url,
            tools=[],
            db=penny.db,
        )
        # Don't set channel

        result = await discovery_agent.execute()
        assert result is False, "Should return False when no channel set"


@pytest.mark.asyncio
async def test_discovery_no_users(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Test that DiscoveryAgent returns False when no users have sent messages."""
    config = make_config()

    async with running_penny(config) as penny:
        from penny.agents import DiscoveryAgent
        from penny.prompts import Prompt

        discovery_agent = DiscoveryAgent(
            system_prompt=Prompt.SEARCH_PROMPT,
            model=config.ollama_foreground_model,
            ollama_api_url=config.ollama_api_url,
            tools=[],
            db=penny.db,
        )
        discovery_agent.set_channel(penny.channel)

        # No messages sent, so no users
        result = await discovery_agent.execute()
        assert result is False, "Should return False when no users"


@pytest.mark.asyncio
async def test_discovery_no_likes(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Test that DiscoveryAgent returns False when user has no likes."""
    config = make_config()
    mock_ollama.set_default_flow(
        search_query="test",
        final_response="test response 🌟",
    )

    async with running_penny(config) as penny:
        # Send a message to create a user (sender)
        await signal_server.push_message(sender=TEST_SENDER, content="hello")
        await signal_server.wait_for_message(timeout=10.0)

        # Clear Ollama requests
        mock_ollama.requests.clear()

        from penny.agents import DiscoveryAgent
        from penny.prompts import Prompt

        discovery_agent = DiscoveryAgent(
            system_prompt=Prompt.SEARCH_PROMPT,
            model=config.ollama_foreground_model,
            ollama_api_url=config.ollama_api_url,
            tools=[],
            db=penny.db,
        )
        discovery_agent.set_channel(penny.channel)

        # User exists but has no likes
        result = await discovery_agent.execute()
        assert result is False, "Should return False when user has no likes"

        # Verify no Ollama calls were made
        assert len(mock_ollama.requests) == 0, "Should not call Ollama when no likes"

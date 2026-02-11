"""Integration tests for the FollowupAgent."""

import pytest
from sqlmodel import select

from penny.database.models import MessageLog
from penny.tests.conftest import TEST_SENDER, wait_until


@pytest.mark.asyncio
async def test_followup_background_task(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
    setup_ollama_flow,
):
    """
    Test the followup background task:
    1. Send a message and get a response (creates conversation leaf)
    2. Wait for idle time + random delay to pass
    3. Verify FollowupAgent sends a spontaneous follow-up message
    """
    config = make_config(
        idle_seconds=0.3,
        followup_min_seconds=0.1,
        followup_max_seconds=0.2,
    )
    setup_ollama_flow(
        search_query="best hiking trails nearby",
        message_response="found some great trails for you! 🥾",
        background_response="oh btw, i found more cool trails you might like! 🌲",
    )

    async with running_penny(config) as penny:
        await signal_server.push_message(
            sender=TEST_SENDER, content="where can i go hiking this weekend?"
        )
        response = await signal_server.wait_for_message(timeout=10.0)
        assert "trails" in response["message"].lower()

        # Verify conversation leaf was created
        leaves = penny.db.get_conversation_leaves()
        assert len(leaves) >= 1, "Should have at least one conversation leaf"

        # Record the message count after first response
        first_response_count = len(signal_server.outgoing_messages)

        # Wait for followup task to send a second message
        await wait_until(lambda: len(signal_server.outgoing_messages) > first_response_count)

        followup = signal_server.outgoing_messages[-1]
        assert followup["recipients"] == [TEST_SENDER]
        assert "trail" in followup["message"].lower()

        # Verify the followup was logged to database with parent link
        with penny.db.get_session() as session:
            outgoing = list(
                session.exec(select(MessageLog).where(MessageLog.direction == "outgoing")).all()
            )
        assert len(outgoing) >= 2, "Should have original response and followup"

        followup_msg = outgoing[-1]
        assert followup_msg.parent_id is not None, "Followup should be linked to thread"


@pytest.mark.asyncio
async def test_followup_excludes_dislikes(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test that FollowupAgent excludes disliked topics from searches:
    1. Create a conversation with a leaf
    2. Add a dislike preference for the user
    3. Manually trigger FollowupAgent
    4. Verify the prompt includes dislike exclusions
    """
    config = make_config()
    mock_ollama.set_default_flow(
        search_query="coffee shops",
        final_response="check out Blue Bottle Coffee! ☕",
    )

    async with running_penny(config) as penny:
        # Send initial message and get response
        await signal_server.push_message(sender=TEST_SENDER, content="recommend a coffee shop")
        await signal_server.wait_for_message(timeout=10.0)

        # Add a dislike preference
        penny.db.add_preference(TEST_SENDER, "Starbucks", "dislike")

        # Verify the preference was added
        dislikes = penny.db.get_preferences(TEST_SENDER, "dislike")
        assert len(dislikes) == 1
        assert dislikes[0].topic == "Starbucks"

        # Clear Ollama requests
        mock_ollama.requests.clear()

        # Manually trigger followup
        from penny.agent.agents import FollowupAgent
        from penny.constants import SYSTEM_PROMPT

        followup_agent = FollowupAgent(
            system_prompt=SYSTEM_PROMPT,
            model=penny.message_agent.model,
            ollama_api_url=config.ollama_api_url,
            tools=penny.message_agent.tools,
            db=penny.db,
        )
        followup_agent.set_channel(penny.channel)

        # Execute the followup agent
        await followup_agent.execute()

        # If there's a conversation leaf, followup should have run
        leaves = penny.db.get_conversation_leaves()
        if leaves:
            # Check that Ollama was called
            assert len(mock_ollama.requests) >= 1, "Followup should have called Ollama"

            # Check that the prompt includes dislike exclusions
            followup_request = mock_ollama.requests[-1]
            messages = followup_request.get("messages", [])
            system_messages = [m for m in messages if m.get("role") == "system"]

            # Should have a system message mentioning dislikes
            exclusion_found = False
            for msg in system_messages:
                content = msg.get("content", "")
                if "don't include" in content.lower() or "avoid" in content.lower():
                    assert "starbucks" in content.lower(), "Dislike should be in exclusions"
                    exclusion_found = True
                    break

            assert exclusion_found, "Followup prompt should include dislike exclusions"


@pytest.mark.asyncio
async def test_followup_no_channel(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Test that FollowupAgent returns False when no channel is set."""
    config = make_config()

    async with running_penny(config) as penny:
        from penny.agent.agents import FollowupAgent
        from penny.constants import SYSTEM_PROMPT

        followup_agent = FollowupAgent(
            system_prompt=SYSTEM_PROMPT,
            model=config.ollama_foreground_model,
            ollama_api_url=config.ollama_api_url,
            tools=[],
            db=penny.db,
        )
        # Don't set channel

        result = await followup_agent.execute()
        assert result is False, "Should return False when no channel set"


@pytest.mark.asyncio
async def test_followup_no_leaves(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Test that FollowupAgent returns False when there are no conversation leaves."""
    config = make_config()

    async with running_penny(config) as penny:
        from penny.agent.agents import FollowupAgent
        from penny.constants import SYSTEM_PROMPT

        followup_agent = FollowupAgent(
            system_prompt=SYSTEM_PROMPT,
            model=config.ollama_foreground_model,
            ollama_api_url=config.ollama_api_url,
            tools=[],
            db=penny.db,
        )
        followup_agent.set_channel(penny.channel)

        # No messages sent, so no conversation leaves
        result = await followup_agent.execute()
        assert result is False, "Should return False when no conversation leaves"

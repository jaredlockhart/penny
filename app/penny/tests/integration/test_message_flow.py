"""Integration tests for the basic message flow."""

import asyncio

import pytest
from sqlmodel import select

from penny.database.models import MessageLog
from penny.tests.conftest import TEST_SENDER


@pytest.mark.asyncio
async def test_basic_message_flow(
    signal_server, mock_ollama, test_config, _mock_search, running_penny
):
    """
    Test the complete message flow:
    1. User sends a message via Signal
    2. Penny receives and processes it
    3. Ollama returns a tool call (search)
    4. Search tool executes (mocked)
    5. Ollama returns final response
    6. Penny sends reply via Signal
    """
    # Configure Ollama to return search tool call, then final response
    mock_ollama.set_default_flow(
        search_query="test search query",
        final_response="here's what i found about your question! 🌟",
    )

    async with running_penny(test_config) as penny:
        # Verify we have a WebSocket connection
        assert len(signal_server._websockets) == 1, "Penny should have connected to WebSocket"

        # Send incoming message
        await signal_server.push_message(
            sender=TEST_SENDER,
            content="what's the weather like today?",
        )

        # Wait for response
        response = await signal_server.wait_for_message(timeout=10.0)

        # Verify the response
        assert response["recipients"] == [TEST_SENDER]
        assert "here's what i found" in response["message"].lower()

        # Verify Ollama was called twice (tool call + final response)
        assert len(mock_ollama.requests) == 2, "Expected 2 Ollama calls (tool + final)"

        # First request should have user message
        first_request = mock_ollama.requests[0]
        messages = first_request.get("messages", [])
        user_messages = [m for m in messages if m.get("role") == "user"]
        assert any("weather" in m.get("content", "").lower() for m in user_messages)

        # Second request should include tool result
        second_request = mock_ollama.requests[1]
        messages = second_request.get("messages", [])
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_messages) >= 1, "Second request should include tool result"

        # Verify typing indicators were sent
        assert len(signal_server.typing_events) >= 1, "Should have sent typing indicator"

        # Verify messages were logged to database
        incoming_messages = penny.db.get_user_messages(TEST_SENDER)
        assert len(incoming_messages) >= 1, "Incoming message should be logged"

        with penny.db.get_session() as session:
            outgoing = list(
                session.exec(select(MessageLog).where(MessageLog.direction == "outgoing")).all()
            )
        assert len(outgoing) >= 1, "Outgoing message should be logged"


@pytest.mark.asyncio
async def test_message_without_tool_call(
    signal_server, mock_ollama, test_config, _mock_search, running_penny
):
    """Test handling a message where Ollama doesn't call a tool."""

    # Configure Ollama to return direct response (no tool call)
    def direct_response(request, count):
        return mock_ollama._make_text_response(request, "just a simple response! 🌟")

    mock_ollama.set_response_handler(direct_response)

    async with running_penny(test_config):
        await signal_server.push_message(
            sender=TEST_SENDER,
            content="hello penny",
        )

        response = await signal_server.wait_for_message(timeout=10.0)

        assert response["recipients"] == [TEST_SENDER]
        assert "simple response" in response["message"].lower()

        # Only one Ollama call (no tool)
        assert len(mock_ollama.requests) == 1


@pytest.mark.asyncio
async def test_summarize_background_task(
    signal_server, mock_ollama, _mock_search, make_config, running_penny, setup_ollama_flow
):
    """
    Test the summarize background task:
    1. Send a message and get a response (creates a thread)
    2. Wait for idle time to pass
    3. Verify SummarizeAgent generates and stores a summary
    """
    config = make_config(idle_seconds=0.5)
    setup_ollama_flow(
        search_query="weather forecast today",
        message_response="here's the weather info! 🌤️",
        background_response="user asked about weather, assistant provided forecast",
    )

    async with running_penny(config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="what's the weather like?")
        response = await signal_server.wait_for_message(timeout=10.0)
        assert "weather" in response["message"].lower()

        # Get the outgoing message id
        with penny.db.get_session() as session:
            outgoing = session.exec(
                select(MessageLog).where(MessageLog.direction == "outgoing")
            ).first()
            assert outgoing is not None
            message_id = outgoing.id
            assert outgoing.parent_id is not None
            assert outgoing.parent_summary is None

        # Wait for summarize task to trigger (idle time + scheduler tick)
        await asyncio.sleep(2.0)

        # Verify summary was generated
        with penny.db.get_session() as session:
            outgoing = session.get(MessageLog, message_id)
            assert outgoing is not None
            assert outgoing.parent_summary is not None
            assert len(outgoing.parent_summary) > 0
            assert "weather" in outgoing.parent_summary.lower()

        assert len(mock_ollama.requests) >= 3, "Expected at least 3 Ollama calls"


@pytest.mark.asyncio
async def test_profile_background_task(
    signal_server, mock_ollama, _mock_search, make_config, running_penny, setup_ollama_flow
):
    """
    Test the profile background task:
    1. Send a message and get a response
    2. Wait for idle time to pass
    3. Verify ProfileAgent generates and stores a user profile
    """
    config = make_config(idle_seconds=0.5)
    setup_ollama_flow(
        search_query="fun facts about cats",
        message_response="cats are amazing! 🐱",
        background_response="curious user interested in animals, especially cats.",
    )

    async with running_penny(config) as penny:
        await signal_server.push_message(
            sender=TEST_SENDER, content="tell me something cool about cats!"
        )
        response = await signal_server.wait_for_message(timeout=10.0)
        assert "cats" in response["message"].lower()

        # Verify no profile exists yet
        profile = penny.db.get_user_profile(TEST_SENDER)
        assert profile is None, "Profile should not exist yet"

        # Wait for profile task to trigger (idle time + scheduler tick)
        await asyncio.sleep(2.0)

        # Verify profile was generated
        profile = penny.db.get_user_profile(TEST_SENDER)
        assert profile is not None, "Profile should have been generated"
        assert len(profile.profile_text) > 0
        assert "cat" in profile.profile_text.lower() or "animal" in profile.profile_text.lower()

        assert len(mock_ollama.requests) >= 3, "Expected at least 3 Ollama calls"


@pytest.mark.asyncio
async def test_followup_background_task(
    signal_server, mock_ollama, _mock_search, make_config, running_penny, setup_ollama_flow
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

        # Wait for followup task to trigger
        await asyncio.sleep(5.0)

        # Check if followup was sent
        assert len(signal_server.outgoing_messages) > first_response_count, (
            f"Followup message should have been sent. "
            f"Messages: {len(signal_server.outgoing_messages)}, "
            f"Expected > {first_response_count}"
        )

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
async def test_signal_reaction_message(
    signal_server, mock_ollama, _mock_search, make_config, running_penny, setup_ollama_flow
):
    """
    Test Signal reaction handling:
    1. Send a message and get a response
    2. React to the response with an emoji
    3. Verify reaction is logged as a message with is_reaction=True
    4. Verify thread is kept alive for followup
    """
    config = make_config(idle_seconds=0.5)
    setup_ollama_flow(
        search_query="test query",
        message_response="here's a cool fact! 🌟",
        background_response="glad you liked that, here's more! 🎉",
    )

    async with running_penny(config) as penny:
        # Send initial message
        await signal_server.push_message(sender=TEST_SENDER, content="tell me something cool")
        response = await signal_server.wait_for_message(timeout=10.0)
        assert "cool fact" in response["message"].lower()

        # Get the outgoing message's signal timestamp
        with penny.db.get_session() as session:
            outgoing = session.exec(
                select(MessageLog).where(MessageLog.direction == "outgoing")
            ).first()
            assert outgoing is not None
            assert outgoing.external_id is not None
            message_id = outgoing.id
            external_id = outgoing.external_id

        # Send a reaction to Penny's response
        await signal_server.push_reaction(
            sender=TEST_SENDER,
            emoji="👍",
            target_timestamp=int(external_id),
        )

        # Wait a bit for processing
        await asyncio.sleep(0.5)

        # Verify reaction was logged
        with penny.db.get_session() as session:
            reactions = list(
                session.exec(
                    select(MessageLog).where(
                        MessageLog.is_reaction == True,  # noqa: E712
                        MessageLog.sender == TEST_SENDER,
                    )
                ).all()
            )
        assert len(reactions) == 1, "Reaction should be logged"
        reaction = reactions[0]
        assert reaction.content == "👍"
        assert reaction.parent_id == message_id
        assert reaction.is_reaction is True

        # Verify thread is no longer a leaf (has reaction as child)
        leaves = penny.db.get_conversation_leaves()
        # The original outgoing message should not be in leaves anymore
        # because it now has a reaction child
        leaf_ids = [leaf.id for leaf in leaves]
        assert message_id not in leaf_ids, "Reacted message should not be a conversation leaf"

        # Verify no response was sent to the reaction
        # (only the initial response should exist)
        assert len(signal_server.outgoing_messages) == 1

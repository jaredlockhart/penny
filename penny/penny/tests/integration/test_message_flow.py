"""Integration tests for the basic message flow."""

import asyncio

import pytest
from sqlmodel import select

from penny.database.models import MessageLog, SearchLog
from penny.tests.conftest import TEST_SENDER


@pytest.mark.asyncio
async def test_basic_message_flow(
    signal_server, mock_ollama, test_config, _mock_search, test_user_info, running_penny
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
    signal_server, mock_ollama, test_config, _mock_search, test_user_info, running_penny
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
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
    setup_ollama_flow,
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
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
    setup_ollama_flow,
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

    async with running_penny(config):
        await signal_server.push_message(
            sender=TEST_SENDER, content="tell me something cool about cats!"
        )
        response = await signal_server.wait_for_message(timeout=10.0)
        assert "cats" in response["message"].lower()

        # Note: ProfileAgent now maintains preferences via reactions,
        # not by generating topics from message history.
        # The old UserTopics infrastructure has been removed.

        assert len(mock_ollama.requests) >= 2, "Expected at least 2 Ollama calls"


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
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
    setup_ollama_flow,
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


@pytest.mark.asyncio
async def test_signal_reaction_raw_format(
    signal_server, mock_ollama, _mock_search, make_config, running_penny
):
    """
    Test Signal reaction handling with the raw format that Signal actually sends.

    This tests the bug fix for issue #34 where Signal sends:
    - message: None (not an empty string)
    - emoji: "👍" (plain string, not {"value": "👍"} object)
    """
    config = make_config()
    mock_ollama.set_default_flow(
        search_query="test query",
        final_response="test response 🌟",
    )

    async with running_penny(config) as penny:
        # Send initial message
        await signal_server.push_message(sender=TEST_SENDER, content="test message")
        await signal_server.wait_for_message(timeout=10.0)

        # Get the outgoing message's signal timestamp
        with penny.db.get_session() as session:
            outgoing = session.exec(
                select(MessageLog).where(MessageLog.direction == "outgoing")
            ).first()
            assert outgoing is not None
            message_id = outgoing.id
            external_id = outgoing.external_id

        # Send a reaction using the raw format that Signal actually sends
        # (not the mock format with {"value": emoji})
        import json
        import time

        ts = int(time.time() * 1000)
        raw_envelope = {
            "envelope": {
                "source": TEST_SENDER,
                "sourceNumber": TEST_SENDER,
                "sourceUuid": "test-uuid-123",
                "sourceName": "Test User",
                "sourceDevice": 1,
                "timestamp": ts,
                "serverReceivedTimestamp": ts,
                "serverDeliveredTimestamp": ts,
                "dataMessage": {
                    "timestamp": ts,
                    "message": None,  # KEY: None, not empty string
                    "reaction": {
                        "emoji": "👍",  # KEY: Plain string, not {"value": "👍"}
                        "targetAuthor": config.signal_number,
                        "targetAuthorNumber": config.signal_number,
                        "targetSentTimestamp": int(external_id),
                        "isRemove": False,
                    },
                },
            },
            "account": config.signal_number,
        }

        # Push the raw envelope to all connected websockets
        for ws in signal_server._websockets:
            if not ws.closed:
                await ws.send_str(json.dumps(raw_envelope))

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


@pytest.mark.asyncio
async def test_startup_announcement(
    signal_server, mock_ollama, test_config, _mock_search, running_penny, monkeypatch
):
    """
    Test that Penny sends a startup announcement (wave emoji) when starting up.

    The announcement should:
    - Be sent to all known recipients (users who have sent messages)
    - Not be logged to the database
    - Be sent after initialization is complete
    """
    # Set up initial message history so we have a known recipient
    config = test_config
    mock_ollama.set_default_flow(
        search_query="test query",
        final_response="test response 🌟",
    )

    # First, send a message to populate the database with a sender
    async with running_penny(config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="initial message")
        await signal_server.wait_for_message(timeout=10.0)

        # Verify the sender is in the database
        senders = penny.db.get_all_senders()
        assert TEST_SENDER in senders

        # Create user profile so they get startup announcements
        penny.db.save_user_info(
            sender=TEST_SENDER,
            name="Test User",
            location="Seattle, WA",
            timezone="America/Los_Angeles",
            date_of_birth="1990-01-01",
        )

    # Clear the outgoing messages from the first run
    signal_server.outgoing_messages.clear()

    # Set commit message in environment variable (fallback case - no commit message)
    monkeypatch.setenv("GIT_COMMIT_MESSAGE", "unknown")

    # Configure mock_ollama for restart message generation (not used with fallback)
    mock_ollama.set_default_flow(
        final_response="I just restarted!",
    )

    # Now start Penny again - it should send startup announcement
    async with running_penny(config) as penny:
        # Wait a bit for startup announcement to be sent
        await asyncio.sleep(0.5)

        # Verify startup announcement was sent
        assert len(signal_server.outgoing_messages) == 1
        startup_msg = signal_server.outgoing_messages[0]
        # Should start with wave and include restart message
        assert startup_msg["message"].startswith("👋")
        assert TEST_SENDER in startup_msg["recipients"]

        # Verify the startup announcement was NOT logged to database
        with penny.db.get_session() as session:
            # Count all outgoing messages (should be just the one from first run)
            outgoing = list(
                session.exec(select(MessageLog).where(MessageLog.direction == "outgoing")).all()
            )
        # Only the response from the first run should be logged, not the startup announcement
        assert len(outgoing) == 1
        assert "👋" not in outgoing[0].content


@pytest.mark.asyncio
async def test_profile_context_excludes_dob_and_redacts_name_from_search(
    signal_server, mock_ollama, test_config, _mock_search, test_user_info, running_penny
):
    """
    Test privacy protections for user profile data:
    1. DOB is not included in the profile context sent to Ollama
    2. User name is redacted from search queries before reaching Perplexity
    """
    # The test user is "Test User" from conftest — have the model generate
    # a search query that includes the user's name
    mock_ollama.set_default_flow(
        search_query="Test User Toronto weather forecast",
        final_response="here's the weather! 🌤️",
    )

    async with running_penny(test_config) as penny:
        await signal_server.push_message(
            sender=TEST_SENDER,
            content="what's the weather?",
        )
        await signal_server.wait_for_message(timeout=10.0)

        # Verify DOB is NOT in the Ollama prompt messages
        first_request = mock_ollama.requests[0]
        messages = first_request.get("messages", [])
        system_messages = [m for m in messages if m.get("role") == "system"]
        all_system_text = " ".join(m.get("content", "") for m in system_messages)
        assert "1990-01-01" not in all_system_text, "DOB should not be in profile context"
        assert "born" not in all_system_text.lower(), "DOB field should not be in profile context"

        # Verify profile context IS present (name + location)
        assert "Test User" in all_system_text, "Name should be in profile context"
        assert "Seattle" in all_system_text, "Location should be in profile context"

        # Verify user name was redacted from the search query logged to DB
        with penny.db.get_session() as session:
            search_logs = list(session.exec(select(SearchLog)).all())
        assert len(search_logs) >= 1, "Search should have been logged"
        logged_query = search_logs[0].query
        assert "Test User" not in logged_query, "User name should be redacted from search query"
        assert "Toronto weather forecast" in logged_query, "Rest of query should be preserved"


@pytest.mark.asyncio
async def test_name_not_redacted_when_user_says_it(
    signal_server, mock_ollama, test_config, _mock_search, test_user_info, running_penny
):
    """
    When the user's message contains their own name (e.g. searching for
    a celebrity with the same name), the name should NOT be redacted from
    the search query.
    """
    # Model echoes the name back in the search query
    mock_ollama.set_default_flow(
        search_query="Test User celebrity gossip",
        final_response="here's what i found! 🌟",
    )

    async with running_penny(test_config) as penny:
        # User explicitly typed their own name in the message
        await signal_server.push_message(
            sender=TEST_SENDER,
            content="search for Test User celebrity gossip",
        )
        await signal_server.wait_for_message(timeout=10.0)

        # Name should be preserved in the search query since the user said it
        with penny.db.get_session() as session:
            search_logs = list(session.exec(select(SearchLog)).all())
        assert len(search_logs) >= 1, "Search should have been logged"
        logged_query = search_logs[0].query
        assert "Test User" in logged_query, "Name should NOT be redacted when user said it"


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
        from penny.agent.agents import DiscoveryAgent
        from penny.constants import SYSTEM_PROMPT

        discovery_agent = DiscoveryAgent(
            system_prompt=SYSTEM_PROMPT,
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
async def test_profile_reaction_processing_idempotency(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test that ProfileAgent only processes each reaction once:
    1. Insert a reaction directly into the database
    2. Mark it as processed
    3. Verify get_user_reactions() returns empty list (no unprocessed reactions)
    4. Add another reaction without marking it processed
    5. Verify get_user_reactions() returns only the new unprocessed reaction
    """
    config = make_config()

    async with running_penny(config) as penny:
        # Insert a processed reaction directly
        processed_reaction_id = penny.db.log_message(
            direction="incoming",
            sender=TEST_SENDER,
            content="❤️",
            parent_id=None,
            is_reaction=True,
        )
        assert processed_reaction_id is not None
        penny.db.mark_reaction_processed(processed_reaction_id)

        # Verify get_user_reactions returns empty (processed reaction is excluded)
        reactions = penny.db.get_user_reactions(TEST_SENDER)
        assert len(reactions) == 0

        # Insert an unprocessed reaction
        unprocessed_reaction_id = penny.db.log_message(
            direction="incoming",
            sender=TEST_SENDER,
            content="👍",
            parent_id=None,
            is_reaction=True,
        )
        assert unprocessed_reaction_id is not None

        # Verify get_user_reactions returns only the unprocessed reaction
        reactions = penny.db.get_user_reactions(TEST_SENDER)
        assert len(reactions) == 1
        assert reactions[0].id == unprocessed_reaction_id
        assert reactions[0].content == "👍"
        assert reactions[0].processed is False

        # Mark the second reaction as processed
        penny.db.mark_reaction_processed(unprocessed_reaction_id)

        # Verify get_user_reactions now returns empty again
        reactions = penny.db.get_user_reactions(TEST_SENDER)
        assert len(reactions) == 0


@pytest.mark.asyncio
async def test_profile_agent_processes_reaction_into_preference(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
    setup_ollama_flow,
):
    """
    Test that ProfileAgent processes a reaction into a preference:
    1. Send a message and get a response
    2. React to the response with a like emoji
    3. Run ProfileAgent.execute() directly
    4. Verify the preference was added to the database
    """
    config = make_config(idle_seconds=0.5)
    setup_ollama_flow(
        search_query="fun facts about cats",
        message_response="cats are amazing! they can jump 6 times their length 🐱",
        background_response='{"topic": "cats"}',
    )

    async with running_penny(config) as penny:
        # Send initial message and get response
        await signal_server.push_message(
            sender=TEST_SENDER, content="tell me something cool about cats!"
        )
        response = await signal_server.wait_for_message(timeout=10.0)
        assert "cats" in response["message"].lower()

        # Get the outgoing message's external_id for reaction targeting
        with penny.db.get_session() as session:
            outgoing = session.exec(
                select(MessageLog).where(MessageLog.direction == "outgoing")
            ).first()
            assert outgoing is not None
            assert outgoing.external_id is not None
            external_id = outgoing.external_id

        # Send a like reaction to Penny's response
        await signal_server.push_reaction(
            sender=TEST_SENDER,
            emoji="❤️",
            target_timestamp=int(external_id),
        )
        await asyncio.sleep(0.5)

        # Verify reaction was logged with parent_id set
        with penny.db.get_session() as session:
            reactions = list(
                session.exec(
                    select(MessageLog).where(
                        MessageLog.is_reaction == True,  # noqa: E712
                        MessageLog.sender == TEST_SENDER,
                    )
                ).all()
            )
        assert len(reactions) == 1
        reaction = reactions[0]
        assert reaction.parent_id is not None, "Reaction should have parent_id set"

        # Run ProfileAgent directly
        work_done = await penny.profile_agent.execute()
        assert work_done, "ProfileAgent should have processed the reaction"

        # Verify preference was added
        from penny.constants import PreferenceType

        prefs = penny.db.get_preferences(TEST_SENDER, PreferenceType.LIKE)
        assert len(prefs) == 1, f"Expected 1 like preference, got {len(prefs)}"
        assert prefs[0].topic == "cats"

        # Verify reaction was marked as processed
        reactions = penny.db.get_user_reactions(TEST_SENDER)
        assert len(reactions) == 0, "Reaction should be marked as processed"

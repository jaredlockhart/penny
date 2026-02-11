"""Integration tests for the PreferenceAgent."""

import pytest
from sqlmodel import select

from penny.constants import PREFERENCE_BATCH_LIMIT, PreferenceType
from penny.database.models import MessageLog
from penny.tests.conftest import TEST_SENDER, wait_until


@pytest.mark.asyncio
async def test_preference_background_task(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
    setup_ollama_flow,
):
    """
    Test the preference background task:
    1. Send a message and get a response
    2. Wait for idle time to pass
    3. Verify PreferenceAgent runs and processes messages
    """
    config = make_config(idle_seconds=0.5)
    setup_ollama_flow(
        search_query="fun facts about cats",
        message_response="cats are amazing! 🐱",
        background_response='{"topics": []}',
    )

    async with running_penny(config):
        await signal_server.push_message(
            sender=TEST_SENDER, content="tell me something cool about cats!"
        )
        response = await signal_server.wait_for_message(timeout=10.0)
        assert "cats" in response["message"].lower()

        assert len(mock_ollama.requests) >= 2, "Expected at least 2 Ollama calls"


@pytest.mark.asyncio
async def test_preference_reaction_processing_idempotency(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test that PreferenceAgent only processes each reaction once:
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
        reactions = penny.db.get_user_reactions(TEST_SENDER, limit=PREFERENCE_BATCH_LIMIT)
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
        reactions = penny.db.get_user_reactions(TEST_SENDER, limit=PREFERENCE_BATCH_LIMIT)
        assert len(reactions) == 1
        assert reactions[0].id == unprocessed_reaction_id
        assert reactions[0].content == "👍"
        assert reactions[0].processed is False

        # Mark the second reaction as processed
        penny.db.mark_reaction_processed(unprocessed_reaction_id)

        # Verify get_user_reactions now returns empty again
        reactions = penny.db.get_user_reactions(TEST_SENDER, limit=PREFERENCE_BATCH_LIMIT)
        assert len(reactions) == 0


@pytest.mark.asyncio
async def test_preference_agent_processes_reaction_into_preference(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test that PreferenceAgent processes a reaction into a preference:
    1. Send a message and get a response
    2. React to the response with a like emoji
    3. Run PreferenceAgent.execute() directly
    4. Verify the preference was added to the database
    """
    config = make_config()

    request_count = [0]

    def handler(request: dict, count: int) -> dict:
        request_count[0] += 1
        if request_count[0] == 1:
            return mock_ollama._make_tool_call_response(
                request, "search", {"query": "fun facts about cats"}
            )
        elif request_count[0] == 2:
            return mock_ollama._make_text_response(
                request, "cats are amazing! they can jump 6 times their length 🐱"
            )
        else:
            # Two-pass preference extraction — return cats for likes pass
            return mock_ollama._make_text_response(request, '{"topics": ["cats"]}')

    mock_ollama.set_response_handler(handler)

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

        # Wait for reaction to be logged in the DB
        def reaction_logged():
            with penny.db.get_session() as session:
                reactions = list(
                    session.exec(
                        select(MessageLog).where(
                            MessageLog.is_reaction == True,  # noqa: E712
                            MessageLog.sender == TEST_SENDER,
                        )
                    ).all()
                )
                return len(reactions) == 1

        await wait_until(reaction_logged)

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

        # Run PreferenceAgent directly
        work_done = await penny.preference_agent.execute()
        assert work_done, "PreferenceAgent should have processed the reaction"

        # Verify preference was added
        prefs = penny.db.get_preferences(TEST_SENDER, PreferenceType.LIKE)
        assert len(prefs) >= 1, f"Expected at least 1 like preference, got {len(prefs)}"
        assert any(p.topic == "cats" for p in prefs)

        # Verify reaction was marked as processed
        reactions = penny.db.get_user_reactions(TEST_SENDER, limit=PREFERENCE_BATCH_LIMIT)
        assert len(reactions) == 0, "Reaction should be marked as processed"


@pytest.mark.asyncio
async def test_preference_no_channel(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Test that PreferenceAgent returns False when no channel is set."""
    config = make_config()

    async with running_penny(config) as penny:
        from penny.agents import PreferenceAgent
        from penny.constants import SYSTEM_PROMPT

        preference_agent = PreferenceAgent(
            system_prompt=SYSTEM_PROMPT,
            model=config.ollama_foreground_model,
            ollama_api_url=config.ollama_api_url,
            tools=[],
            db=penny.db,
        )
        # Don't set channel

        result = await preference_agent.execute()
        assert result is False, "Should return False when no channel set"


@pytest.mark.asyncio
async def test_preference_no_senders(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Test that PreferenceAgent returns False when no senders exist."""
    config = make_config()

    async with running_penny(config) as penny:
        # The penny.preference_agent already has channel set,
        # but no messages have been sent, so no senders
        result = await penny.preference_agent.execute()
        assert result is False, "Should return False when no senders"


@pytest.mark.asyncio
async def test_preference_unknown_emoji(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
    setup_ollama_flow,
):
    """
    Test that PreferenceAgent skips unknown reaction emojis but still marks them processed.
    """
    config = make_config()

    # The agent will still run the likes/dislikes passes for user messages
    setup_ollama_flow(
        search_query="test query",
        message_response="interesting fact! 🌟",
        background_response='{"topics": []}',
    )

    async with running_penny(config) as penny:
        # Send message to create a sender
        await signal_server.push_message(sender=TEST_SENDER, content="tell me a fact")
        await signal_server.wait_for_message(timeout=10.0)

        # Mark user messages as processed so only the unknown reaction is left
        penny.db.mark_messages_processed(
            [
                m.id
                for m in penny.db.get_unprocessed_messages(
                    TEST_SENDER, limit=PREFERENCE_BATCH_LIMIT
                )
                if m.id is not None
            ]
        )

        # Get the outgoing message
        with penny.db.get_session() as session:
            outgoing = session.exec(
                select(MessageLog).where(MessageLog.direction == "outgoing")
            ).first()
            assert outgoing is not None
            outgoing_id = outgoing.id

        # Insert a reaction with an unknown emoji directly
        reaction_id = penny.db.log_message(
            direction="incoming",
            sender=TEST_SENDER,
            content="🤷",  # Not in LIKE_REACTIONS or DISLIKE_REACTIONS
            parent_id=outgoing_id,
            is_reaction=True,
        )
        assert reaction_id is not None

        # Run PreferenceAgent
        result = await penny.preference_agent.execute()
        # Unknown emoji doesn't create preference, but reaction is still processed
        assert result is False, "Should return False (no preference updated)"

        # Verify reaction was marked as processed
        reactions = penny.db.get_user_reactions(TEST_SENDER, limit=PREFERENCE_BATCH_LIMIT)
        assert len(reactions) == 0, "Reaction should be marked as processed even for unknown emoji"


@pytest.mark.asyncio
async def test_preference_agent_extracts_preferences_from_messages(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test that PreferenceAgent extracts preferences from regular user messages:
    1. Send a message mentioning a topic the user likes
    2. Run PreferenceAgent.execute() directly
    3. Verify preferences were extracted from message content
    4. Verify messages are marked as processed
    """
    config = make_config()

    request_count = [0]

    def handler(request: dict, count: int) -> dict:
        request_count[0] += 1
        if request_count[0] == 1:
            # First call: message agent tool call
            return mock_ollama._make_tool_call_response(
                request, "search", {"query": "playing guitar"}
            )
        elif request_count[0] == 2:
            # Second call: message agent final response
            return mock_ollama._make_text_response(
                request, "guitar is awesome! here are some tips 🎸"
            )
        elif request_count[0] == 3:
            # Likes pass: extract guitar as a like
            return mock_ollama._make_text_response(request, '{"topics": ["guitar"]}')
        else:
            # Dislikes pass: nothing
            return mock_ollama._make_text_response(request, '{"topics": []}')

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Send a message about something the user likes
        await signal_server.push_message(sender=TEST_SENDER, content="I love playing guitar!")
        await signal_server.wait_for_message(timeout=10.0)

        # Verify unprocessed messages exist before running agent
        unprocessed = penny.db.get_unprocessed_messages(TEST_SENDER, limit=PREFERENCE_BATCH_LIMIT)
        assert len(unprocessed) == 1

        # Run PreferenceAgent directly
        work_done = await penny.preference_agent.execute()
        assert work_done, "PreferenceAgent should have extracted preferences"

        # Verify preference was added
        prefs = penny.db.get_preferences(TEST_SENDER, PreferenceType.LIKE)
        assert any(p.topic == "guitar" for p in prefs), f"Expected 'guitar' in likes, got {prefs}"

        # Verify messages were marked processed
        unprocessed = penny.db.get_unprocessed_messages(TEST_SENDER, limit=PREFERENCE_BATCH_LIMIT)
        assert len(unprocessed) == 0, "Messages should be marked as processed"


@pytest.mark.asyncio
async def test_preference_agent_skips_processed_messages(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Messages already marked processed should not appear in get_unprocessed_messages."""
    config = make_config()

    async with running_penny(config) as penny:
        # Insert a message and mark it processed
        msg_id = penny.db.log_message(
            direction="incoming",
            sender=TEST_SENDER,
            content="I love jazz music",
        )
        assert msg_id is not None
        penny.db.mark_messages_processed([msg_id])

        # Verify it doesn't appear in unprocessed
        unprocessed = penny.db.get_unprocessed_messages(TEST_SENDER, limit=PREFERENCE_BATCH_LIMIT)
        assert len(unprocessed) == 0

        # Insert another message without marking it
        msg_id2 = penny.db.log_message(
            direction="incoming",
            sender=TEST_SENDER,
            content="I also like rock music",
        )
        assert msg_id2 is not None

        # Only the new message should be unprocessed
        unprocessed = penny.db.get_unprocessed_messages(TEST_SENDER, limit=PREFERENCE_BATCH_LIMIT)
        assert len(unprocessed) == 1
        assert unprocessed[0].id == msg_id2


@pytest.mark.asyncio
async def test_preference_agent_batches_notifications(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test that PreferenceAgent sends a single batched message for multiple likes/dislikes
    instead of one message per preference.
    """
    config = make_config()

    request_count = [0]

    def handler(request: dict, count: int) -> dict:
        request_count[0] += 1
        if request_count[0] == 1:
            # First call: message agent tool call
            return mock_ollama._make_tool_call_response(request, "search", {"query": "hobbies"})
        elif request_count[0] == 2:
            # Second call: message agent final response
            return mock_ollama._make_text_response(request, "great hobbies! here are some tips 🎨")
        elif request_count[0] == 3:
            # Likes pass: extract multiple topics
            return mock_ollama._make_text_response(
                request, '{"topics": ["painting", "drawing", "sculpting"]}'
            )
        else:
            # Dislikes pass: nothing
            return mock_ollama._make_text_response(request, '{"topics": []}')

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Send a message about multiple things the user likes
        await signal_server.push_message(
            sender=TEST_SENDER, content="I love painting, drawing, and sculpting!"
        )
        await signal_server.wait_for_message(timeout=10.0)

        # Clear outgoing messages so we can count only preference notifications
        signal_server.outgoing_messages.clear()

        # Run PreferenceAgent directly
        work_done = await penny.preference_agent.execute()
        assert work_done, "PreferenceAgent should have extracted preferences"

        # Wait for notification messages
        await wait_until(lambda: len(signal_server.outgoing_messages) > 0)

        # Verify preferences were added
        prefs = penny.db.get_preferences(TEST_SENDER, PreferenceType.LIKE)
        assert len(prefs) == 3
        topics = {p.topic for p in prefs}
        assert topics == {"painting", "drawing", "sculpting"}

        # The bug: currently sends 3 separate messages
        # The fix: should send 1 batched message
        assert len(signal_server.outgoing_messages) == 1, (
            "Should send a single batched message for all new preferences, "
            f"but sent {len(signal_server.outgoing_messages)} messages"
        )

        # Verify the message contains all three topics in a bullet list
        notification = signal_server.outgoing_messages[0]["message"]
        assert "painting" in notification
        assert "drawing" in notification
        assert "sculpting" in notification

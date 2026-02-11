"""Integration tests for the ProfileAgent."""

import pytest
from sqlmodel import select

from penny.database.models import MessageLog
from penny.tests.conftest import TEST_SENDER, wait_until


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


@pytest.mark.asyncio
async def test_profile_no_channel(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Test that ProfileAgent returns False when no channel is set."""
    config = make_config()

    async with running_penny(config) as penny:
        from penny.agents import ProfileAgent
        from penny.constants import SYSTEM_PROMPT

        profile_agent = ProfileAgent(
            system_prompt=SYSTEM_PROMPT,
            model=config.ollama_foreground_model,
            ollama_api_url=config.ollama_api_url,
            tools=[],
            db=penny.db,
        )
        # Don't set channel

        result = await profile_agent.execute()
        assert result is False, "Should return False when no channel set"


@pytest.mark.asyncio
async def test_profile_no_senders(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Test that ProfileAgent returns False when no senders exist."""
    config = make_config()

    async with running_penny(config) as penny:
        # The penny.profile_agent already has channel set,
        # but no messages have been sent, so no senders
        result = await penny.profile_agent.execute()
        assert result is False, "Should return False when no senders"


@pytest.mark.asyncio
async def test_profile_unknown_emoji(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
    setup_ollama_flow,
):
    """
    Test that ProfileAgent skips unknown reaction emojis but still marks them processed.
    """
    config = make_config()
    setup_ollama_flow(
        search_query="test query",
        message_response="interesting fact! 🌟",
        background_response='{"topic": "facts"}',
    )

    async with running_penny(config) as penny:
        # Send message to create a sender
        await signal_server.push_message(sender=TEST_SENDER, content="tell me a fact")
        await signal_server.wait_for_message(timeout=10.0)

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

        # Run ProfileAgent
        result = await penny.profile_agent.execute()
        # Unknown emoji doesn't create preference, but reaction is still processed
        assert result is False, "Should return False (no preference updated)"

        # Verify reaction was marked as processed
        reactions = penny.db.get_user_reactions(TEST_SENDER)
        assert len(reactions) == 0, "Reaction should be marked as processed even for unknown emoji"

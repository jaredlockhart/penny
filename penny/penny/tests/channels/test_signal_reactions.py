"""Integration tests for Signal reaction handling."""

import json
import time

import pytest
from sqlmodel import select

from penny.constants import PennyConstants
from penny.database.models import MessageLog
from penny.ollama.embeddings import serialize_embedding
from penny.tests.conftest import TEST_SENDER, wait_until


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
    4. Verify reaction is available for extraction pipeline
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

        # Verify reaction details
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

        # No embedding model configured → no engagements created
        engagements = penny.db.get_user_engagements(TEST_SENDER)
        reaction_engagements = [
            e
            for e in engagements
            if e.engagement_type == PennyConstants.EngagementType.EMOJI_REACTION
        ]
        assert len(reaction_engagements) == 0, "No engagements without embedding model"

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

        # Verify reaction details
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
async def test_reaction_creates_entity_engagements(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
    setup_ollama_flow,
):
    """
    Reacting to Penny's message creates engagement records for entities
    mentioned in that message (matched via embedding similarity).
    """
    config = make_config(ollama_embedding_model="test-embed-model")
    setup_ollama_flow(
        search_query="test query",
        message_response="the KEF LS50 Meta sounds amazing! 🎵",
    )

    # Embed handler returns identical vectors → high cosine similarity
    mock_ollama.set_embed_handler(lambda model, input_text: [[1.0, 0.0, 0.0, 0.0]])

    async with running_penny(config) as penny:
        # Seed entity with matching embedding
        entity = penny.db.get_or_create_entity(TEST_SENDER, "kef ls50 meta")
        assert entity is not None and entity.id is not None
        penny.db.update_entity_embedding(entity.id, serialize_embedding([1.0, 0.0, 0.0, 0.0]))

        # Send message and get response
        await signal_server.push_message(sender=TEST_SENDER, content="tell me about speakers")
        response = await signal_server.wait_for_message(timeout=10.0)
        assert "kef" in response["message"].lower()

        # Get the outgoing message's external_id
        with penny.db.get_session() as session:
            outgoing = session.exec(
                select(MessageLog).where(MessageLog.direction == "outgoing")
            ).first()
            assert outgoing is not None
            assert outgoing.external_id is not None
            external_id = outgoing.external_id

        # React with thumbs up
        await signal_server.push_reaction(
            sender=TEST_SENDER,
            emoji="👍",
            target_timestamp=int(external_id),
        )

        # Wait for engagement to be created
        def engagement_exists():
            engagements = penny.db.get_entity_engagements(TEST_SENDER, entity.id)
            return any(
                e.engagement_type == PennyConstants.EngagementType.EMOJI_REACTION
                for e in engagements
            )

        await wait_until(engagement_exists)

        # Verify engagement details
        engagements = penny.db.get_entity_engagements(TEST_SENDER, entity.id)
        reaction_engagements = [
            e
            for e in engagements
            if e.engagement_type == PennyConstants.EngagementType.EMOJI_REACTION
        ]
        assert len(reaction_engagements) == 1
        eng = reaction_engagements[0]
        assert eng.valence == PennyConstants.EngagementValence.POSITIVE
        # Normal response (has incoming parent) → normal strength
        assert eng.strength == PennyConstants.ENGAGEMENT_STRENGTH_EMOJI_REACTION_NORMAL


@pytest.mark.asyncio
async def test_negative_reaction_on_proactive_message(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Negative reaction on a proactive message (parent_id=None) creates a
    high-strength negative engagement — the 'stop telling me about this' mechanism.
    """
    config = make_config(ollama_embedding_model="test-embed-model")
    mock_ollama.set_default_flow(
        search_query="test query",
        final_response="test response 🌟",
    )

    # Embed handler returns identical vectors → high cosine similarity
    mock_ollama.set_embed_handler(lambda model, input_text: [[1.0, 0.0, 0.0, 0.0]])

    async with running_penny(config) as penny:
        # Seed entity with matching embedding
        entity = penny.db.get_or_create_entity(TEST_SENDER, "sports news")
        assert entity is not None and entity.id is not None
        penny.db.update_entity_embedding(entity.id, serialize_embedding([1.0, 0.0, 0.0, 0.0]))

        # Insert a proactive outgoing message directly (simulates discovery/research)
        proactive_msg_id = penny.db.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            penny.channel.sender_id,
            "hey! the latest sports news is really exciting today! 🏈",
            parent_id=None,  # No parent = proactive
        )
        assert proactive_msg_id is not None
        # Set an external_id so the reaction can target it
        external_id = str(int(time.time() * 1000))
        penny.db.set_external_id(proactive_msg_id, external_id)

        # React with thumbs down to the proactive message
        await signal_server.push_reaction(
            sender=TEST_SENDER,
            emoji="👎",
            target_timestamp=int(external_id),
        )

        # Wait for engagement to be created
        def engagement_exists():
            engagements = penny.db.get_entity_engagements(TEST_SENDER, entity.id)
            return any(
                e.engagement_type == PennyConstants.EngagementType.EMOJI_REACTION
                for e in engagements
            )

        await wait_until(engagement_exists)

        # Verify engagement details — strong negative engagement
        engagements = penny.db.get_entity_engagements(TEST_SENDER, entity.id)
        reaction_engagements = [
            e
            for e in engagements
            if e.engagement_type == PennyConstants.EngagementType.EMOJI_REACTION
        ]
        assert len(reaction_engagements) == 1
        eng = reaction_engagements[0]
        assert eng.valence == PennyConstants.EngagementValence.NEGATIVE
        assert eng.strength == PennyConstants.ENGAGEMENT_STRENGTH_EMOJI_REACTION_PROACTIVE_NEGATIVE

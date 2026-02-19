"""Integration tests for the unified ExtractionPipeline agent."""

import json

import pytest
from sqlmodel import select

from penny.agents.extraction import _is_valid_entity_name
from penny.constants import PennyConstants
from penny.database.models import SearchLog
from penny.ollama.embeddings import deserialize_embedding, serialize_embedding
from penny.tests.conftest import TEST_SENDER

# --- Search log entity/fact extraction (migrated from test_entity_extractor) ---


@pytest.mark.asyncio
async def test_extraction_processes_search_log(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test two-pass entity extraction from SearchLog entries:
    1. Send a message (triggers search, creates SearchLog entry)
    2. Run ExtractionPipeline.execute() directly
    3. Pass 1: identify entities (known + new), Pass 2: extract facts per entity
    4. Verify entities and facts were stored as individual Fact rows
    5. Verify SearchLog is marked as extracted
    """
    config = make_config()

    request_count = [0]

    def handler(request: dict, count: int) -> dict:
        request_count[0] += 1
        if request_count[0] == 1:
            return mock_ollama._make_tool_call_response(
                request, "search", {"query": "best hifi speakers"}
            )
        elif request_count[0] == 2:
            return mock_ollama._make_text_response(request, "check out the KEF LS50 Meta! 🎵")
        elif request_count[0] == 3:
            # Pass 1: identify entities (no known entities yet)
            return mock_ollama._make_text_response(
                request,
                json.dumps(
                    {
                        "known": [],
                        "new": [
                            {"name": "KEF LS50 Meta"},
                        ],
                    }
                ),
            )
        elif request_count[0] == 4:
            # Pass 2: facts for KEF LS50 Meta
            return mock_ollama._make_text_response(
                request,
                json.dumps(
                    {
                        "facts": [
                            "Costs $1,599 per pair",
                            "Features Metamaterial Absorption Technology",
                        ]
                    }
                ),
            )
        else:
            # Notification composition call
            return mock_ollama._make_text_response(
                request, "Hey! I just found out about KEF LS50 Meta — pretty cool speaker!"
            )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        await signal_server.push_message(
            sender=TEST_SENDER, content="what are the best hifi speakers?"
        )
        await signal_server.wait_for_message(timeout=10.0)

        # Verify SearchLog was created
        with penny.db.get_session() as session:
            search_logs = list(session.exec(select(SearchLog)).all())
            assert len(search_logs) >= 1

        # Run ExtractionPipeline directly
        work_done = await penny.extraction_pipeline.execute()
        assert work_done, "ExtractionPipeline should have processed the search log"

        # Verify entity was stored
        entities = penny.db.get_user_entities(TEST_SENDER)
        assert len(entities) >= 1
        entity = next(e for e in entities if e.name == "kef ls50 meta")

        # Verify facts stored as individual Fact rows
        facts = penny.db.get_entity_facts(entity.id)
        fact_contents = [f.content for f in facts]
        assert "Costs $1,599 per pair" in fact_contents
        assert "Features Metamaterial Absorption Technology" in fact_contents

        # Verify facts have source_search_log_id set
        assert all(f.source_search_log_id == search_logs[0].id for f in facts)

        # Verify SEARCH_INITIATED engagement was created
        engagements = penny.db.get_entity_engagements(TEST_SENDER, entity.id)
        search_engagements = [
            e
            for e in engagements
            if e.engagement_type == PennyConstants.EngagementType.SEARCH_INITIATED
        ]
        assert len(search_engagements) >= 1
        assert search_engagements[0].valence == PennyConstants.EngagementValence.POSITIVE
        assert search_engagements[0].strength == PennyConstants.ENGAGEMENT_STRENGTH_SEARCH_INITIATED

        # Verify SearchLog is marked as extracted
        with penny.db.get_session() as session:
            sl = session.get(SearchLog, search_logs[0].id)
            assert sl is not None
            assert sl.extracted is True

        # Verify fact discovery notification was sent (model-composed)
        notification_messages = [
            msg
            for msg in signal_server.outgoing_messages
            if "KEF LS50 Meta" in msg["message"]
            and msg["message"] != "check out the KEF LS50 Meta! 🎵"
        ]
        assert len(notification_messages) >= 1, (
            "Expected model-composed fact discovery notification, "
            f"got messages: {[m['message'][:80] for m in signal_server.outgoing_messages]}"
        )


@pytest.mark.asyncio
async def test_extraction_skips_processed_search_logs(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test that ExtractionPipeline doesn't reprocess entries:
    1. Process entries once (extracted flag set)
    2. Run execute() again
    3. Verify no additional Ollama calls for extraction
    """
    config = make_config()

    request_count = [0]
    extraction_calls = [0]

    def handler(request: dict, count: int) -> dict:
        request_count[0] += 1
        if request_count[0] == 1:
            return mock_ollama._make_tool_call_response(
                request, "search", {"query": "local ai hardware"}
            )
        elif request_count[0] == 2:
            return mock_ollama._make_text_response(request, "check out the NVIDIA Jetson! 🤖")
        else:
            extraction_calls[0] += 1
            if extraction_calls[0] == 1:
                return mock_ollama._make_text_response(
                    request,
                    json.dumps(
                        {
                            "known": [],
                            "new": [
                                {"name": "NVIDIA Jetson"},
                            ],
                        }
                    ),
                )
            else:
                return mock_ollama._make_text_response(
                    request,
                    json.dumps({"facts": ["Edge AI computing module"]}),
                )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        await signal_server.push_message(
            sender=TEST_SENDER, content="tell me about local ai hardware"
        )
        await signal_server.wait_for_message(timeout=10.0)

        # First extraction run
        await penny.extraction_pipeline.execute()
        first_extraction_calls = extraction_calls[0]
        assert first_extraction_calls >= 1

        # Second extraction run — should find nothing new
        work_done = await penny.extraction_pipeline.execute()
        assert work_done is False, "Should return False when nothing to process"
        assert extraction_calls[0] == first_extraction_calls, (
            "Should not make additional Ollama calls"
        )


@pytest.mark.asyncio
async def test_extraction_known_and_new_entities(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test mixed known + new entity extraction:
    1. Pre-seed an existing entity with a fact
    2. Send a message that mentions the existing entity and a new one
    3. Verify both entities have facts and SearchLog is marked as extracted
    """
    config = make_config()

    request_count = [0]
    pass2_calls = [0]

    def handler(request: dict, count: int) -> dict:
        request_count[0] += 1
        if request_count[0] == 1:
            return mock_ollama._make_tool_call_response(
                request, "search", {"query": "KEF LS50 vs Wharfedale Linton"}
            )
        elif request_count[0] == 2:
            return mock_ollama._make_text_response(request, "both are great speakers! 🎶")
        elif request_count[0] == 3:
            return mock_ollama._make_text_response(
                request,
                json.dumps(
                    {
                        "known": ["kef ls50 meta"],
                        "new": [
                            {"name": "Wharfedale Linton"},
                        ],
                    }
                ),
            )
        else:
            pass2_calls[0] += 1
            if pass2_calls[0] == 1:
                return mock_ollama._make_text_response(
                    request,
                    json.dumps(
                        {"facts": ["Heritage design with modern drivers", "Costs $1,199 per pair"]}
                    ),
                )
            else:
                return mock_ollama._make_text_response(
                    request,
                    json.dumps({"facts": ["Won What Hi-Fi 2024 award"]}),
                )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        entity = penny.db.get_or_create_entity(TEST_SENDER, "kef ls50 meta")
        assert entity is not None and entity.id is not None
        penny.db.add_fact(entity.id, "Costs $1,599 per pair")

        await signal_server.push_message(
            sender=TEST_SENDER, content="compare KEF LS50 Meta vs Wharfedale Linton"
        )
        await signal_server.wait_for_message(timeout=10.0)

        work_done = await penny.extraction_pipeline.execute()
        assert work_done

        entities = penny.db.get_user_entities(TEST_SENDER)
        assert len(entities) == 2

        kef = next(e for e in entities if e.name == "kef ls50 meta")
        kef_facts = [f.content for f in penny.db.get_entity_facts(kef.id)]
        assert "Costs $1,599 per pair" in kef_facts
        assert "Won What Hi-Fi 2024 award" in kef_facts

        wharfedale = next(e for e in entities if e.name == "wharfedale linton")
        wharfedale_facts = [f.content for f in penny.db.get_entity_facts(wharfedale.id)]
        assert "Heritage design with modern drivers" in wharfedale_facts
        assert "Costs $1,199 per pair" in wharfedale_facts

        with penny.db.get_session() as session:
            search_logs = list(
                session.exec(select(SearchLog).where(SearchLog.extracted == True)).all()  # noqa: E712
            )
            assert len(search_logs) >= 1


@pytest.mark.asyncio
async def test_extraction_empty_results(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test that ExtractionPipeline handles empty extraction results gracefully
    and still marks the SearchLog as extracted.
    """
    config = make_config()

    request_count = [0]

    def handler(request: dict, count: int) -> dict:
        request_count[0] += 1
        if request_count[0] == 1:
            return mock_ollama._make_tool_call_response(request, "search", {"query": "hello"})
        elif request_count[0] == 2:
            return mock_ollama._make_text_response(request, "hey there! 👋")
        else:
            return mock_ollama._make_text_response(request, json.dumps({"known": [], "new": []}))

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="hello!")
        await signal_server.wait_for_message(timeout=10.0)

        work_done = await penny.extraction_pipeline.execute()
        assert work_done is False, "No entities extracted means no work done"

        with penny.db.get_session() as session:
            search_logs = list(session.exec(select(SearchLog)).all())
            assert len(search_logs) >= 1
            assert all(sl.extracted for sl in search_logs)

        entities = penny.db.get_user_entities(TEST_SENDER)
        assert len(entities) == 0


@pytest.mark.asyncio
async def test_extraction_generates_embeddings(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test that ExtractionPipeline generates embeddings for facts and entities
    when embedding_model is configured.
    """
    config = make_config(ollama_embedding_model="nomic-embed-text")

    request_count = [0]

    def handler(request: dict, count: int) -> dict:
        request_count[0] += 1
        if request_count[0] == 1:
            return mock_ollama._make_tool_call_response(
                request, "search", {"query": "best speakers"}
            )
        elif request_count[0] == 2:
            return mock_ollama._make_text_response(request, "check out KEF speakers! 🎵")
        elif request_count[0] == 3:
            return mock_ollama._make_text_response(
                request,
                json.dumps({"known": [], "new": [{"name": "KEF LS50 Meta"}]}),
            )
        else:
            return mock_ollama._make_text_response(
                request,
                json.dumps({"facts": ["Costs $1,599", "Award winning"]}),
            )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="best speakers?")
        await signal_server.wait_for_message(timeout=10.0)

        work_done = await penny.extraction_pipeline.execute()
        assert work_done

        entities = penny.db.get_user_entities(TEST_SENDER)
        entity = next(e for e in entities if e.name == "kef ls50 meta")
        facts = penny.db.get_entity_facts(entity.id)
        assert len(facts) == 2
        for fact in facts:
            assert fact.embedding is not None, f"Fact '{fact.content}' should have embedding"
            vec = deserialize_embedding(fact.embedding)
            assert len(vec) == 4

        assert entity.embedding is not None, "Entity should have embedding"

        assert len(mock_ollama.embed_requests) >= 2
        assert len(penny.db.get_facts_without_embeddings(limit=10)) == 0
        assert len(penny.db.get_entities_without_embeddings(limit=10)) == 0


@pytest.mark.asyncio
async def test_extraction_backfills_all_embeddings(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test that ExtractionPipeline backfills embeddings for existing
    entities, facts, AND preferences that don't have them.
    """
    config = make_config(ollama_embedding_model="nomic-embed-text")

    def handler(request: dict, count: int) -> dict:
        return mock_ollama._make_tool_call_response(request, "search", {"query": "test"})

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Pre-seed entity, facts, and preferences WITHOUT embeddings
        entity = penny.db.get_or_create_entity(TEST_SENDER, "test entity")
        assert entity is not None and entity.id is not None
        penny.db.add_fact(entity.id, "fact one")
        penny.db.add_fact(entity.id, "fact two")
        penny.db.add_preference(TEST_SENDER, "cats", "like")
        penny.db.add_preference(TEST_SENDER, "dogs", "like")

        assert len(penny.db.get_facts_without_embeddings(limit=10)) == 2
        assert len(penny.db.get_entities_without_embeddings(limit=10)) == 1
        assert len(penny.db.get_preferences_without_embeddings(limit=10)) == 2

        work_done = await penny.extraction_pipeline.execute()
        assert work_done, "Backfill should count as work done"

        assert len(penny.db.get_facts_without_embeddings(limit=10)) == 0
        assert len(penny.db.get_entities_without_embeddings(limit=10)) == 0
        assert len(penny.db.get_preferences_without_embeddings(limit=10)) == 0


# --- Message entity extraction (new functionality) ---


@pytest.mark.asyncio
async def test_extraction_processes_messages_for_entities(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test that ExtractionPipeline extracts entities from user messages
    and creates MESSAGE_MENTION engagements.
    """
    config = make_config()

    request_count = [0]

    def handler(request: dict, count: int) -> dict:
        request_count[0] += 1
        if request_count[0] == 1:
            # Message agent tool call
            return mock_ollama._make_tool_call_response(
                request, "search", {"query": "KEF LS50 Meta review"}
            )
        elif request_count[0] == 2:
            # Message agent final response
            return mock_ollama._make_text_response(request, "great choice! 🎵")
        elif request_count[0] == 3:
            # Entity identification from message
            return mock_ollama._make_text_response(
                request,
                json.dumps({"known": [], "new": [{"name": "KEF LS50 Meta"}]}),
            )
        elif request_count[0] == 4:
            # Fact extraction from message
            return mock_ollama._make_text_response(
                request,
                json.dumps({"facts": ["User is interested in this speaker"]}),
            )
        else:
            # Preference extraction passes (likes, dislikes)
            return mock_ollama._make_text_response(request, '{"topics": []}')

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        await signal_server.push_message(
            sender=TEST_SENDER, content="I just bought a KEF LS50 Meta and it sounds amazing"
        )
        await signal_server.wait_for_message(timeout=10.0)

        # Pre-mark search logs so only message processing runs
        for sl in penny.db.get_unprocessed_search_logs(limit=100):
            penny.db.mark_search_extracted(sl.id)

        work_done = await penny.extraction_pipeline.execute()
        assert work_done

        # Verify entity was extracted from the message
        entities = penny.db.get_user_entities(TEST_SENDER)
        assert any(e.name == "kef ls50 meta" for e in entities)

        entity = next(e for e in entities if e.name == "kef ls50 meta")

        # Verify facts have source_message_id set (not source_search_log_id)
        facts = penny.db.get_entity_facts(entity.id)
        message_sourced_facts = [f for f in facts if f.source_message_id is not None]
        assert len(message_sourced_facts) >= 1

        # Verify MESSAGE_MENTION engagement was created
        engagements = penny.db.get_entity_engagements(TEST_SENDER, entity.id)
        mention_engagements = [
            e
            for e in engagements
            if e.engagement_type == PennyConstants.EngagementType.MESSAGE_MENTION
        ]
        assert len(mention_engagements) >= 1
        assert mention_engagements[0].valence == PennyConstants.EngagementValence.NEUTRAL
        assert mention_engagements[0].strength == PennyConstants.ENGAGEMENT_STRENGTH_MESSAGE_MENTION


@pytest.mark.asyncio
async def test_extraction_creates_follow_up_engagements(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    When a user quote-replies to Penny's message, FOLLOW_UP_QUESTION
    engagements are created for entities related to the parent message.
    Short follow-ups (< 20 chars) should still trigger detection.
    """
    config = make_config(ollama_embedding_model="test-embed-model")

    # Embed handler: return identical vectors so everything matches
    def embed_handler(model, input_text):
        texts = [input_text] if isinstance(input_text, str) else input_text
        return [[1.0, 0.0, 0.0, 0.0]] * len(texts)

    mock_ollama.set_embed_handler(embed_handler)

    async with running_penny(config) as penny:
        # Seed entity with embedding
        entity = penny.db.get_or_create_entity(TEST_SENDER, "kef ls50 meta")
        assert entity is not None and entity.id is not None
        penny.db.update_entity_embedding(entity.id, serialize_embedding([1.0, 0.0, 0.0, 0.0]))
        penny.db.add_fact(entity.id, "Costs $1,599 per pair")

        # Simulate Penny's outgoing message about the entity
        outgoing_id = penny.db.log_message(
            direction="outgoing",
            sender="penny",
            content="the KEF LS50 Meta costs $1,599 and uses Metamaterial Absorption Technology!",
        )
        assert outgoing_id is not None

        # User replies with a short follow-up (< 20 chars, skipped by entity extraction)
        incoming_id = penny.db.log_message(
            direction="incoming",
            sender=TEST_SENDER,
            content="how much?",  # 9 chars — below MIN_EXTRACTION_MESSAGE_LENGTH
            parent_id=outgoing_id,
        )
        assert incoming_id is not None

        work_done = await penny.extraction_pipeline.execute()
        assert work_done, "Follow-up detection should count as work done"

        # Verify FOLLOW_UP_QUESTION engagement was created
        engagements = penny.db.get_entity_engagements(TEST_SENDER, entity.id)
        follow_up_engagements = [
            e
            for e in engagements
            if e.engagement_type == PennyConstants.EngagementType.FOLLOW_UP_QUESTION
        ]
        assert len(follow_up_engagements) == 1
        assert follow_up_engagements[0].valence == PennyConstants.EngagementValence.POSITIVE
        assert (
            follow_up_engagements[0].strength
            == PennyConstants.ENGAGEMENT_STRENGTH_FOLLOW_UP_QUESTION
        )
        assert follow_up_engagements[0].source_message_id == incoming_id


@pytest.mark.asyncio
async def test_extraction_no_follow_up_for_reply_to_user(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Replying to an incoming message (not Penny's) should NOT create
    FOLLOW_UP_QUESTION engagements.
    """
    config = make_config(ollama_embedding_model="test-embed-model")

    def embed_handler(model, input_text):
        texts = [input_text] if isinstance(input_text, str) else input_text
        return [[1.0, 0.0, 0.0, 0.0]] * len(texts)

    mock_ollama.set_embed_handler(embed_handler)

    async with running_penny(config) as penny:
        entity = penny.db.get_or_create_entity(TEST_SENDER, "test entity")
        assert entity is not None and entity.id is not None
        penny.db.update_entity_embedding(entity.id, serialize_embedding([1.0, 0.0, 0.0, 0.0]))

        # Parent is an incoming message (user's own message, not Penny's)
        parent_id = penny.db.log_message(
            direction="incoming",
            sender=TEST_SENDER,
            content="I love this test entity so much!",
        )
        assert parent_id is not None
        penny.db.mark_messages_processed([parent_id])

        # Reply to own message
        reply_id = penny.db.log_message(
            direction="incoming",
            sender=TEST_SENDER,
            content="actually let me tell you more about it",
            parent_id=parent_id,
        )
        assert reply_id is not None

        await penny.extraction_pipeline.execute()

        engagements = penny.db.get_entity_engagements(TEST_SENDER, entity.id)
        follow_up_engagements = [
            e
            for e in engagements
            if e.engagement_type == PennyConstants.EngagementType.FOLLOW_UP_QUESTION
        ]
        assert len(follow_up_engagements) == 0, (
            "No FOLLOW_UP_QUESTION for replies to user's own messages"
        )


@pytest.mark.asyncio
async def test_extraction_skips_short_messages(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Messages shorter than MIN_EXTRACTION_MESSAGE_LENGTH are skipped for entity extraction."""
    config = make_config()

    async with running_penny(config) as penny:
        # Insert a short message directly
        msg_id = penny.db.log_message(
            direction="incoming",
            sender=TEST_SENDER,
            content="hi",  # < 20 chars
        )
        assert msg_id is not None

        work_done = await penny.extraction_pipeline.execute()
        # No entities should be extracted, but message gets marked processed
        assert work_done is False

        # Message should be marked as processed
        unprocessed = penny.db.get_unprocessed_messages(
            TEST_SENDER, limit=PennyConstants.PREFERENCE_BATCH_LIMIT
        )
        assert len(unprocessed) == 0


@pytest.mark.asyncio
async def test_extraction_skips_commands(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Messages starting with / are commands and should be skipped for entity extraction."""
    config = make_config()

    async with running_penny(config) as penny:
        msg_id = penny.db.log_message(
            direction="incoming",
            sender=TEST_SENDER,
            content="/like espresso machines",  # command
        )
        assert msg_id is not None

        work_done = await penny.extraction_pipeline.execute()
        assert work_done is False

        unprocessed = penny.db.get_unprocessed_messages(
            TEST_SENDER, limit=PennyConstants.PREFERENCE_BATCH_LIMIT
        )
        assert len(unprocessed) == 0


# --- Preference extraction (migrated from test_preference) ---


@pytest.mark.asyncio
async def test_extraction_skips_processed_messages(
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
        msg_id = penny.db.log_message(
            direction="incoming",
            sender=TEST_SENDER,
            content="I love jazz music",
        )
        assert msg_id is not None
        penny.db.mark_messages_processed([msg_id])

        unprocessed = penny.db.get_unprocessed_messages(
            TEST_SENDER, limit=PennyConstants.PREFERENCE_BATCH_LIMIT
        )
        assert len(unprocessed) == 0

        msg_id2 = penny.db.log_message(
            direction="incoming",
            sender=TEST_SENDER,
            content="I also like rock music",
        )
        assert msg_id2 is not None

        unprocessed = penny.db.get_unprocessed_messages(
            TEST_SENDER, limit=PennyConstants.PREFERENCE_BATCH_LIMIT
        )
        assert len(unprocessed) == 1
        assert unprocessed[0].id == msg_id2


# --- Two-mode extraction + entity validation ---


@pytest.mark.asyncio
async def test_extraction_penny_enrichment_known_only(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    penny_enrichment searches only produce facts for known entities:
    1. Pre-seed a known entity
    2. Create a SearchLog with trigger=penny_enrichment
    3. Mock LLM to return the known entity + facts
    4. Verify: known entity gets new facts, NO new entities created,
       NO SEARCH_INITIATED engagements
    """
    config = make_config()

    request_count = [0]

    def handler(request: dict, count: int) -> dict:
        request_count[0] += 1
        if request_count[0] == 1:
            # Known-only identification: return the known entity
            return mock_ollama._make_text_response(
                request,
                json.dumps({"known": ["kef ls50 meta"]}),
            )
        else:
            # Fact extraction for the known entity
            return mock_ollama._make_text_response(
                request,
                json.dumps({"facts": ["Won What Hi-Fi 2024 award"]}),
            )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Pre-seed known entity with a fact
        entity = penny.db.get_or_create_entity(TEST_SENDER, "kef ls50 meta")
        assert entity is not None and entity.id is not None
        penny.db.add_fact(entity.id, "Costs $1,599 per pair")

        # Seed a recent incoming message so find_sender_for_timestamp works
        penny.db.log_message(
            direction="incoming", sender=TEST_SENDER, content="tell me more about speakers"
        )

        # Create a SearchLog with penny_enrichment trigger
        penny.db.log_search(
            query="kef ls50 meta latest",
            response="The KEF LS50 Meta won the What Hi-Fi 2024 award.",
            trigger=PennyConstants.SearchTrigger.PENNY_ENRICHMENT,
        )

        work_done = await penny.extraction_pipeline.execute()
        assert work_done, "Should have extracted facts for the known entity"

        # Verify known entity got new facts
        facts = penny.db.get_entity_facts(entity.id)
        fact_contents = [f.content for f in facts]
        assert "Won What Hi-Fi 2024 award" in fact_contents

        # Verify NO new entities were created
        entities = penny.db.get_user_entities(TEST_SENDER)
        assert len(entities) == 1, (
            f"Expected only 1 entity, got {len(entities)}: {[e.name for e in entities]}"
        )

        # Verify NO SEARCH_INITIATED engagements
        engagements = penny.db.get_entity_engagements(TEST_SENDER, entity.id)
        search_engagements = [
            e
            for e in engagements
            if e.engagement_type == PennyConstants.EngagementType.SEARCH_INITIATED
        ]
        assert len(search_engagements) == 0, (
            "penny_enrichment should not create SEARCH_INITIATED engagements"
        )

        # Verify SearchLog is marked as extracted
        with penny.db.get_session() as session:
            search_logs = list(session.exec(select(SearchLog)).all())
            assert all(sl.extracted for sl in search_logs)

        # Verify NO fact discovery notification was sent (learn loop handles its own)
        fact_notifications = [
            msg
            for msg in signal_server.outgoing_messages
            if "I just discovered" in msg.get("message", "")
            or "I just learned" in msg.get("message", "")
        ]
        assert len(fact_notifications) == 0, (
            "penny_enrichment should not trigger fact discovery notifications"
        )


def test_structural_entity_name_validation():
    """_is_valid_entity_name rejects garbage names and accepts valid ones."""
    # Valid entity names
    assert _is_valid_entity_name("KEF LS50 Meta") is True
    assert _is_valid_entity_name("Leonard Susskind") is True
    assert _is_valid_entity_name("ROCm") is True
    assert _is_valid_entity_name("SYK model") is True
    assert _is_valid_entity_name("a") is True

    # Too many words (> 8)
    assert (
        _is_valid_entity_name("this is a very long entity name that exceeds eight words total")
        is False
    )

    # Numbered list items
    assert _is_valid_entity_name("1. Some Entity") is False
    assert _is_valid_entity_name("23. Another Item") is False

    # URLs
    assert _is_valid_entity_name("check out https://example.com") is False

    # Markdown bold
    assert _is_valid_entity_name("**bold name**") is False

    # Newlines
    assert _is_valid_entity_name("name with\nnewline") is False

    # LLM output artifacts
    assert _is_valid_entity_name("{topic} description") is False
    assert _is_valid_entity_name("some confidence score: high entity") is False
    assert _is_valid_entity_name("result-brief: something") is False
    assert _is_valid_entity_name("{description} of thing") is False


@pytest.mark.asyncio
async def test_semantic_entity_name_validation(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Semantically unrelated entity candidates are rejected via embedding similarity.
    """
    config = make_config(ollama_embedding_model="test-embed-model")

    request_count = [0]

    def handler(request: dict, count: int) -> dict:
        request_count[0] += 1
        if request_count[0] == 1:
            # Entity identification: return a related and an unrelated entity
            return mock_ollama._make_text_response(
                request,
                json.dumps(
                    {
                        "known": [],
                        "new": [
                            {"name": "KEF LS50 Meta"},
                            {"name": "Random Conference Sponsor"},
                        ],
                    }
                ),
            )
        else:
            # Fact extraction
            return mock_ollama._make_text_response(
                request,
                json.dumps({"facts": ["A well-known product"]}),
            )

    mock_ollama.set_response_handler(handler)

    # Embed handler: return high similarity for KEF (related to speakers query),
    # low similarity for the unrelated entity
    def embed_handler(model, input_text):
        texts = [input_text] if isinstance(input_text, str) else input_text
        vecs = []
        for text in texts:
            if "kef" in text.lower() or "speaker" in text.lower():
                vecs.append([1.0, 0.0, 0.0, 0.0])
            else:
                # Orthogonal vector — cosine similarity = 0.0
                vecs.append([0.0, 1.0, 0.0, 0.0])
        return vecs

    mock_ollama.set_embed_handler(embed_handler)

    async with running_penny(config) as penny:
        # Seed a recent incoming message so find_sender_for_timestamp works
        penny.db.log_message(direction="incoming", sender=TEST_SENDER, content="best speakers?")

        # Create a SearchLog about speakers
        penny.db.log_search(
            query="best bookshelf speakers",
            response="The KEF LS50 Meta is excellent. "
            "Also, Random Conference Sponsor attended CES.",
            trigger=PennyConstants.SearchTrigger.USER_MESSAGE,
        )

        work_done = await penny.extraction_pipeline.execute()
        assert work_done

        entities = penny.db.get_user_entities(TEST_SENDER)
        entity_names = [e.name for e in entities]

        # KEF should be created (semantically related to "best bookshelf speakers")
        assert "kef ls50 meta" in entity_names

        # Random Conference Sponsor should be rejected (semantically unrelated)
        assert "random conference sponsor" not in entity_names, (
            f"Unrelated entity should be rejected by semantic filter, got: {entity_names}"
        )


# --- Entity pre-filter ---


@pytest.mark.asyncio
async def test_extraction_prefilters_entities_by_embedding(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    When entity count exceeds ENTITY_PREFILTER_MIN_COUNT, the extraction pipeline
    pre-filters entities by embedding similarity before sending to the LLM.
    Only semantically relevant entities should appear in the identification prompt.
    """
    config = make_config(ollama_embedding_model="test-embed-model")

    # Embed handler: speaker-related text gets [1,0,0,0], everything else [0,1,0,0]
    def embed_handler(model, input_text):
        texts = [input_text] if isinstance(input_text, str) else input_text
        vecs = []
        for text in texts:
            if "speaker" in text.lower() or "kef" in text.lower() or "sonos" in text.lower():
                vecs.append([1.0, 0.0, 0.0, 0.0])
            else:
                vecs.append([0.0, 1.0, 0.0, 0.0])
        return vecs

    mock_ollama.set_embed_handler(embed_handler)

    # Track which entity names the LLM sees in the identification prompt
    seen_entity_names: list[str] = []

    request_count = [0]

    def handler(request: dict, count: int) -> dict:
        request_count[0] += 1
        messages = request.get("messages", [])
        prompt = messages[-1]["content"] if messages else request.get("prompt", "")

        if "Known entities" in prompt:
            for line in prompt.split("\n"):
                if line.strip().startswith("- "):
                    seen_entity_names.append(line.strip()[2:])
            return mock_ollama._make_text_response(
                request,
                json.dumps({"known": ["kef ls50 meta"], "new": []}),
            )
        elif "Entity:" in prompt:
            return mock_ollama._make_text_response(
                request,
                json.dumps({"facts": ["Great bookshelf speaker"]}),
            )
        else:
            return mock_ollama._make_text_response(
                request,
                json.dumps({"known": [], "new": []}),
            )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Create 2 speaker-related entities with similar embeddings
        kef = penny.db.get_or_create_entity(TEST_SENDER, "kef ls50 meta")
        assert kef is not None and kef.id is not None
        penny.db.update_entity_embedding(kef.id, serialize_embedding([1.0, 0.0, 0.0, 0.0]))
        penny.db.add_fact(kef.id, "Costs $1,599")

        sonos = penny.db.get_or_create_entity(TEST_SENDER, "sonos era 300")
        assert sonos is not None and sonos.id is not None
        penny.db.update_entity_embedding(sonos.id, serialize_embedding([1.0, 0.0, 0.0, 0.0]))
        penny.db.add_fact(sonos.id, "Spatial audio speaker")

        # Create 23 unrelated entities with orthogonal embeddings (total 25 > threshold 20)
        for i in range(23):
            entity = penny.db.get_or_create_entity(TEST_SENDER, f"unrelated entity {i}")
            assert entity is not None and entity.id is not None
            penny.db.update_entity_embedding(entity.id, serialize_embedding([0.0, 1.0, 0.0, 0.0]))

        assert len(penny.db.get_user_entities(TEST_SENDER)) == 25

        # Seed message and search log about speakers
        penny.db.log_message(
            direction="incoming", sender=TEST_SENDER, content="tell me about speakers"
        )
        penny.db.log_search(
            query="best bookshelf speakers 2025",
            response="The KEF LS50 Meta remains a top pick for bookshelf speakers.",
            trigger=PennyConstants.SearchTrigger.USER_MESSAGE,
        )

        await penny.extraction_pipeline.execute()

        # Only speaker-related entities should appear in the LLM prompt
        assert "kef ls50 meta" in seen_entity_names
        assert "sonos era 300" in seen_entity_names
        assert not any("unrelated" in name for name in seen_entity_names), (
            f"Unrelated entities should be filtered out, but LLM saw: {seen_entity_names}"
        )

        # Dedup still works: kef ls50 meta not duplicated
        kef_entities = [
            e for e in penny.db.get_user_entities(TEST_SENDER) if e.name == "kef ls50 meta"
        ]
        assert len(kef_entities) == 1


# --- Insertion-time entity dedup ---


@pytest.mark.asyncio
async def test_extraction_deduplicates_entities_at_insertion_time(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    When the LLM returns an entity name that is a duplicate of an existing entity
    (by TCR + embedding similarity), the extraction pipeline routes facts to the
    existing entity instead of creating a new one.
    """
    config = make_config(ollama_embedding_model="test-embed-model")

    # Embed handler: "stanford" and "stanford university" get identical vectors,
    # and are also similar to the search content about stanford
    def embed_handler(model, input_text):
        texts = [input_text] if isinstance(input_text, str) else input_text
        vecs = []
        for text in texts:
            if "stanford" in text.lower():
                vecs.append([1.0, 0.0, 0.0, 0.0])
            else:
                vecs.append([0.0, 1.0, 0.0, 0.0])
        return vecs

    mock_ollama.set_embed_handler(embed_handler)

    request_count = [0]

    def handler(request: dict, count: int) -> dict:
        request_count[0] += 1
        messages = request.get("messages", [])
        prompt = messages[-1]["content"] if messages else ""
        if "Entity:" in prompt:
            return mock_ollama._make_text_response(
                request,
                json.dumps({"facts": ["Located in California"]}),
            )
        elif "topics" in prompt.lower():
            return mock_ollama._make_text_response(request, '{"topics": []}')
        elif "New facts:" in prompt:
            return mock_ollama._make_text_response(request, "dedup-notification")
        else:
            # LLM returns "stanford university" as a new entity
            return mock_ollama._make_text_response(
                request,
                json.dumps({"known": [], "new": [{"name": "Stanford University"}]}),
            )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Pre-seed "stanford" with embedding
        stanford = penny.db.get_or_create_entity(TEST_SENDER, "stanford")
        assert stanford is not None and stanford.id is not None
        penny.db.add_fact(stanford.id, "Founded in 1885")
        penny.db.update_entity_embedding(stanford.id, serialize_embedding([1.0, 0.0, 0.0, 0.0]))

        # Seed message and search log
        msg_id = penny.db.log_message(
            direction="incoming",
            sender=TEST_SENDER,
            content="tell me about stanford physics",
        )
        penny.db.mark_messages_processed([msg_id])

        penny.db.log_search(
            query="stanford physics department",
            response="Stanford University has a renowned physics department.",
            trigger=PennyConstants.SearchTrigger.USER_MESSAGE,
        )

        await penny.extraction_pipeline.execute()

        # Verify NO new entity was created — "stanford university" should be deduped
        entities = penny.db.get_user_entities(TEST_SENDER)
        entity_names = [e.name for e in entities]
        assert "stanford" in entity_names
        assert "stanford university" not in entity_names, (
            f"Duplicate entity should not be created, got: {entity_names}"
        )
        assert len(entities) == 1

        # Verify the new fact was attached to the existing "stanford" entity
        facts = penny.db.get_entity_facts(stanford.id)
        fact_contents = [f.content for f in facts]
        assert "Founded in 1885" in fact_contents
        assert "Located in California" in fact_contents


@pytest.mark.asyncio
async def test_extraction_discards_fragment_entities(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    When the LLM returns both a multi-word entity and its single-word fragments
    in the same batch (e.g., "totem loon", "totem", "loon"), the fragments are
    discarded as token-subsets of the full entity name.
    """
    config = make_config()

    request_count = [0]

    def handler(request: dict, count: int) -> dict:
        request_count[0] += 1
        messages = request.get("messages", [])
        prompt = messages[-1]["content"] if messages else ""
        if "Entity:" in prompt:
            return mock_ollama._make_text_response(
                request,
                json.dumps({"facts": ["A bookshelf speaker"]}),
            )
        elif "topics" in prompt.lower():
            return mock_ollama._make_text_response(request, '{"topics": []}')
        else:
            # LLM returns both full name and fragments
            return mock_ollama._make_text_response(
                request,
                json.dumps(
                    {
                        "known": [],
                        "new": [
                            {"name": "Totem Loon"},
                            {"name": "Totem"},
                            {"name": "Loon"},
                        ],
                    }
                ),
            )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        msg_id = penny.db.log_message(
            direction="incoming",
            sender=TEST_SENDER,
            content="tell me about totem loon speakers",
        )
        penny.db.mark_messages_processed([msg_id])

        penny.db.log_search(
            query="totem loon speakers",
            response="The Totem Loon is an excellent bookshelf speaker.",
            trigger=PennyConstants.SearchTrigger.USER_MESSAGE,
        )

        await penny.extraction_pipeline.execute()

        entities = penny.db.get_user_entities(TEST_SENDER)
        entity_names = [e.name for e in entities]
        assert "totem loon" in entity_names
        assert "totem" not in entity_names, (
            f"Fragment 'totem' should be discarded, got: {entity_names}"
        )
        assert "loon" not in entity_names, (
            f"Fragment 'loon' should be discarded, got: {entity_names}"
        )
        assert len(entities) == 1


# --- Fact discovery notifications ---


@pytest.mark.asyncio
async def test_extraction_fact_notification_backoff(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Fact notifications respect exponential backoff:
    1. First cycle: notification sent (no backoff)
    2. Second cycle: notification suppressed (backoff > 0, no user reply)
    3. User sends message → backoff resets
    4. Third cycle: notification sent again
    """
    config = make_config()

    call_count = [0]

    def handler(request: dict, count: int) -> dict:
        call_count[0] += 1
        # Inspect the prompt from the chat messages
        messages = request.get("messages", [])
        prompt = messages[-1]["content"] if messages else ""
        if "Entity:" in prompt:
            # Fact extraction
            return mock_ollama._make_text_response(
                request,
                json.dumps({"facts": [f"interesting fact {call_count[0]}"]}),
            )
        elif "topics" in prompt.lower():
            # Preference extraction — return nothing
            return mock_ollama._make_text_response(request, '{"topics": []}')
        elif "discovered a new topic" in prompt or "learned some new things" in prompt:
            # Per-entity fact notification composition
            return mock_ollama._make_text_response(request, "backoff-notification-composed")
        else:
            # Entity identification — return a unique new entity each time
            n = call_count[0]
            return mock_ollama._make_text_response(
                request,
                json.dumps({"known": [], "new": [{"name": f"backoff entity {n}"}]}),
            )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Seed message for find_sender_for_timestamp and mark as processed
        msg_id = penny.db.log_message(
            direction="incoming", sender=TEST_SENDER, content="hello there friend"
        )
        penny.db.mark_messages_processed([msg_id])

        # --- Cycle 1: notification sent (no backoff) ---
        penny.db.log_search(
            query="backoff test 1",
            response="Info about backoff entity 1.",
            trigger=PennyConstants.SearchTrigger.USER_MESSAGE,
        )

        signal_server.outgoing_messages.clear()
        await penny.extraction_pipeline.execute()

        cycle1_notifications = [
            msg
            for msg in signal_server.outgoing_messages
            if "backoff-notification-composed" in msg["message"]
        ]
        assert len(cycle1_notifications) >= 1, "First cycle should send notification"

        # --- Cycle 2: notification suppressed (backoff, no user reply) ---
        penny.db.log_search(
            query="backoff test 2",
            response="Info about backoff entity 2.",
            trigger=PennyConstants.SearchTrigger.USER_MESSAGE,
        )

        signal_server.outgoing_messages.clear()
        await penny.extraction_pipeline.execute()

        cycle2_notifications = [
            msg
            for msg in signal_server.outgoing_messages
            if "backoff-notification-composed" in msg["message"]
        ]
        assert len(cycle2_notifications) == 0, "Second cycle should suppress notification (backoff)"

        # --- User sends message → resets backoff ---
        msg_id2 = penny.db.log_message(
            direction="incoming", sender=TEST_SENDER, content="thanks for the info!"
        )
        penny.db.mark_messages_processed([msg_id2])

        # --- Cycle 3: notification sent (backoff reset) ---
        penny.db.log_search(
            query="backoff test 3",
            response="Info about backoff entity 3.",
            trigger=PennyConstants.SearchTrigger.USER_MESSAGE,
        )

        signal_server.outgoing_messages.clear()
        await penny.extraction_pipeline.execute()

        cycle3_notifications = [
            msg
            for msg in signal_server.outgoing_messages
            if "backoff-notification-composed" in msg["message"]
        ]
        assert len(cycle3_notifications) >= 1, (
            "Third cycle should send notification (backoff reset by user message)"
        )


@pytest.mark.asyncio
async def test_extraction_fact_notification_new_vs_known_entity(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Per-entity discovery notifications are sent for each entity. New entities use
    the new-entity prompt, known entities use the known-entity prompt.
    """
    config = make_config()

    call_count = [0]

    def handler(request: dict, count: int) -> dict:
        call_count[0] += 1
        messages = request.get("messages", [])
        prompt = messages[-1]["content"] if messages else ""
        if "Entity:" in prompt:
            # Fact extraction — check entity name in the "Entity: <name>" line
            entity_line = [line for line in prompt.split("\n") if line.startswith("Entity:")]
            entity_name = entity_line[0].split(":", 1)[1].strip().lower() if entity_line else ""
            if "kef" in entity_name:
                return mock_ollama._make_text_response(
                    request,
                    json.dumps({"facts": ["Won What Hi-Fi 2024 award"]}),
                )
            else:
                return mock_ollama._make_text_response(
                    request,
                    json.dumps({"facts": ["Heritage cabinet design"]}),
                )
        elif "topics" in prompt.lower():
            # Preference extraction — return nothing
            return mock_ollama._make_text_response(request, '{"topics": []}')
        elif "discovered a new topic" in prompt:
            # Per-entity notification for NEW entity — echo prompt for assertion
            return mock_ollama._make_text_response(request, prompt)
        elif "learned some new things" in prompt:
            # Per-entity notification for KNOWN entity — echo prompt for assertion
            return mock_ollama._make_text_response(request, prompt)
        else:
            # Entity identification — return known + new
            return mock_ollama._make_text_response(
                request,
                json.dumps(
                    {
                        "known": ["kef ls50 meta"],
                        "new": [{"name": "Wharfedale Denton 85"}],
                    }
                ),
            )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Pre-seed known entity
        entity = penny.db.get_or_create_entity(TEST_SENDER, "kef ls50 meta")
        assert entity is not None and entity.id is not None
        penny.db.add_fact(entity.id, "Costs $1,599 per pair")

        # Seed message for find_sender_for_timestamp and mark as processed
        msg_id = penny.db.log_message(
            direction="incoming", sender=TEST_SENDER, content="compare speakers please"
        )
        penny.db.mark_messages_processed([msg_id])

        penny.db.log_search(
            query="speaker comparison",
            response="KEF LS50 Meta vs Wharfedale Denton 85 comparison.",
            trigger=PennyConstants.SearchTrigger.USER_MESSAGE,
        )

        signal_server.outgoing_messages.clear()
        await penny.extraction_pipeline.execute()

        # Should have two per-entity notifications (one new, one known)
        new_notifications = [
            msg
            for msg in signal_server.outgoing_messages
            if "discovered a new topic" in msg["message"]
        ]
        known_notifications = [
            msg
            for msg in signal_server.outgoing_messages
            if "learned some new things" in msg["message"]
        ]
        assert len(new_notifications) == 1, (
            "Should send one new-entity notification, "
            f"got messages: {[m['message'][:80] for m in signal_server.outgoing_messages]}"
        )
        assert len(known_notifications) == 1, (
            "Should send one known-entity notification, "
            f"got messages: {[m['message'][:80] for m in signal_server.outgoing_messages]}"
        )

        # New entity notification mentions the entity name
        assert "wharfedale denton 85" in new_notifications[0]["message"]
        # Known entity notification mentions the entity name
        assert "kef ls50 meta" in known_notifications[0]["message"]

"""Integration tests for the unified ExtractionPipeline agent."""

import json

import pytest
from sqlmodel import select

from penny.constants import PennyConstants
from penny.database.models import MessageLog, SearchLog
from penny.ollama.embeddings import deserialize_embedding
from penny.tests.conftest import TEST_SENDER, wait_until

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
        else:
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

        # Verify SearchLog is marked as extracted
        with penny.db.get_session() as session:
            sl = session.get(SearchLog, search_logs[0].id)
            assert sl is not None
            assert sl.extracted is True


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
async def test_extraction_extracts_preferences_from_messages(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test that ExtractionPipeline extracts preferences from regular user messages.
    """
    config = make_config()

    request_count = [0]

    def handler(request: dict, count: int) -> dict:
        request_count[0] += 1
        if request_count[0] == 1:
            return mock_ollama._make_tool_call_response(
                request, "search", {"query": "playing guitar"}
            )
        elif request_count[0] == 2:
            return mock_ollama._make_text_response(
                request, "guitar is awesome! here are some tips 🎸"
            )
        elif request_count[0] == 3:
            # Entity identification from message — no entities
            return mock_ollama._make_text_response(request, json.dumps({"known": [], "new": []}))
        elif request_count[0] == 4:
            # Likes pass: extract guitar as a like
            return mock_ollama._make_text_response(request, '{"topics": ["guitar"]}')
        else:
            # Dislikes pass: nothing
            return mock_ollama._make_text_response(request, '{"topics": []}')

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="I love playing guitar!")
        await signal_server.wait_for_message(timeout=10.0)

        # Pre-mark search logs so only message processing runs
        for sl in penny.db.get_unprocessed_search_logs(limit=100):
            penny.db.mark_search_extracted(sl.id)

        unprocessed = penny.db.get_unprocessed_messages(
            TEST_SENDER, limit=PennyConstants.PREFERENCE_BATCH_LIMIT
        )
        assert len(unprocessed) == 1

        work_done = await penny.extraction_pipeline.execute()
        assert work_done, "ExtractionPipeline should have extracted preferences"

        prefs = penny.db.get_preferences(TEST_SENDER, PennyConstants.PreferenceType.LIKE)
        assert any(p.topic == "guitar" for p in prefs), f"Expected 'guitar' in likes, got {prefs}"

        unprocessed = penny.db.get_unprocessed_messages(
            TEST_SENDER, limit=PennyConstants.PREFERENCE_BATCH_LIMIT
        )
        assert len(unprocessed) == 0, "Messages should be marked as processed"


@pytest.mark.asyncio
async def test_extraction_processes_reactions_into_preferences(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test that ExtractionPipeline processes reactions into preferences:
    1. Send a message and get a response
    2. React to the response with a like emoji
    3. Run execute() directly
    4. Verify the preference was added
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
            # Preference extraction — return cats for likes pass
            return mock_ollama._make_text_response(request, '{"topics": ["cats"]}')

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        await signal_server.push_message(
            sender=TEST_SENDER, content="tell me something cool about cats!"
        )
        response = await signal_server.wait_for_message(timeout=10.0)
        assert "cats" in response["message"].lower()

        with penny.db.get_session() as session:
            outgoing = session.exec(
                select(MessageLog).where(MessageLog.direction == "outgoing")
            ).first()
            assert outgoing is not None
            assert outgoing.external_id is not None
            external_id = outgoing.external_id

        await signal_server.push_reaction(
            sender=TEST_SENDER,
            emoji="❤️",
            target_timestamp=int(external_id),
        )

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

        work_done = await penny.extraction_pipeline.execute()
        assert work_done, "ExtractionPipeline should have processed the reaction"

        prefs = penny.db.get_preferences(TEST_SENDER, PennyConstants.PreferenceType.LIKE)
        assert len(prefs) >= 1, f"Expected at least 1 like preference, got {len(prefs)}"
        assert any(p.topic == "cats" for p in prefs)

        reactions = penny.db.get_user_reactions(
            TEST_SENDER, limit=PennyConstants.PREFERENCE_BATCH_LIMIT
        )
        assert len(reactions) == 0, "Reaction should be marked as processed"


@pytest.mark.asyncio
async def test_extraction_batches_notifications(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test that ExtractionPipeline sends a single batched message for multiple preferences.
    """
    config = make_config()

    request_count = [0]

    def handler(request: dict, count: int) -> dict:
        request_count[0] += 1
        if request_count[0] == 1:
            return mock_ollama._make_tool_call_response(request, "search", {"query": "hobbies"})
        elif request_count[0] == 2:
            return mock_ollama._make_text_response(request, "great hobbies! here are some tips 🎨")
        elif request_count[0] == 3:
            # Entity identification from message — no entities
            return mock_ollama._make_text_response(request, json.dumps({"known": [], "new": []}))
        elif request_count[0] == 4:
            # Likes pass: extract multiple topics
            return mock_ollama._make_text_response(
                request, '{"topics": ["painting", "drawing", "sculpting"]}'
            )
        else:
            # Dislikes pass: nothing
            return mock_ollama._make_text_response(request, '{"topics": []}')

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        await signal_server.push_message(
            sender=TEST_SENDER, content="I love painting, drawing, and sculpting!"
        )
        await signal_server.wait_for_message(timeout=10.0)

        # Pre-mark search logs so only message processing runs
        for sl in penny.db.get_unprocessed_search_logs(limit=100):
            penny.db.mark_search_extracted(sl.id)

        signal_server.outgoing_messages.clear()

        work_done = await penny.extraction_pipeline.execute()
        assert work_done, "ExtractionPipeline should have extracted preferences"

        await wait_until(lambda: len(signal_server.outgoing_messages) > 0)

        prefs = penny.db.get_preferences(TEST_SENDER, PennyConstants.PreferenceType.LIKE)
        assert len(prefs) == 3
        topics = {p.topic for p in prefs}
        assert topics == {"painting", "drawing", "sculpting"}

        assert len(signal_server.outgoing_messages) == 1, (
            "Should send a single batched message for all new preferences, "
            f"but sent {len(signal_server.outgoing_messages)} messages"
        )

        notification = signal_server.outgoing_messages[0]["message"]
        assert "painting" in notification
        assert "drawing" in notification
        assert "sculpting" in notification


@pytest.mark.asyncio
async def test_extraction_preference_generates_embeddings(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test that ExtractionPipeline generates embeddings for preferences
    when embedding_model is configured.
    """
    config = make_config(ollama_embedding_model="nomic-embed-text")

    request_count = [0]

    def handler(request: dict, count: int) -> dict:
        request_count[0] += 1
        if request_count[0] == 1:
            return mock_ollama._make_tool_call_response(request, "search", {"query": "guitar tips"})
        elif request_count[0] == 2:
            return mock_ollama._make_text_response(request, "guitar is great! 🎸")
        elif request_count[0] == 3:
            # Entity identification from message — no entities
            return mock_ollama._make_text_response(request, json.dumps({"known": [], "new": []}))
        elif request_count[0] == 4:
            return mock_ollama._make_text_response(request, '{"topics": ["guitar"]}')
        else:
            return mock_ollama._make_text_response(request, '{"topics": []}')

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="I love playing guitar!")
        await signal_server.wait_for_message(timeout=10.0)

        # Pre-mark search logs so only message processing runs
        for sl in penny.db.get_unprocessed_search_logs(limit=100):
            penny.db.mark_search_extracted(sl.id)

        work_done = await penny.extraction_pipeline.execute()
        assert work_done

        prefs = penny.db.get_preferences(TEST_SENDER, PennyConstants.PreferenceType.LIKE)
        assert len(prefs) >= 1
        guitar_pref = next(p for p in prefs if p.topic == "guitar")
        assert guitar_pref.embedding is not None, "Preference should have embedding"

        assert len(mock_ollama.embed_requests) >= 1
        assert len(penny.db.get_preferences_without_embeddings(limit=10)) == 0


# --- Idempotency and edge cases ---


@pytest.mark.asyncio
async def test_extraction_reaction_idempotency(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test that reactions are only processed once.
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

        reactions = penny.db.get_user_reactions(
            TEST_SENDER, limit=PennyConstants.PREFERENCE_BATCH_LIMIT
        )
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

        reactions = penny.db.get_user_reactions(
            TEST_SENDER, limit=PennyConstants.PREFERENCE_BATCH_LIMIT
        )
        assert len(reactions) == 1
        assert reactions[0].id == unprocessed_reaction_id


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


@pytest.mark.asyncio
async def test_extraction_unknown_emoji(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
    setup_ollama_flow,
):
    """
    Unknown reaction emojis are skipped but still marked processed.
    """
    config = make_config()

    setup_ollama_flow(
        search_query="test query",
        message_response="interesting fact! 🌟",
        background_response='{"topics": []}',
    )

    async with running_penny(config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="tell me a fact")
        await signal_server.wait_for_message(timeout=10.0)

        penny.db.mark_messages_processed(
            [
                m.id
                for m in penny.db.get_unprocessed_messages(
                    TEST_SENDER, limit=PennyConstants.PREFERENCE_BATCH_LIMIT
                )
                if m.id is not None
            ]
        )

        with penny.db.get_session() as session:
            outgoing = session.exec(
                select(MessageLog).where(MessageLog.direction == "outgoing")
            ).first()
            assert outgoing is not None
            outgoing_id = outgoing.id

        reaction_id = penny.db.log_message(
            direction="incoming",
            sender=TEST_SENDER,
            content="🤷",
            parent_id=outgoing_id,
            is_reaction=True,
        )
        assert reaction_id is not None

        result = await penny.extraction_pipeline.execute()
        assert result is False, "Should return False (no preference updated)"

        reactions = penny.db.get_user_reactions(
            TEST_SENDER, limit=PennyConstants.PREFERENCE_BATCH_LIMIT
        )
        assert len(reactions) == 0, "Reaction should be marked as processed even for unknown emoji"

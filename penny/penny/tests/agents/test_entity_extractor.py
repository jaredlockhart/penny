"""Integration tests for the EntityExtractor agent."""

import json

import pytest
from sqlmodel import select

from penny.database.models import EntitySearchLog, SearchLog
from penny.tests.conftest import TEST_SENDER


@pytest.mark.asyncio
async def test_entity_extractor_processes_search_log(
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
    2. Run EntityExtractor.execute() directly
    3. Pass 1: identify entities (known + new), Pass 2: extract facts per entity
    4. Verify entities and facts were stored in the database
    """
    config = make_config()

    request_count = [0]

    def handler(request: dict, count: int) -> dict:
        request_count[0] += 1
        if request_count[0] == 1:
            # Message agent: tool call
            return mock_ollama._make_tool_call_response(
                request, "search", {"query": "best hifi speakers"}
            )
        elif request_count[0] == 2:
            # Message agent: final response
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
        # Send message to create a SearchLog entry
        await signal_server.push_message(
            sender=TEST_SENDER, content="what are the best hifi speakers?"
        )
        await signal_server.wait_for_message(timeout=10.0)

        # Verify SearchLog was created
        with penny.db.get_session() as session:
            search_logs = list(session.exec(select(SearchLog)).all())
            assert len(search_logs) >= 1

        # Run EntityExtractor directly
        work_done = await penny.entity_extractor.execute()
        assert work_done, "EntityExtractor should have processed the search log"

        # Verify entity was stored
        entities = penny.db.get_user_entities(TEST_SENDER)
        assert len(entities) >= 1
        entity = next(e for e in entities if e.name == "kef ls50 meta")
        assert "Costs $1,599 per pair" in entity.facts
        assert "Metamaterial Absorption Technology" in entity.facts

        # Verify entity-search link was created
        with penny.db.get_session() as session:
            links = list(session.exec(select(EntitySearchLog)).all())
            assert len(links) >= 1
            assert any(
                link.entity_id == entity.id and link.search_log_id == search_logs[0].id
                for link in links
            )


@pytest.mark.asyncio
async def test_entity_extractor_skips_processed(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test that EntityExtractor doesn't reprocess entries:
    1. Process entries once (join table rows created)
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
            # Calls 3+: alternate pass 1 (identification) and pass 2 (facts)
            if extraction_calls[0] == 1:
                # Pass 1: identify entities
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
                # Pass 2: facts
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
        await penny.entity_extractor.execute()
        first_extraction_calls = extraction_calls[0]
        assert first_extraction_calls >= 1

        # Second extraction run — should find nothing new
        work_done = await penny.entity_extractor.execute()
        assert work_done is False, "Should return False when nothing to process"
        assert extraction_calls[0] == first_extraction_calls, (
            "Should not make additional Ollama calls"
        )


@pytest.mark.asyncio
async def test_entity_extractor_known_and_new_entities(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test mixed known + new entity extraction:
    1. Pre-seed an existing entity with facts
    2. Send a message that mentions the existing entity and a new one
    3. Pass 1 returns the existing entity as known and the new one as new
    4. Pass 2 discovers new facts for the existing entity and facts for the new one
    5. Verify existing entity got new facts appended, new entity was created with facts
    6. Verify join table links both entities to the search log
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
            # Pass 1: identify known + new entities
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
            # Pass 2: facts for each entity (called twice)
            pass2_calls[0] += 1
            if pass2_calls[0] == 1:
                # Facts for new entity (Wharfedale Linton — processed first as new)
                return mock_ollama._make_text_response(
                    request,
                    json.dumps(
                        {"facts": ["Heritage design with modern drivers", "Costs $1,199 per pair"]}
                    ),
                )
            else:
                # New facts for known entity (KEF LS50 Meta)
                return mock_ollama._make_text_response(
                    request,
                    json.dumps({"facts": ["Won What Hi-Fi 2024 award"]}),
                )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Pre-seed an existing entity with a fact
        entity = penny.db.get_or_create_entity(TEST_SENDER, "kef ls50 meta")
        assert entity is not None and entity.id is not None
        penny.db.update_entity_facts(entity.id, "- Costs $1,599 per pair")

        # Send message to create a SearchLog entry
        await signal_server.push_message(
            sender=TEST_SENDER, content="compare KEF LS50 Meta vs Wharfedale Linton"
        )
        await signal_server.wait_for_message(timeout=10.0)

        # Run extraction
        work_done = await penny.entity_extractor.execute()
        assert work_done

        entities = penny.db.get_user_entities(TEST_SENDER)
        assert len(entities) == 2

        # Existing entity should have original fact + new fact
        kef = next(e for e in entities if e.name == "kef ls50 meta")
        assert "Costs $1,599 per pair" in kef.facts
        assert "Won What Hi-Fi 2024 award" in kef.facts

        # New entity should have its facts
        wharfedale = next(e for e in entities if e.name == "wharfedale linton")
        assert "Heritage design with modern drivers" in wharfedale.facts
        assert "Costs $1,199 per pair" in wharfedale.facts

        # Verify join table has links for both entities
        with penny.db.get_session() as session:
            links = list(session.exec(select(EntitySearchLog)).all())
            entity_ids = {link.entity_id for link in links}
            assert kef.id in entity_ids
            assert wharfedale.id in entity_ids


@pytest.mark.asyncio
async def test_entity_extractor_empty_extraction(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test that EntityExtractor handles empty extraction results gracefully
    and still marks the search as processed via a sentinel join table row.
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
            # Pass 1: no entities found
            return mock_ollama._make_text_response(request, json.dumps({"known": [], "new": []}))

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="hello!")
        await signal_server.wait_for_message(timeout=10.0)

        work_done = await penny.entity_extractor.execute()
        assert work_done is False, "No entities extracted means no work done"

        # Sentinel row should exist so we don't reprocess
        with penny.db.get_session() as session:
            links = list(session.exec(select(EntitySearchLog)).all())
            assert len(links) >= 1
            assert any(link.entity_id is None for link in links)

        # No entities stored
        entities = penny.db.get_user_entities(TEST_SENDER)
        assert len(entities) == 0

"""Integration tests for the EntityExtractor agent."""

import json

import pytest
from sqlmodel import select

from penny.database.models import SearchLog
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
    Test that EntityExtractor processes SearchLog entries and stores entities:
    1. Send a message (triggers search, creates SearchLog entry)
    2. Run EntityExtractor.execute() directly
    3. Verify entities and facts were stored in the database
    4. Verify the extraction cursor was advanced
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
        else:
            # Entity extractor: structured output
            return mock_ollama._make_text_response(
                request,
                json.dumps(
                    {
                        "entities": [
                            {
                                "name": "KEF LS50 Meta",
                                "entity_type": "product",
                                "facts": [
                                    "Costs $1,599 per pair",
                                    "Features Metamaterial Absorption Technology",
                                ],
                            }
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
        assert entity.entity_type == "product"
        assert "Costs $1,599 per pair" in entity.facts
        assert "Metamaterial Absorption Technology" in entity.facts

        # Verify cursor was advanced
        cursor = penny.db.get_extraction_cursor("search")
        assert cursor > 0


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
    1. Process entries once (cursor advances)
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
            return mock_ollama._make_text_response(
                request,
                json.dumps(
                    {
                        "entities": [
                            {
                                "name": "NVIDIA Jetson",
                                "entity_type": "product",
                                "facts": ["Edge AI computing module"],
                            }
                        ]
                    }
                ),
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
    and still advances the cursor.
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
            # Empty extraction
            return mock_ollama._make_text_response(request, '{"entities": []}')

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="hello!")
        await signal_server.wait_for_message(timeout=10.0)

        work_done = await penny.entity_extractor.execute()
        assert work_done is False, "No entities extracted means no work done"

        # Cursor should still advance so we don't reprocess
        cursor = penny.db.get_extraction_cursor("search")
        assert cursor > 0

        # No entities stored
        entities = penny.db.get_user_entities(TEST_SENDER)
        assert len(entities) == 0

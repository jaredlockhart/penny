"""Integration tests for the EntityCleaner agent."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlmodel import select

from penny.constants import PennyConstants
from penny.database.models import EntitySearchLog
from penny.tests.conftest import TEST_SENDER


@pytest.mark.asyncio
async def test_entity_cleaner_merges_duplicates(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test that EntityCleaner merges duplicate entities:
    1. Pre-seed three entities (two duplicates + one distinct)
    2. Create entity_search_log links for all three
    3. Mock LLM returns a merge group for the duplicate pair
    4. Verify: entities merged, facts combined, search log refs updated, duplicate deleted
    """
    config = make_config()

    request_count = [0]

    def handler(request: dict, count: int) -> dict:
        request_count[0] += 1
        if request_count[0] == 1:
            return mock_ollama._make_tool_call_response(
                request, "search", {"query": "stanford physics"}
            )
        elif request_count[0] == 2:
            return mock_ollama._make_text_response(request, "here's what I found 🔬")
        else:
            # Entity cleaner merge identification
            return mock_ollama._make_text_response(
                request,
                json.dumps(
                    {
                        "groups": [
                            {
                                "canonical_name": "stanford",
                                "duplicates": ["stanford university"],
                            }
                        ]
                    }
                ),
            )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Send a message to create a SearchLog entry
        await signal_server.push_message(
            sender=TEST_SENDER, content="tell me about stanford physics"
        )
        await signal_server.wait_for_message(timeout=10.0)

        # Pre-seed three entities
        stanford = penny.db.get_or_create_entity(TEST_SENDER, "stanford")
        assert stanford is not None and stanford.id is not None
        penny.db.update_entity_facts(stanford.id, "- Located in California")

        stanford_uni = penny.db.get_or_create_entity(TEST_SENDER, "stanford university")
        assert stanford_uni is not None and stanford_uni.id is not None
        penny.db.update_entity_facts(stanford_uni.id, "- Founded in 1885\n- Located in California")

        kef = penny.db.get_or_create_entity(TEST_SENDER, "kef ls50 meta")
        assert kef is not None and kef.id is not None
        penny.db.update_entity_facts(kef.id, "- Costs $1,599 per pair")

        # Create entity_search_log links
        with penny.db.get_session() as session:
            search_log = session.exec(select(EntitySearchLog)).first()
            search_log_id = search_log.search_log_id if search_log else 1

        penny.db.link_entity_to_search_log(stanford.id, search_log_id)
        penny.db.link_entity_to_search_log(stanford_uni.id, search_log_id)

        # Run the entity cleaner
        work_done = await penny.entity_cleaner.execute()
        assert work_done, "EntityCleaner should have merged duplicates"

        # Verify merge result: 3 entities → 2 (duplicate deleted)
        entities = penny.db.get_user_entities(TEST_SENDER)
        assert len(entities) == 2
        entity_names = {e.name for e in entities}
        assert "stanford" in entity_names, "Canonical entity should survive"
        assert "stanford university" not in entity_names, "Duplicate should be deleted"
        assert "kef ls50 meta" in entity_names, "Unrelated entity should be untouched"

        # Verify facts were combined and deduplicated
        merged = next(e for e in entities if e.name == "stanford")
        assert "Located in California" in merged.facts
        assert "Founded in 1885" in merged.facts
        # "Located in California" was on both entities — should appear only once after dedup
        assert merged.facts.count("Located in California") == 1

        # Verify unrelated entity is unchanged
        kef_after = next(e for e in entities if e.name == "kef ls50 meta")
        assert kef_after.facts == "- Costs $1,599 per pair"

        # Verify entity_search_log references were reassigned from duplicate to primary
        with penny.db.get_session() as session:
            links = list(
                session.exec(
                    select(EntitySearchLog).where(EntitySearchLog.entity_id == stanford.id)
                ).all()
            )
            # Original link + reassigned link from duplicate
            assert len(links) == 2

            # No orphaned references to the deleted entity
            orphans = list(
                session.exec(
                    select(EntitySearchLog).where(EntitySearchLog.entity_id == stanford_uni.id)
                ).all()
            )
            assert len(orphans) == 0

        # Verify cleaning timestamp was stored
        ts = penny.db.get_entity_cleaning_timestamp()
        assert ts is not None


@pytest.mark.asyncio
async def test_entity_cleaner_skips_when_recent(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test that EntityCleaner skips cleaning when last run was recent:
    1. Pre-seed a recent cleaning timestamp
    2. Run execute()
    3. Verify it returns False without making LLM calls
    """
    config = make_config()

    ollama_calls = [0]

    def handler(request: dict, count: int) -> dict:
        ollama_calls[0] += 1
        if ollama_calls[0] == 1:
            return mock_ollama._make_tool_call_response(request, "search", {"query": "test"})
        return mock_ollama._make_text_response(request, "test 🧪")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Send a message so we have a valid sender
        await signal_server.push_message(sender=TEST_SENDER, content="test")
        await signal_server.wait_for_message(timeout=10.0)

        # Set a recent cleaning timestamp
        penny.db.set_entity_cleaning_timestamp(datetime.now(UTC))
        calls_before = ollama_calls[0]

        # Run cleaner — should skip due to recent timestamp
        work_done = await penny.entity_cleaner.execute()
        assert work_done is False, "Should skip when recently cleaned"
        assert ollama_calls[0] == calls_before, "Should not make any LLM calls when skipping"


@pytest.mark.asyncio
async def test_entity_cleaner_no_duplicates(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test that EntityCleaner handles no duplicates gracefully:
    1. Pre-seed distinct entities
    2. Mock LLM returns empty merge groups
    3. Verify no entities were deleted, timestamp was still updated
    """
    config = make_config()

    request_count = [0]

    def handler(request: dict, count: int) -> dict:
        request_count[0] += 1
        if request_count[0] == 1:
            return mock_ollama._make_tool_call_response(request, "search", {"query": "speakers"})
        elif request_count[0] == 2:
            return mock_ollama._make_text_response(request, "nice speakers! 🎵")
        else:
            # Entity cleaner: no duplicates found
            return mock_ollama._make_text_response(request, json.dumps({"groups": []}))

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="tell me about speakers")
        await signal_server.wait_for_message(timeout=10.0)

        # Pre-seed distinct entities
        penny.db.get_or_create_entity(TEST_SENDER, "kef ls50 meta")
        penny.db.get_or_create_entity(TEST_SENDER, "wharfedale linton")

        # Set an old cleaning timestamp so it actually runs
        old_ts = datetime.now(UTC) - timedelta(
            seconds=PennyConstants.ENTITY_CLEANING_INTERVAL_SECONDS + 1
        )
        penny.db.set_entity_cleaning_timestamp(old_ts)

        work_done = await penny.entity_cleaner.execute()
        assert work_done is False, "No merges means no work done"

        # All entities should still exist
        entities = penny.db.get_user_entities(TEST_SENDER)
        assert len(entities) == 2

        # Timestamp should have been updated
        ts = penny.db.get_entity_cleaning_timestamp()
        assert ts is not None
        assert ts > old_ts

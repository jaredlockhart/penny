"""Integration tests for /unlearn command."""

import pytest

from penny.constants import PennyConstants
from penny.tests.conftest import TEST_SENDER


@pytest.mark.asyncio
async def test_unlearn_no_args_lists_topics(signal_server, test_config, mock_ollama, running_penny):
    """Test /unlearn with no args lists all learn topics oldest first."""
    async with running_penny(test_config) as penny:
        lp1 = penny.db.create_learn_prompt(
            user=TEST_SENDER, prompt_text="kef speakers", searches_remaining=0
        )
        assert lp1 is not None and lp1.id is not None
        penny.db.update_learn_prompt_status(lp1.id, PennyConstants.LearnPromptStatus.COMPLETED)

        lp2 = penny.db.create_learn_prompt(
            user=TEST_SENDER, prompt_text="travel in japan", searches_remaining=3
        )
        assert lp2 is not None

        await signal_server.push_message(sender=TEST_SENDER, content="/unlearn")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "Learn History" in response["message"]
        # Oldest first
        assert "1. kef speakers" in response["message"]
        assert "2. travel in japan" in response["message"]
        # Active topics show status
        assert "(active)" in response["message"]


@pytest.mark.asyncio
async def test_unlearn_empty(signal_server, test_config, mock_ollama, running_penny):
    """Test /unlearn with no learn history shows empty message."""
    async with running_penny(test_config) as _penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/unlearn")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "No learn history" in response["message"]


@pytest.mark.asyncio
async def test_unlearn_deletes_entities_and_facts(
    signal_server, test_config, mock_ollama, running_penny
):
    """Test /unlearn <number> deletes learn prompt, entities, and facts."""
    async with running_penny(test_config) as penny:
        # Set up a learn prompt with linked search → entity → fact chain
        lp = penny.db.create_learn_prompt(
            user=TEST_SENDER, prompt_text="kef speakers", searches_remaining=0
        )
        assert lp is not None and lp.id is not None
        penny.db.update_learn_prompt_status(lp.id, PennyConstants.LearnPromptStatus.COMPLETED)

        penny.db.log_search(
            query="kef speakers overview",
            response="KEF makes great speakers...",
            trigger=PennyConstants.SearchTrigger.LEARN_COMMAND,
            learn_prompt_id=lp.id,
        )

        entity = penny.db.get_or_create_entity(TEST_SENDER, "kef ls50 meta")
        assert entity is not None and entity.id is not None

        search_logs = penny.db.get_search_logs_by_learn_prompt(lp.id)
        sl_id = search_logs[0].id

        penny.db.add_fact(entity.id, "Costs $1,599 per pair", source_search_log_id=sl_id)
        penny.db.add_fact(entity.id, "Won What Hi-Fi award", source_search_log_id=sl_id)

        await signal_server.push_message(sender=TEST_SENDER, content="/unlearn 1")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "Forgetting what I learned about" in response["message"]
        assert "kef speakers" in response["message"]
        assert "kef ls50 meta" in response["message"]
        assert "2 facts" in response["message"]

        # Verify data is deleted
        assert penny.db.get_learn_prompt(lp.id) is None
        assert penny.db.get_entity_facts(entity.id) == []
        assert penny.db.get_search_logs_by_learn_prompt(lp.id) == []


@pytest.mark.asyncio
async def test_unlearn_invalid_number(signal_server, test_config, mock_ollama, running_penny):
    """Test /unlearn with out-of-range number shows error."""
    async with running_penny(test_config) as penny:
        penny.db.create_learn_prompt(
            user=TEST_SENDER, prompt_text="kef speakers", searches_remaining=0
        )

        await signal_server.push_message(sender=TEST_SENDER, content="/unlearn 99")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "doesn't match any topic" in response["message"]


@pytest.mark.asyncio
async def test_unlearn_no_entities_discovered(
    signal_server, test_config, mock_ollama, running_penny
):
    """Test /unlearn for a topic that yielded no entities."""
    async with running_penny(test_config) as penny:
        lp = penny.db.create_learn_prompt(
            user=TEST_SENDER, prompt_text="obscure topic", searches_remaining=0
        )
        assert lp is not None and lp.id is not None
        penny.db.update_learn_prompt_status(lp.id, PennyConstants.LearnPromptStatus.COMPLETED)

        await signal_server.push_message(sender=TEST_SENDER, content="/unlearn 1")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "Forgetting what I learned about" in response["message"]
        assert "obscure topic" in response["message"]
        assert "No entities were discovered" in response["message"]

        # Learn prompt should still be deleted
        assert penny.db.get_learn_prompt(lp.id) is None

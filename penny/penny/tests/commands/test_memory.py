"""Integration tests for /memory command."""

import pytest

from penny.tests.conftest import TEST_SENDER


@pytest.mark.asyncio
async def test_memory_list_empty(signal_server, test_config, mock_ollama, running_penny):
    """Test /memory with no stored entities."""
    async with running_penny(test_config) as _penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/memory")
        response = await signal_server.wait_for_message(timeout=5.0)
        assert "You don't have any stored memories yet" in response["message"]


@pytest.mark.asyncio
async def test_memory_list_and_show(signal_server, test_config, mock_ollama, running_penny):
    """Test /memory lists entities and /memory <number> shows details."""
    async with running_penny(test_config) as penny:
        # Seed entities (ordered by updated_at DESC, so last-updated appears first)
        entity1 = penny.db.get_or_create_entity(TEST_SENDER, "nvidia jetson")
        penny.db.update_entity_facts(entity1.id, "- Edge AI compute module")

        entity2 = penny.db.get_or_create_entity(TEST_SENDER, "kef ls50 meta")
        penny.db.update_entity_facts(entity2.id, "- Costs $1,599 per pair\n- Uses MAT driver")

        # List entities (most recently updated first)
        await signal_server.push_message(sender=TEST_SENDER, content="/memory")
        response = await signal_server.wait_for_message(timeout=5.0)
        assert "Here's what I remember:" in response["message"]
        assert "kef ls50 meta (2 facts)" in response["message"]
        assert "nvidia jetson (1 fact)" in response["message"]

        # Show entity details — #1 is kef (most recently updated)
        await signal_server.push_message(sender=TEST_SENDER, content="/memory 1")
        response2 = await signal_server.wait_for_message(timeout=5.0)
        assert "kef ls50 meta" in response2["message"]
        assert "Costs $1,599 per pair" in response2["message"]
        assert "Uses MAT driver" in response2["message"]


@pytest.mark.asyncio
async def test_memory_show_not_found(signal_server, test_config, mock_ollama, running_penny):
    """Test /memory <number> with invalid number."""
    async with running_penny(test_config) as _penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/memory 99")
        response = await signal_server.wait_for_message(timeout=5.0)
        assert "doesn't match any memory" in response["message"]


@pytest.mark.asyncio
async def test_memory_delete(signal_server, test_config, mock_ollama, running_penny):
    """Test /memory <number> delete removes entity."""
    async with running_penny(test_config) as penny:
        # Seed entity
        entity = penny.db.get_or_create_entity(TEST_SENDER, "wharfedale linton")
        penny.db.update_entity_facts(entity.id, "- Classic heritage speaker\n- 3-way design")

        # Delete it
        await signal_server.push_message(sender=TEST_SENDER, content="/memory 1 delete")
        response = await signal_server.wait_for_message(timeout=5.0)
        assert "Deleted 'wharfedale linton'" in response["message"]
        assert "2 fact(s)" in response["message"]

        # Verify it's gone
        await signal_server.push_message(sender=TEST_SENDER, content="/memory")
        response2 = await signal_server.wait_for_message(timeout=5.0)
        assert "You don't have any stored memories yet" in response2["message"]

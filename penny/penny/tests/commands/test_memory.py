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
async def test_memory_list_ordered_by_recency(
    signal_server, test_config, mock_ollama, running_penny
):
    """Test /memory lists entities ordered by most recently created."""
    async with running_penny(test_config) as penny:
        # Create entities — entity2 created second, so it appears first
        entity1 = penny.db.entities.get_or_create(TEST_SENDER, "nvidia jetson")
        penny.db.facts.add(entity1.id, "Edge AI compute module")

        entity2 = penny.db.entities.get_or_create(TEST_SENDER, "kef ls50 meta")
        penny.db.facts.add(entity2.id, "Costs $1,599 per pair")
        penny.db.facts.add(entity2.id, "Uses MAT driver")

        await signal_server.push_message(sender=TEST_SENDER, content="/memory")
        response = await signal_server.wait_for_message(timeout=5.0)

        msg = response["message"]
        assert "Your Memory" in msg
        # Entity2 created more recently, should be listed first
        assert "1. **kef ls50 meta** (2 facts)" in msg
        assert "2. **nvidia jetson** (1 fact)" in msg

        # Show entity details — #1 is kef (most recent)
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

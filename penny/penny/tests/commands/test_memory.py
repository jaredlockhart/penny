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
async def test_memory_list_ranked_by_heat(signal_server, test_config, mock_ollama, running_penny):
    """Test /memory lists entities ranked by heat with scores displayed."""
    async with running_penny(test_config) as penny:
        # Create entities with different heat levels
        entity1 = penny.db.entities.get_or_create(TEST_SENDER, "nvidia jetson")
        penny.db.facts.add(entity1.id, "Edge AI compute module")
        penny.db.entities.update_heat(entity1.id, 1.50)

        entity2 = penny.db.entities.get_or_create(TEST_SENDER, "kef ls50 meta")
        penny.db.facts.add(entity2.id, "Costs $1,599 per pair")
        penny.db.facts.add(entity2.id, "Uses MAT driver")
        penny.db.entities.update_heat(entity2.id, 5.00)

        await signal_server.push_message(sender=TEST_SENDER, content="/memory")
        response = await signal_server.wait_for_message(timeout=5.0)

        msg = response["message"]
        assert "Your Memory" in msg
        # Entity2 should be ranked above entity1 (higher heat)
        assert "1. **kef ls50 meta** (2 facts, heat: 5.00)" in msg
        assert "2. **nvidia jetson** (1 fact, heat: 1.50)" in msg

        # Show entity details — #1 is kef (highest heat)
        await signal_server.push_message(sender=TEST_SENDER, content="/memory 1")
        response2 = await signal_server.wait_for_message(timeout=5.0)
        assert "kef ls50 meta" in response2["message"]
        assert "Costs $1,599 per pair" in response2["message"]
        assert "Uses MAT driver" in response2["message"]


@pytest.mark.asyncio
async def test_memory_shows_zero_heat(signal_server, test_config, mock_ollama, running_penny):
    """Test /memory shows entities with zero heat (e.g. vetoed)."""
    async with running_penny(test_config) as penny:
        entity = penny.db.entities.get_or_create(TEST_SENDER, "sports")
        penny.db.facts.add(entity.id, "Various athletic activities")
        # Heat defaults to 0.0 (e.g. after a veto)

        await signal_server.push_message(sender=TEST_SENDER, content="/memory")
        response = await signal_server.wait_for_message(timeout=5.0)

        msg = response["message"]
        assert "sports" in msg
        assert "heat: 0.00" in msg


@pytest.mark.asyncio
async def test_memory_show_not_found(signal_server, test_config, mock_ollama, running_penny):
    """Test /memory <number> with invalid number."""
    async with running_penny(test_config) as _penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/memory 99")
        response = await signal_server.wait_for_message(timeout=5.0)
        assert "doesn't match any memory" in response["message"]

"""Integration tests for /forget command."""

import pytest

from penny.tests.conftest import TEST_SENDER


@pytest.mark.asyncio
async def test_forget_deletes_entity(signal_server, test_config, mock_ollama, running_penny):
    """Test /forget <number> removes entity and its facts."""
    async with running_penny(test_config) as penny:
        entity = penny.db.entities.get_or_create(TEST_SENDER, "wharfedale linton")
        penny.db.facts.add(entity.id, "Classic heritage speaker")
        penny.db.facts.add(entity.id, "3-way design")

        await signal_server.push_message(sender=TEST_SENDER, content="/forget 1")
        response = await signal_server.wait_for_message(timeout=5.0)
        assert "Deleted 'wharfedale linton'" in response["message"]
        assert "2 fact(s)" in response["message"]

        # Verify it's gone
        await signal_server.push_message(sender=TEST_SENDER, content="/memory")
        response2 = await signal_server.wait_for_message(timeout=5.0)
        assert "You don't have any stored memories yet" in response2["message"]


@pytest.mark.asyncio
async def test_forget_no_args(signal_server, test_config, mock_ollama, running_penny):
    """Test /forget with no args returns usage hint."""
    async with running_penny(test_config) as _penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/forget")
        response = await signal_server.wait_for_message(timeout=5.0)
        assert "/forget" in response["message"]
        assert "/memory" in response["message"]


@pytest.mark.asyncio
async def test_forget_invalid_number(signal_server, test_config, mock_ollama, running_penny):
    """Test /forget with an out-of-range number returns not-found message."""
    async with running_penny(test_config) as _penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/forget 99")
        response = await signal_server.wait_for_message(timeout=5.0)
        assert "doesn't match any memory" in response["message"]

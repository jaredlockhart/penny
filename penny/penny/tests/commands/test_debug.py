"""Integration tests for /debug command."""

import os

import pytest

from penny.tests.conftest import TEST_SENDER


@pytest.mark.asyncio
async def test_debug_command(signal_server, test_config, mock_ollama, running_penny):
    """Test /debug returns diagnostic information."""
    async with running_penny(test_config) as _penny:
        # Send /debug
        await signal_server.push_message(sender=TEST_SENDER, content="/debug")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should show debug info
        assert "**Debug Information**" in response["message"]
        assert "**Git Commit**:" in response["message"]
        assert "**Uptime**:" in response["message"]
        assert "**Channel**: Signal" in response["message"]
        assert "**Database**:" in response["message"]
        assert "messages" in response["message"]
        assert "**Models**:" in response["message"]
        assert "**Background Tasks**:" in response["message"]
        assert "**Memory**:" in response["message"]

        # Should show actual scheduler status, not "Unknown (no scheduler)"
        assert "Unknown (no scheduler)" not in response["message"]
        # Should show at least one agent name from the scheduler
        assert any(agent in response["message"] for agent in ["extraction", "learn"])


@pytest.mark.asyncio
async def test_debug_git_commit_from_env(signal_server, test_config, mock_ollama, running_penny):
    """Test /debug shows git commit from environment variable."""
    # Set GIT_COMMIT environment variable
    old_commit = os.environ.get("GIT_COMMIT")
    os.environ["GIT_COMMIT"] = "abc1234"

    try:
        async with running_penny(test_config) as _penny:
            # Send /debug
            await signal_server.push_message(sender=TEST_SENDER, content="/debug")

            # Wait for response
            response = await signal_server.wait_for_message(timeout=5.0)

            # Should show git commit from environment
            assert "**Git Commit**: abc1234" in response["message"]
    finally:
        # Restore original environment
        if old_commit is None:
            os.environ.pop("GIT_COMMIT", None)
        else:
            os.environ["GIT_COMMIT"] = old_commit

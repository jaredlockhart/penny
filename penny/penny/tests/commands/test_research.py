"""Integration tests for /research command."""

import pytest
from sqlmodel import select

from penny.database.models import ResearchTask
from penny.tests.conftest import TEST_SENDER


@pytest.mark.asyncio
async def test_research_command_creates_task(
    signal_server, test_config, mock_ollama, test_user_info, running_penny
):
    """Test /research creates a research task in the database."""
    async with running_penny(test_config) as penny:
        # Send /research command
        await signal_server.push_message(sender=TEST_SENDER, content="/research quantum computing")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should confirm research started
        assert "started research" in response["message"].lower()
        assert "quantum computing" in response["message"]

        # Verify task was created in database
        with penny.db.get_session() as session:
            tasks = list(session.exec(select(ResearchTask)).all())
            assert len(tasks) == 1
            task = tasks[0]
            assert task.topic == "quantum computing"
            assert task.status == "in_progress"
            assert task.thread_id == TEST_SENDER
            assert task.max_iterations == test_config.research_max_iterations


@pytest.mark.asyncio
async def test_research_command_lists_when_no_topic(
    signal_server, test_config, mock_ollama, test_user_info, running_penny
):
    """Test /research without a topic lists active research tasks."""
    async with running_penny(test_config) as penny:
        # Send /research without topic when no tasks exist
        await signal_server.push_message(sender=TEST_SENDER, content="/research")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should report no active tasks
        assert "no active research" in response["message"].lower()

        # Should not create a task
        with penny.db.get_session() as session:
            tasks = list(session.exec(select(ResearchTask)).all())
            assert len(tasks) == 0


@pytest.mark.asyncio
async def test_research_command_lists_active_tasks(
    signal_server, test_config, mock_ollama, test_user_info, running_penny
):
    """Test /research without topic lists active research tasks."""
    async with running_penny(test_config):
        # Start a research task
        await signal_server.push_message(sender=TEST_SENDER, content="/research AI trends")
        await signal_server.wait_for_message(timeout=5.0)

        # List tasks with /research (no topic)
        await signal_server.push_message(sender=TEST_SENDER, content="/research")
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should list the active task with progress
        assert "currently researching" in response["message"].lower()
        assert "ai trends" in response["message"].lower()
        # Should show progress indicator (*Not Started* since no iterations yet)
        assert "*not started*" in response["message"].lower()


@pytest.mark.asyncio
async def test_research_command_rejects_duplicate(
    signal_server, test_config, mock_ollama, test_user_info, running_penny
):
    """Test /research rejects duplicate in-progress research in same thread."""
    async with running_penny(test_config) as penny:
        # Send first research request
        await signal_server.push_message(sender=TEST_SENDER, content="/research AI trends")
        await signal_server.wait_for_message(timeout=5.0)

        # Send second research request while first is in-progress
        await signal_server.push_message(sender=TEST_SENDER, content="/research quantum computing")
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should reject with message about existing research
        assert "already researching" in response["message"].lower()
        assert "ai trends" in response["message"].lower()

        # Should only have one task in database
        with penny.db.get_session() as session:
            tasks = list(session.exec(select(ResearchTask)).all())
            assert len(tasks) == 1
            assert tasks[0].topic == "AI trends"


@pytest.mark.asyncio
async def test_research_command_respects_config(
    signal_server, make_config, mock_ollama, test_user_info, running_penny
):
    """Test /research respects research_max_iterations config."""
    # Create config with custom iteration count
    config = make_config(research_max_iterations=5)

    async with running_penny(config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/research coffee")
        await signal_server.wait_for_message(timeout=5.0)

        # Verify task has custom max_iterations
        with penny.db.get_session() as session:
            tasks = list(session.exec(select(ResearchTask)).all())
            assert len(tasks) == 1
            assert tasks[0].max_iterations == 5

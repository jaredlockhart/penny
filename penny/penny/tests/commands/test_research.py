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
        # Should show progress indicator (*Not Started* or N/10 depending on scheduler timing)
        msg_lower = response["message"].lower()
        assert "*not started*" in msg_lower or "/10" in msg_lower


@pytest.mark.asyncio
async def test_research_command_queues_when_active(
    signal_server, test_config, mock_ollama, test_user_info, running_penny
):
    """Test /research queues new research when one is already in-progress."""
    async with running_penny(test_config) as penny:
        # Send first research request
        await signal_server.push_message(sender=TEST_SENDER, content="/research AI trends")
        await signal_server.wait_for_message(timeout=5.0)

        # Send second research request while first is in-progress
        await signal_server.push_message(sender=TEST_SENDER, content="/research quantum computing")
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should confirm queueing
        assert "queued" in response["message"].lower()
        assert "quantum computing" in response["message"].lower()
        assert "ai trends" in response["message"].lower()

        # Should have two tasks in database: one in_progress, one pending
        with penny.db.get_session() as session:
            tasks = list(
                session.exec(select(ResearchTask).order_by(ResearchTask.created_at.asc())).all()  # type: ignore[unresolved-attribute]
            )
            assert len(tasks) == 2
            assert tasks[0].topic == "AI trends"
            assert tasks[0].status == "in_progress"
            assert tasks[1].topic == "quantum computing"
            assert tasks[1].status == "pending"


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


@pytest.mark.asyncio
async def test_research_command_lists_pending_tasks(
    signal_server, test_config, mock_ollama, test_user_info, running_penny
):
    """Test /research without topic lists both active and pending tasks."""
    async with running_penny(test_config) as penny:
        # Start first research task
        await signal_server.push_message(sender=TEST_SENDER, content="/research AI trends")
        await signal_server.wait_for_message(timeout=5.0)

        # Queue second research task
        await signal_server.push_message(sender=TEST_SENDER, content="/research quantum computing")
        await signal_server.wait_for_message(timeout=5.0)

        # List all tasks
        await signal_server.push_message(sender=TEST_SENDER, content="/research")
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should list both tasks
        assert "currently researching" in response["message"].lower()
        assert "ai trends" in response["message"].lower()
        assert "quantum computing" in response["message"].lower()
        # First task should show progress or "Not Started"
        # Second task should show "Queued"
        assert "*queued*" in response["message"].lower()

        # Verify both tasks exist with correct statuses
        with penny.db.get_session() as session:
            tasks = list(
                session.exec(select(ResearchTask).order_by(ResearchTask.created_at.asc())).all()  # type: ignore[unresolved-attribute]
            )
            assert len(tasks) == 2
            assert tasks[0].status == "in_progress"
            assert tasks[1].status == "pending"


@pytest.mark.asyncio
async def test_research_list_uses_bullet_format(
    signal_server, test_config, mock_ollama, test_user_info, running_penny
):
    """Test /research list output uses proper markdown bullet format."""
    async with running_penny(test_config):
        # Start two research tasks
        await signal_server.push_message(sender=TEST_SENDER, content="/research AI trends")
        await signal_server.wait_for_message(timeout=5.0)
        await signal_server.push_message(sender=TEST_SENDER, content="/research quantum computing")
        await signal_server.wait_for_message(timeout=5.0)

        # List all tasks
        await signal_server.push_message(sender=TEST_SENDER, content="/research")
        response = await signal_server.wait_for_message(timeout=5.0)

        # Verify markdown bullet format: header should be followed immediately by bullets
        # No blank line between header and first bullet
        msg = response["message"]
        assert "**Currently researching:**\n*" in msg or "**Currently researching:**\r\n*" in msg
        # Each task should be a bullet point
        assert msg.count("* ") >= 2  # At least 2 bullet items

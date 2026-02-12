"""Integration tests for ResearchAgent."""

import pytest
from sqlmodel import select

from penny.database.models import MessageLog, ResearchIteration, ResearchTask
from penny.tests.conftest import TEST_SENDER, wait_until


@pytest.mark.asyncio
async def test_research_agent_executes_iterations(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test ResearchAgent executes multiple search iterations and posts report.

    1. Trigger /research command
    2. Wait for agent to run iterations
    3. Verify report is posted when max_iterations is reached
    """
    # Set low iteration count to keep test fast
    config = make_config(
        research_max_iterations=3,
        idle_seconds=0.3,
        scheduler_tick_interval=0.05,
    )

    # Configure mock to return different responses for each iteration
    iteration_responses = [
        "Iteration 1 findings: quantum computers use qubits",
        "Iteration 2 findings: quantum entanglement enables parallel computation",
        "Iteration 3 findings: major applications in cryptography and drug discovery",
    ]
    response_index = [0]

    def research_handler(request: dict, count: int) -> dict:
        # Each research iteration gets a simple text response
        if response_index[0] < len(iteration_responses):
            response = iteration_responses[response_index[0]]
            response_index[0] += 1
            return mock_ollama._make_text_response(request, response)
        return mock_ollama._make_text_response(request, "additional iteration")

    mock_ollama.set_response_handler(research_handler)

    async with running_penny(config) as penny:
        # Start research
        await signal_server.push_message(sender=TEST_SENDER, content="/research quantum computing")
        confirmation = await signal_server.wait_for_message(timeout=5.0)
        assert "started research" in confirmation["message"].lower()

        # Wait for research report to be posted
        # Report comes after the confirmation, so we need >1 messages
        # Research runs every 5 seconds, so need to wait for 4 iterations (3 research + 1 report)
        await wait_until(lambda: len(signal_server.outgoing_messages) >= 2, timeout=25.0)

        # Last message should be the research report
        report = signal_server.outgoing_messages[-1]
        assert "research complete" in report["message"].lower()
        assert "quantum computing" in report["message"].lower()
        assert "summary" in report["message"].lower()
        assert "key findings" in report["message"].lower()
        assert "sources" in report["message"].lower()

        # Verify task is marked complete in database
        with penny.db.get_session() as session:
            tasks = list(session.exec(select(ResearchTask)).all())
            assert len(tasks) == 1
            task = tasks[0]
            assert task.status == "completed"
            assert task.completed_at is not None
            assert task.message_id is not None
            # Verify message_id is a valid integer string (not "True"/"False")
            assert task.message_id.isdigit(), f"Expected integer string, got: {task.message_id}"

            # Verify 3 iterations were stored
            iterations = list(
                session.exec(
                    select(ResearchIteration).where(ResearchIteration.research_task_id == task.id)
                ).all()
            )
            assert len(iterations) == 3
            assert iterations[0].iteration_num == 1
            assert iterations[1].iteration_num == 2
            assert iterations[2].iteration_num == 3


@pytest.mark.asyncio
async def test_research_agent_no_channel(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Test ResearchAgent returns False when no channel is set."""
    config = make_config()

    async with running_penny(config) as penny:
        from penny.agents import ResearchAgent
        from penny.constants import SYSTEM_PROMPT

        research_agent = ResearchAgent(
            config=config,
            system_prompt=SYSTEM_PROMPT,
            model=config.ollama_foreground_model,
            ollama_api_url=config.ollama_api_url,
            tools=[],
            db=penny.db,
        )
        # Don't set channel

        result = await research_agent.execute()
        assert result is False, "Should return False when no channel set"


@pytest.mark.asyncio
async def test_research_agent_no_tasks(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Test ResearchAgent returns False when no in-progress tasks exist."""
    config = make_config()

    async with running_penny(config) as penny:
        from penny.agents import ResearchAgent
        from penny.constants import SYSTEM_PROMPT

        research_agent = ResearchAgent(
            config=config,
            system_prompt=SYSTEM_PROMPT,
            model=config.ollama_foreground_model,
            ollama_api_url=config.ollama_api_url,
            tools=[],
            db=penny.db,
        )
        research_agent.set_channel(penny.channel)

        # No research tasks in database
        result = await research_agent.execute()
        assert result is False, "Should return False when no tasks"


@pytest.mark.asyncio
async def test_research_agent_truncates_long_reports(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Test ResearchAgent truncates reports exceeding configured max length."""
    # Set a very low max length to force truncation (300 chars)
    config = make_config(
        research_max_iterations=2,
        research_output_max_length=300,
        idle_seconds=0.3,
        scheduler_tick_interval=0.05,
    )

    # Generate long responses that will definitely exceed 300 chars when formatted
    long_finding = "A" * 200  # 200 chars of 'A' per iteration
    responses = [f"Iteration 1: {long_finding}", f"Iteration 2: {long_finding}"]
    response_index = [0]

    def long_response_handler(request: dict, count: int) -> dict:
        if response_index[0] < len(responses):
            response = responses[response_index[0]]
            response_index[0] += 1
            return mock_ollama._make_text_response(request, response)
        return mock_ollama._make_text_response(request, "fallback")

    mock_ollama.set_response_handler(long_response_handler)

    async with running_penny(config):
        await signal_server.push_message(sender=TEST_SENDER, content="/research test topic")
        await signal_server.wait_for_message(timeout=5.0)

        # Wait for report (research runs every 5s, so need ~20-25s for 3 iterations + report)
        await wait_until(lambda: len(signal_server.outgoing_messages) >= 2, timeout=25.0)

        report = signal_server.outgoing_messages[-1]
        # Report should be truncated to 300 chars max
        assert len(report["message"]) <= 300, (
            f"Report should be <= 300 chars, got {len(report['message'])}"
        )
        assert "Report truncated" in report["message"]


@pytest.mark.asyncio
async def test_research_agent_stores_iterations(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Test ResearchAgent stores each iteration in database for resumability."""
    config = make_config(
        research_max_iterations=2,
        idle_seconds=0.3,
        scheduler_tick_interval=0.05,
    )

    responses = [
        "Iteration 1: Arabica beans are known for smooth flavor",
        "Iteration 2: Robusta has more caffeine but bitter taste",
    ]
    response_index = [0]

    def coffee_handler(request: dict, count: int) -> dict:
        if response_index[0] < len(responses):
            response = responses[response_index[0]]
            response_index[0] += 1
            return mock_ollama._make_text_response(request, response)
        return mock_ollama._make_text_response(request, "fallback")

    mock_ollama.set_response_handler(coffee_handler)

    async with running_penny(config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/research coffee beans")
        await signal_server.wait_for_message(timeout=5.0)

        # Wait for completion (research runs every 5s, so need ~20-25s for 2 iterations + report)
        await wait_until(lambda: len(signal_server.outgoing_messages) >= 2, timeout=25.0)

        # Check iterations in database
        with penny.db.get_session() as session:
            tasks = list(session.exec(select(ResearchTask)).all())
            assert len(tasks) == 1

            iterations = list(
                session.exec(
                    select(ResearchIteration)
                    .where(ResearchIteration.research_task_id == tasks[0].id)
                    .order_by(ResearchIteration.iteration_num.asc())  # type: ignore[unresolved-attribute]
                ).all()
            )
            assert len(iterations) == 2

            # First iteration
            assert iterations[0].iteration_num == 1
            assert "arabica" in iterations[0].findings.lower()
            assert iterations[0].timestamp is not None

            # Second iteration
            assert iterations[1].iteration_num == 2
            assert "robusta" in iterations[1].findings.lower()


@pytest.mark.asyncio
async def test_research_report_logged_to_database(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Test research report is logged to MessageLog table."""
    config = make_config(
        research_max_iterations=2,
        idle_seconds=0.3,
        scheduler_tick_interval=0.05,
    )

    responses = [
        "Iteration 1: Machine learning is a subset of AI",
        "Iteration 2: Deep learning uses neural networks",
    ]
    response_index = [0]

    def ai_handler(request: dict, count: int) -> dict:
        if response_index[0] < len(responses):
            response = responses[response_index[0]]
            response_index[0] += 1
            return mock_ollama._make_text_response(request, response)
        return mock_ollama._make_text_response(request, "fallback")

    mock_ollama.set_response_handler(ai_handler)

    async with running_penny(config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/research AI")
        await signal_server.wait_for_message(timeout=5.0)

        # Wait for report (research runs every 5s, so need ~20-25s for 3 iterations + report)
        await wait_until(lambda: len(signal_server.outgoing_messages) >= 2, timeout=25.0)

        # Verify report is in MessageLog
        with penny.db.get_session() as session:
            outgoing = list(
                session.exec(select(MessageLog).where(MessageLog.direction == "outgoing")).all()
            )
            # Should have at least the research report (confirmation may not be logged)
            assert len(outgoing) >= 1

            # Last message should be the research report
            report_msg = outgoing[-1]
            assert "research complete" in report_msg.content.lower()
            assert report_msg.sender == config.signal_number


@pytest.mark.asyncio
async def test_research_generates_proper_report_format(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Test research report has correct markdown format with all sections."""
    config = make_config(
        research_max_iterations=2,
        idle_seconds=0.3,
        scheduler_tick_interval=0.05,
    )

    # Use simple, predictable responses
    responses = ["Finding one about testing", "Finding two about quality"]
    response_index = [0]

    def format_handler(request: dict, count: int) -> dict:
        if response_index[0] < len(responses):
            response = responses[response_index[0]]
            response_index[0] += 1
            return mock_ollama._make_text_response(request, response)
        return mock_ollama._make_text_response(request, "fallback")

    mock_ollama.set_response_handler(format_handler)

    async with running_penny(config):
        await signal_server.push_message(sender=TEST_SENDER, content="/research test topic")
        await signal_server.wait_for_message(timeout=5.0)

        # Wait for report (research runs every 5s, so need ~20-25s for 3 iterations + report)
        await wait_until(lambda: len(signal_server.outgoing_messages) >= 2, timeout=25.0)

        report = signal_server.outgoing_messages[-1]["message"]

        # Verify structure
        assert "research complete: test topic" in report.lower()
        assert "summary" in report.lower()
        assert "key findings" in report.lower()
        assert "sources" in report.lower()

        # Verify sections appear in order
        summary_pos = report.lower().find("summary")
        findings_pos = report.lower().find("key findings")
        sources_pos = report.lower().find("sources")
        assert summary_pos < findings_pos < sources_pos

        # Verify markdown headers are stripped (Signal doesn't support ## headers)
        assert "##" not in report, "Markdown headers should be stripped by prepare_outgoing()"


@pytest.mark.asyncio
async def test_research_filters_markdown_from_llm_findings(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test ResearchAgent filters out markdown headers and table delimiters from LLM findings.

    Regression test for issue #195: LLM may include markdown headers and tables in findings,
    which should not appear as bullet points in the final report.
    """
    config = make_config(
        research_max_iterations=2,
        idle_seconds=0.3,
        scheduler_tick_interval=0.05,
    )

    # Simulate LLM returning findings with markdown headers and table delimiters
    # (This is what caused issue #195)
    responses = [
        "## TL;DR – A solid starter kit\n"
        "| Model | Size | VRAM |\n"
        "|-------|------|------|\n"
        "| Z-Image | 6B | 8GB |\n"
        "Finding: Model runs on 16GB GPU",
        "## Summary\nSecond iteration findings about performance",
    ]
    response_index = [0]

    def markdown_handler(request: dict, count: int) -> dict:
        if response_index[0] < len(responses):
            response = responses[response_index[0]]
            response_index[0] += 1
            return mock_ollama._make_text_response(request, response)
        return mock_ollama._make_text_response(request, "fallback")

    mock_ollama.set_response_handler(markdown_handler)

    async with running_penny(config):
        await signal_server.push_message(sender=TEST_SENDER, content="/research GPU models")
        await signal_server.wait_for_message(timeout=5.0)

        # Wait for report
        await wait_until(lambda: len(signal_server.outgoing_messages) >= 2, timeout=25.0)

        report = signal_server.outgoing_messages[-1]["message"]

        # Verify markdown headers are NOT present (neither standalone nor as bullets)
        assert "##" not in report, "Markdown headers should be filtered from findings"
        # Verify table delimiters are NOT present
        assert "|---" not in report, "Table delimiters should be filtered from findings"
        assert "TL;DR" not in report, "Header text should not appear as bullets"
        # Verify table rows are NOT present (lines starting with |)
        assert "| Model |" not in report, "Table rows should be filtered from findings"
        assert "| Z-Image |" not in report, "Table rows should be filtered from findings"
        # Verify actual findings ARE present
        assert "16GB GPU" in report or "16gb gpu" in report.lower()

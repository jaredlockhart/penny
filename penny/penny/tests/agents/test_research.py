"""Integration tests for ResearchAgent."""

from datetime import UTC, datetime

import pytest
from sqlmodel import select

from penny.database.models import MessageLog, ResearchIteration, ResearchTask
from penny.tests.conftest import TEST_SENDER, wait_until


def _make_research_handler(
    mock_ollama, responses, *, report_text="## report\nResearch report complete"
):
    """Factory for the standard research mock handler.

    Handles the two kinds of Ollama calls the research agent makes:
    1. Tool calls (search iterations) — returns responses in order
    2. Report-building calls (no tools) — returns report_text
    """
    response_index = [0]

    def handler(request: dict, count: int) -> dict:
        if not request.get("tools"):
            return mock_ollama._make_text_response(request, report_text)
        if response_index[0] < len(responses):
            response = responses[response_index[0]]
            response_index[0] += 1
            return mock_ollama._make_text_response(request, response)
        return mock_ollama._make_text_response(request, "fallback")

    return handler


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
    2. Drive iterations via direct execute() calls
    3. Verify report is posted when max_iterations is reached
    """
    config = make_config(
        research_max_iterations=3,
        research_schedule_interval=9999,
        scheduler_tick_interval=9999,
    )

    mock_ollama.set_response_handler(
        _make_research_handler(
            mock_ollama,
            [
                "Iteration 1 findings: quantum computers use qubits",
                "Iteration 2 findings: quantum entanglement enables parallel computation",
                "Iteration 3 findings: major applications in cryptography and drug discovery",
            ],
        )
    )

    async with running_penny(config) as penny:
        # Start research
        await signal_server.push_message(
            sender=TEST_SENDER, content="/research ! quantum computing"
        )
        confirmation = await signal_server.wait_for_message(timeout=5.0)
        assert "started research" in confirmation["message"].lower()

        # Drive iterations directly — 3 iterations + 1 completion call
        search_tool = penny.research_agent._search_tool
        for i in range(4):
            await penny.research_agent.execute()
            # After each iteration, verify skip_images was set correctly:
            # iterations 0,1 → skip_images=True; iteration 2 (last) → skip_images=False
            # iteration 3 is the completion call (doesn't change the flag)
            if search_tool and i < 3:
                expected = i < 2  # True for iter 0,1; False for iter 2
                assert search_tool.skip_images == expected, (
                    f"Iteration {i}: expected skip_images={expected}, got {search_tool.skip_images}"
                )

        # Last message should be the research report
        report = signal_server.outgoing_messages[-1]
        assert "research report complete" in report["message"].lower()

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
        from penny.prompts import SEARCH_PROMPT

        research_agent = ResearchAgent(
            config=config,
            system_prompt=SEARCH_PROMPT,
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
        from penny.prompts import SEARCH_PROMPT

        research_agent = ResearchAgent(
            config=config,
            system_prompt=SEARCH_PROMPT,
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
    config = make_config(
        research_max_iterations=2,
        research_output_max_length=300,
        research_schedule_interval=9999,
        scheduler_tick_interval=9999,
    )

    # Generate long responses that will definitely exceed 300 chars when formatted
    long_finding = "A" * 200  # 200 chars of 'A' per iteration

    mock_ollama.set_response_handler(
        _make_research_handler(
            mock_ollama,
            [f"Iteration 1: {long_finding}", f"Iteration 2: {long_finding}"],
            report_text="A" * 500,
        )
    )

    async with running_penny(config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/research ! test topic")
        await signal_server.wait_for_message(timeout=5.0)

        # Drive iterations directly — 2 iterations + 1 completion call
        for _ in range(3):
            await penny.research_agent.execute()

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
        research_schedule_interval=9999,
        scheduler_tick_interval=9999,
    )

    report_iter1 = "## Coffee Report\nArabica beans are known for smooth flavor"
    report_iter2 = (
        "## Coffee Report\nArabica beans are known for smooth flavor. "
        "Robusta has more caffeine but bitter taste"
    )
    report_index = [0]
    report_texts = [report_iter1, report_iter2]

    def handler(request: dict, count: int) -> dict:
        if not request.get("tools"):
            text = report_texts[min(report_index[0], len(report_texts) - 1)]
            report_index[0] += 1
            return mock_ollama._make_text_response(request, text)
        return mock_ollama._make_text_response(request, "search results about coffee")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/research ! coffee beans")
        await signal_server.wait_for_message(timeout=5.0)

        # Drive iterations directly — 2 iterations + 1 completion call
        for _ in range(3):
            await penny.research_agent.execute()

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

            # First iteration — initial report draft
            assert iterations[0].iteration_num == 1
            assert "arabica" in iterations[0].findings.lower()
            assert iterations[0].timestamp is not None

            # Second iteration — augmented report draft
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
        research_schedule_interval=9999,
        scheduler_tick_interval=9999,
    )

    mock_ollama.set_response_handler(
        _make_research_handler(
            mock_ollama,
            [
                "Iteration 1: Machine learning is a subset of AI",
                "Iteration 2: Deep learning uses neural networks",
            ],
        )
    )

    async with running_penny(config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/research ! AI")
        await signal_server.wait_for_message(timeout=5.0)

        # Drive iterations directly — 2 iterations + 1 completion call
        for _ in range(3):
            await penny.research_agent.execute()

        # Verify report is in MessageLog
        with penny.db.get_session() as session:
            outgoing = list(
                session.exec(select(MessageLog).where(MessageLog.direction == "outgoing")).all()
            )
            # Should have at least the research report (confirmation may not be logged)
            assert len(outgoing) >= 1

            # Last message should be the research report
            report_msg = outgoing[-1]
            assert "research report complete" in report_msg.content.lower()
            assert report_msg.sender == config.signal_number


@pytest.mark.asyncio
async def test_research_agent_activates_pending_task(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test ResearchAgent activates next pending task when current one completes.

    Regression test for issue #207: queued research tasks should auto-activate.
    """
    config = make_config(
        research_max_iterations=2,
        research_schedule_interval=9999,
        scheduler_tick_interval=9999,
    )

    # Track which task we're on based on request count
    response_index = [0]
    first_task_responses = [
        "Iteration 1: AI is transforming industries",
        "Iteration 2: Machine learning powers AI",
    ]
    second_task_responses = [
        "Iteration 1: Quantum computers use qubits",
        "Iteration 2: Quantum entanglement enables speedup",
    ]

    def multi_task_handler(request: dict, count: int) -> dict:
        # Report-building calls (no tools) — extract topic from user content
        if not request.get("tools"):
            user_content = next(
                (m["content"] for m in request.get("messages", []) if m["role"] == "user"),
                "",
            )
            topic = user_content.split("\n")[0].replace("Research topic: ", "")
            return mock_ollama._make_text_response(
                request, f"## report\nResearch report complete about {topic}"
            )
        # First 2 tool calls are for first task
        if response_index[0] < len(first_task_responses):
            response = first_task_responses[response_index[0]]
            response_index[0] += 1
            return mock_ollama._make_text_response(request, response)
        # Next 2 tool calls are for second task
        idx = response_index[0] - len(first_task_responses)
        if idx < len(second_task_responses):
            response = second_task_responses[idx]
            response_index[0] += 1
            return mock_ollama._make_text_response(request, response)
        return mock_ollama._make_text_response(request, "fallback")

    mock_ollama.set_response_handler(multi_task_handler)

    async with running_penny(config) as penny:
        # Start first research task
        await signal_server.push_message(sender=TEST_SENDER, content="/research ! AI trends")
        await signal_server.wait_for_message(timeout=5.0)

        # Queue second research task while first is in progress
        await signal_server.push_message(
            sender=TEST_SENDER, content="/research ! quantum computing"
        )
        response = await signal_server.wait_for_message(timeout=5.0)
        assert "queued" in response["message"].lower()

        # Verify both tasks exist in DB with correct statuses
        with penny.db.get_session() as session:
            tasks = list(
                session.exec(select(ResearchTask).order_by(ResearchTask.created_at.asc())).all()  # type: ignore[unresolved-attribute]
            )
            assert len(tasks) == 2
            assert tasks[0].topic == "AI trends"
            assert tasks[0].status == "in_progress"
            assert tasks[1].topic == "quantum computing"
            assert tasks[1].status == "pending"

        # Drive task 1: 2 iterations + 1 completion (which auto-activates task 2)
        for _ in range(3):
            await penny.research_agent.execute()

        # Verify task 1 completed and task 2 activated
        with penny.db.get_session() as session:
            tasks = list(
                session.exec(select(ResearchTask).order_by(ResearchTask.created_at.asc())).all()  # type: ignore[unresolved-attribute]
            )
            assert tasks[0].status == "completed"
            assert tasks[0].completed_at is not None
            assert tasks[1].status == "in_progress"

        # Drive task 2: 2 iterations + 1 completion
        for _ in range(3):
            await penny.research_agent.execute()

        # Verify both tasks completed with timestamps
        with penny.db.get_session() as session:
            tasks = list(
                session.exec(select(ResearchTask).order_by(ResearchTask.created_at.asc())).all()  # type: ignore[unresolved-attribute]
            )
            assert tasks[0].completed_at is not None
            assert tasks[1].completed_at is not None
            assert tasks[1].status == "completed"

        # Verify both reports were posted
        messages = signal_server.outgoing_messages
        reports = [msg for msg in messages if "research report complete" in msg["message"].lower()]
        assert len(reports) == 2
        # First report should be about AI trends
        assert "ai trends" in reports[0]["message"].lower()
        # Second report should be about quantum computing
        assert "quantum computing" in reports[1]["message"].lower()


@pytest.mark.asyncio
async def test_research_suspended_during_foreground_work(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test research agent is suspended while processing foreground messages.

    Regression test for issue #216: research tasks should not run concurrently
    with user messages to avoid resource contention.
    """
    config = make_config(
        research_max_iterations=5,
        idle_seconds=0.3,
        scheduler_tick_interval=0.05,
    )

    # Track when research iterations vs user messages are processed
    call_log = []
    response_index = [0]
    iteration_responses = [
        "Iteration 1: finding one",
        "Iteration 2: finding two",
        "Iteration 3: finding three",
        "Iteration 4: finding four",
        "Iteration 5: finding five",
    ]

    def track_handler(request: dict, count: int) -> dict:
        # Report-building calls (no tools) - return report text without logging
        if not request.get("tools"):
            return mock_ollama._make_text_response(request, "## report\nResearch report complete")
        # Identify if this is a research iteration or user message
        if any("research" in msg.get("content", "").lower() for msg in request.get("messages", [])):
            call_log.append("research")
            if response_index[0] < len(iteration_responses):
                response = iteration_responses[response_index[0]]
                response_index[0] += 1
                return mock_ollama._make_text_response(request, response)
            return mock_ollama._make_text_response(request, "extra iteration")
        else:
            call_log.append("message")
            return mock_ollama._make_text_response(request, "hi there")

    mock_ollama.set_response_handler(track_handler)

    async with running_penny(config):
        # Start research task
        await signal_server.push_message(sender=TEST_SENDER, content="/research ! test topic")
        await signal_server.wait_for_message(timeout=5.0)

        # Wait for at least one research iteration to start
        await wait_until(lambda: "research" in call_log, timeout=10.0)

        # Send a user message while research is ongoing
        await signal_server.push_message(sender=TEST_SENDER, content="hello")
        await signal_server.wait_for_message(timeout=5.0)

        # Send another message shortly after
        await signal_server.push_message(sender=TEST_SENDER, content="how are you")
        await signal_server.wait_for_message(timeout=5.0)

        # Wait for research to complete
        # Need to see the research report (2 user responses + 1 confirmation + 1 report)
        await wait_until(lambda: len(signal_server.outgoing_messages) >= 4, timeout=10.0)

        # Analyze call log: look for patterns where message calls are interrupted by research
        # The call log should show message processing happening in uninterrupted sequences
        # Expected pattern: [research*, message+, research*, message+, research*]
        # NOT allowed: [message, research, message] (research interrupting message processing)

        for i in range(1, len(call_log) - 1):
            # Check if this is a research call sandwiched between message calls
            if (
                call_log[i] == "research"
                and call_log[i - 1] == "message"
                and call_log[i + 1] == "message"
            ):
                pytest.fail(
                    f"Research call at index {i} interrupted message processing. "
                    f"Research should be suspended during foreground work. "
                    f"Context: {call_log[max(0, i - 2) : min(len(call_log), i + 3)]}, "
                    f"Full log: {call_log}"
                )


@pytest.mark.asyncio
async def test_research_focus_reply_starts_research(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test that replying with a focus to an awaiting_focus task starts research.

    Flow: /research topic -> clarification -> user replies with focus -> research starts
    """
    config = make_config(
        research_max_iterations=2,
        research_schedule_interval=9999,
        scheduler_tick_interval=9999,
    )

    responses = [
        "Iteration 1: Found conferences in Berlin and Amsterdam",
        "Iteration 2: Found conferences in Paris and London",
    ]
    response_index = [0]

    def focus_handler(request: dict, count: int) -> dict:
        if not request.get("tools"):
            system_content = next(
                (m["content"] for m in request.get("messages", []) if m["role"] == "system"),
                "",
            )
            if "interpreting" in system_content:
                return mock_ollama._make_text_response(
                    request, "comprehensive list with dates and locations"
                )
            return mock_ollama._make_text_response(request, "## report\nResearch report complete")
        if response_index[0] < len(responses):
            response = responses[response_index[0]]
            response_index[0] += 1
            return mock_ollama._make_text_response(request, response)
        return mock_ollama._make_text_response(request, "fallback")

    mock_ollama.set_response_handler(focus_handler)

    async with running_penny(config) as penny:
        # Step 1: Start research (default — creates awaiting_focus)
        await signal_server.push_message(
            sender=TEST_SENDER, content="/research AI conferences in Europe"
        )
        clarification = await signal_server.wait_for_message(timeout=5.0)
        assert "what should the report" in clarification["message"].lower()

        # Verify task is awaiting_focus
        with penny.db.get_session() as session:
            tasks = list(session.exec(select(ResearchTask)).all())
            assert len(tasks) == 1
            assert tasks[0].status == "awaiting_focus"

        # Step 2: Reply with focus
        await signal_server.push_message(
            sender=TEST_SENDER, content="comprehensive list with dates and locations"
        )
        focus_response = await signal_server.wait_for_message(timeout=5.0)
        assert "researching" in focus_response["message"].lower()
        assert "focus" in focus_response["message"].lower()

        # Verify task is now in_progress with focus stored
        with penny.db.get_session() as session:
            tasks = list(session.exec(select(ResearchTask)).all())
            assert len(tasks) == 1
            assert tasks[0].status == "in_progress"
            assert tasks[0].focus == "comprehensive list with dates and locations"

        # Step 3: Drive iterations directly — 2 iterations + 1 completion call
        for _ in range(3):
            await penny.research_agent.execute()

        report = signal_server.outgoing_messages[-1]
        assert "research report complete" in report["message"].lower()


@pytest.mark.asyncio
async def test_research_focus_reply_go_starts_without_focus(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test that replying 'go' to an awaiting_focus task starts research without focus.
    """
    config = make_config(
        research_max_iterations=2,
        research_schedule_interval=9999,
        scheduler_tick_interval=9999,
    )

    mock_ollama.set_response_handler(
        _make_research_handler(
            mock_ollama,
            [
                "Iteration 1: finding one",
                "Iteration 2: finding two",
            ],
        )
    )

    async with running_penny(config) as penny:
        # Start research (default — creates awaiting_focus)
        await signal_server.push_message(sender=TEST_SENDER, content="/research test topic")
        await signal_server.wait_for_message(timeout=5.0)

        # Reply with "go" to skip focus
        await signal_server.push_message(sender=TEST_SENDER, content="go")
        go_response = await signal_server.wait_for_message(timeout=5.0)
        assert "started research" in go_response["message"].lower()

        # Verify task is in_progress with no focus
        with penny.db.get_session() as session:
            tasks = list(session.exec(select(ResearchTask)).all())
            assert len(tasks) == 1
            assert tasks[0].status == "in_progress"
            assert tasks[0].focus is None

        # Drive iterations directly — 2 iterations + 1 completion call
        for _ in range(3):
            await penny.research_agent.execute()

        report = signal_server.outgoing_messages[-1]
        assert "research report complete" in report["message"].lower()


@pytest.mark.asyncio
async def test_research_focus_timeout_auto_starts(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test that awaiting_focus tasks auto-start after focus timeout expires.
    """
    config = make_config(
        research_max_iterations=2,
        research_schedule_interval=9999,
        scheduler_tick_interval=9999,
    )

    mock_ollama.set_response_handler(
        _make_research_handler(
            mock_ollama,
            [
                "Iteration 1: finding one",
                "Iteration 2: finding two",
            ],
        )
    )

    async with running_penny(config) as penny:
        from datetime import timedelta

        from sqlmodel import Session

        # Manually create an awaiting_focus task with old created_at (past timeout)
        with Session(penny.db.engine) as session:
            task = ResearchTask(
                thread_id=TEST_SENDER,
                topic="stale topic",
                status="awaiting_focus",
                max_iterations=2,
                created_at=datetime.now(UTC) - timedelta(seconds=400),
            )
            session.add(task)
            session.commit()

        # First execute() triggers focus timeout check (auto-starts task) + runs iteration 1
        # Then iteration 2, then completion
        for _ in range(3):
            await penny.research_agent.execute()

        report = signal_server.outgoing_messages[-1]
        assert "research report complete" in report["message"].lower()

        # Verify task is completed
        with penny.db.get_session() as session:
            tasks = list(session.exec(select(ResearchTask)).all())
            assert len(tasks) == 1
            assert tasks[0].status == "completed"

"""Tests for BackgroundScheduler."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from penny.scheduler.base import BackgroundScheduler, Schedule
from penny.scheduler.schedules import PeriodicSchedule
from penny.tests.conftest import TEST_SENDER


class _SlowAgent:
    """Agent that blocks until an event is set or cancelled."""

    name = "slow_agent"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.completed = False
        self.cancelled = False

    async def execute(self) -> bool:
        self.started.set()
        try:
            await self.release.wait()
            self.completed = True
            return True
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class _SimpleAgent:
    """Agent that returns a fixed value from execute()."""

    def __init__(self, name: str, return_value: bool) -> None:
        self.name = name
        self.return_value = return_value
        self.execute_count = 0

    async def execute(self) -> bool:
        self.execute_count += 1
        return self.return_value


class _AlwaysRunSchedule(Schedule):
    """Schedule that always fires (idle-independent)."""

    def __init__(self, agent: _SlowAgent | _SimpleAgent) -> None:
        self.agent = agent  # type: ignore[assignment]
        self._completed = False

    def should_run(self, is_idle: bool) -> bool:
        return not self._completed

    def mark_complete(self) -> None:
        self._completed = True


class _AlwaysEligibleSchedule(Schedule):
    """Schedule that is always eligible (never marks complete internally)."""

    def __init__(self, agent: _SimpleAgent) -> None:
        self.agent = agent  # type: ignore[assignment]
        self.mark_complete_count = 0

    def should_run(self, is_idle: bool) -> bool:
        return True

    def mark_complete(self) -> None:
        self.mark_complete_count += 1


@pytest.mark.asyncio
async def test_foreground_cancels_active_background_task():
    """When foreground work starts, the active background task is cancelled."""
    agent = _SlowAgent()
    schedule = _AlwaysRunSchedule(agent)
    scheduler = BackgroundScheduler(
        schedules=[schedule],
        idle_threshold=0.0,
        tick_interval=0.01,
    )

    scheduler_task = asyncio.create_task(scheduler.run())
    try:
        # Wait for the agent to start executing
        await asyncio.wait_for(agent.started.wait(), timeout=2.0)

        # Simulate a message arriving — should cancel the background task
        scheduler.notify_foreground_start()

        # Give the scheduler a moment to process the cancellation
        await asyncio.sleep(0.05)

        assert agent.cancelled, "Agent should have been cancelled"
        assert not agent.completed, "Agent should not have completed normally"
        assert scheduler._active_task is None, "Active task should be cleared"
    finally:
        scheduler.stop()
        scheduler_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await scheduler_task


@pytest.mark.asyncio
async def test_foreground_during_idle_prevents_task_start():
    """Background tasks don't start while foreground is active."""
    agent = _SlowAgent()
    schedule = _AlwaysRunSchedule(agent)
    scheduler = BackgroundScheduler(
        schedules=[schedule],
        idle_threshold=0.0,
        tick_interval=0.01,
    )

    # Block background tasks before starting
    scheduler.notify_foreground_start()

    scheduler_task = asyncio.create_task(scheduler.run())
    try:
        # Let several ticks pass
        await asyncio.sleep(0.1)

        assert not agent.started.is_set(), "Agent should not have started while foreground active"

        # Release foreground — background should start
        scheduler.notify_foreground_end()
        await asyncio.wait_for(agent.started.wait(), timeout=2.0)

        assert agent.started.is_set(), "Agent should start after foreground ends"
    finally:
        agent.release.set()
        scheduler.stop()
        scheduler_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await scheduler_task


@pytest.mark.asyncio
async def test_command_does_not_cancel_background_task(
    signal_server, make_config, mock_ollama, running_penny
):
    """Commands don't use Ollama, so they should not interrupt background tasks."""
    config = make_config(IDLE_SECONDS=0.0, MAINTENANCE_INTERVAL_SECONDS=0.01)

    async with running_penny(config) as penny:
        # Spy on the scheduler's notify_foreground_start
        original = penny.scheduler.notify_foreground_start
        calls: list[bool] = []

        def tracking_notify() -> None:
            calls.append(True)
            original()

        penny.scheduler.notify_foreground_start = tracking_notify

        # Send a command (doesn't need Ollama)
        await signal_server.push_message(sender=TEST_SENDER, content="/commands")
        await signal_server.wait_for_message(timeout=5.0)

        assert not calls, "Command should not trigger notify_foreground_start"


@pytest.mark.asyncio
async def test_scheduler_skips_agents_with_no_work():
    """When a higher-priority agent returns False, lower-priority agents get a turn."""
    agent_a = _SimpleAgent("agent_a", return_value=False)
    agent_b = _SimpleAgent("agent_b", return_value=True)

    schedule_a = _AlwaysEligibleSchedule(agent_a)
    schedule_b = _AlwaysEligibleSchedule(agent_b)

    scheduler = BackgroundScheduler(
        schedules=[schedule_a, schedule_b],
        idle_threshold=0.0,
        tick_interval=0.01,
    )

    scheduler_task = asyncio.create_task(scheduler.run())
    try:
        # Let a few ticks run
        await asyncio.sleep(0.1)

        # Agent A (no work) should have been called, but agent B (has work) should also run
        assert agent_a.execute_count > 0, "Higher-priority agent should be called"
        assert agent_b.execute_count > 0, (
            "Lower-priority agent should run when higher returns False"
        )
    finally:
        scheduler.stop()
        scheduler_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await scheduler_task


@pytest.mark.asyncio
async def test_scheduler_mark_complete_always_called():
    """mark_complete is called after every execution, even when agent has no work."""
    agent_no_work = _SimpleAgent("no_work", return_value=False)
    agent_has_work = _SimpleAgent("has_work", return_value=True)

    schedule_no_work = _AlwaysEligibleSchedule(agent_no_work)
    schedule_has_work = _AlwaysEligibleSchedule(agent_has_work)

    scheduler = BackgroundScheduler(
        schedules=[schedule_no_work, schedule_has_work],
        idle_threshold=0.0,
        tick_interval=0.01,
    )

    scheduler_task = asyncio.create_task(scheduler.run())
    try:
        await asyncio.sleep(0.1)

        assert schedule_no_work.mark_complete_count > 0, (
            "mark_complete should be called even when agent returns False"
        )
        assert schedule_has_work.mark_complete_count > 0, (
            "mark_complete should be called when agent returns True"
        )
    finally:
        scheduler.stop()
        scheduler_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await scheduler_task


@pytest.mark.asyncio
async def test_periodic_schedule_interval_respected_without_work():
    """PeriodicSchedule interval gates execution even when the agent has no work."""
    agent = _SimpleAgent("idle_agent", return_value=False)
    schedule = PeriodicSchedule(agent=agent, interval=0.5)  # type: ignore[arg-type]

    scheduler = BackgroundScheduler(
        schedules=[schedule],
        idle_threshold=0.0,
        tick_interval=0.01,
    )

    scheduler_task = asyncio.create_task(scheduler.run())
    try:
        # Let it run for 0.3s — should only get 1 execution (the first immediate run)
        await asyncio.sleep(0.3)
        count_at_300ms = agent.execute_count
        assert count_at_300ms == 1, (
            f"Expected 1 execution in first 0.3s (before interval), got {count_at_300ms}"
        )

        # After 0.5s total the interval elapses — should get a second execution
        await asyncio.sleep(0.3)
        assert agent.execute_count == 2, (
            f"Expected 2 executions after interval elapsed, got {agent.execute_count}"
        )
    finally:
        scheduler.stop()
        scheduler_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await scheduler_task


@pytest.mark.asyncio
async def test_scheduler_breaks_when_agent_does_work():
    """When an agent does work, lower-priority agents don't run in the same tick."""
    agent_a = _SimpleAgent("agent_a", return_value=True)
    agent_b = _SimpleAgent("agent_b", return_value=True)

    schedule_a = _AlwaysEligibleSchedule(agent_a)
    schedule_b = _AlwaysEligibleSchedule(agent_b)

    scheduler = BackgroundScheduler(
        schedules=[schedule_a, schedule_b],
        idle_threshold=0.0,
        tick_interval=0.01,
    )

    scheduler_task = asyncio.create_task(scheduler.run())
    try:
        await asyncio.sleep(0.1)

        # Agent A always does work and breaks — B never gets a turn
        assert agent_a.execute_count > 0
        assert agent_b.execute_count == 0, (
            "Lower-priority agent should not run when higher-priority does work"
        )
    finally:
        scheduler.stop()
        scheduler_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await scheduler_task

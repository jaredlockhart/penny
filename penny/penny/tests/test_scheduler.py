"""Tests for BackgroundScheduler task cancellation."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from penny.scheduler.base import BackgroundScheduler, Schedule


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


class _AlwaysRunSchedule(Schedule):
    """Schedule that always fires (idle-independent)."""

    def __init__(self, agent: _SlowAgent) -> None:
        self.agent = agent  # type: ignore[assignment]
        self._completed = False

    def should_run(self, is_idle: bool) -> bool:
        return not self._completed

    def mark_complete(self) -> None:
        self._completed = True


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

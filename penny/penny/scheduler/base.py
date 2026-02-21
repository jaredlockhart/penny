"""Background task scheduling."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from penny.agents import Agent

logger = logging.getLogger(__name__)


class Schedule:
    """Base class for schedule policies."""

    agent: Agent

    def should_run(self, is_idle: bool) -> bool:
        """
        Check if the schedule condition is met.

        Args:
            is_idle: True if the system has been idle past the global threshold

        Returns:
            True if the task should run now
        """
        return False

    def reset(self) -> None:
        """Reset schedule state. Called when a new message arrives."""
        pass

    def mark_complete(self) -> None:
        """Called after task execution completes."""
        pass


class BackgroundScheduler:
    """Unified scheduler for background tasks."""

    def __init__(
        self,
        schedules: list[Schedule],
        idle_threshold: float,
        tick_interval: float = 1.0,
    ):
        """
        Initialize the scheduler.

        Args:
            schedules: List of schedules in priority order (first checked first)
            idle_threshold: Global idle threshold in seconds before background tasks can run
            tick_interval: How often to check schedules in seconds
        """
        self._schedules = schedules
        self._idle_threshold = idle_threshold
        self._tick_interval = tick_interval
        self._last_message_time = time.monotonic()
        self._running = True
        self._current_task: str | None = None
        self._last_run_times: dict[str, float] = {}
        self._foreground_active = False
        self._active_task: asyncio.Task[bool] | None = None

    def notify_message(self) -> None:
        """Called when a new message arrives. Resets all schedules."""
        self._last_message_time = time.monotonic()
        for schedule in self._schedules:
            schedule.reset()
        logger.debug("Scheduler: all schedules reset by incoming message")

    def notify_foreground_start(self) -> None:
        """Called when foreground work (message/command processing) starts.

        Cancels any running background task so Ollama is immediately free
        to serve the user's message.
        """
        self._foreground_active = True
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()
            logger.info(
                "Scheduler: cancelled background task %s for foreground priority",
                self._current_task,
            )

    def notify_foreground_end(self) -> None:
        """Called when foreground work (message/command processing) ends."""
        self._foreground_active = False
        logger.debug("Scheduler: foreground work ended, background tasks resumed")

    def stop(self) -> None:
        """Signal the scheduler to stop."""
        self._running = False

    def get_agent_status(self) -> dict[str, float | None]:
        """
        Get the time elapsed since each agent last ran.

        Returns:
            Dictionary mapping agent names to seconds since last run (None if never run)
        """
        now = time.monotonic()
        return {
            schedule.agent.name: (
                now - self._last_run_times[schedule.agent.name]
                if schedule.agent.name in self._last_run_times
                else None
            )
            for schedule in self._schedules
        }

    async def run(self) -> None:
        """Main scheduler loop."""
        task_names = [s.agent.name for s in self._schedules]
        logger.info(
            "Background scheduler started with tasks: %s (idle_threshold=%.0fs)",
            task_names,
            self._idle_threshold,
        )

        while self._running:
            idle_seconds = time.monotonic() - self._last_message_time
            is_idle = idle_seconds >= self._idle_threshold

            # Skip all background tasks if foreground work is active
            if not self._foreground_active:
                # Find first schedule that fires
                for schedule in self._schedules:
                    if schedule.should_run(is_idle):
                        agent = schedule.agent
                        self._current_task = agent.name
                        logger.debug("Running background task: %s", agent.name)

                        try:
                            self._active_task = asyncio.create_task(agent.execute())
                            did_work = await self._active_task
                            schedule.mark_complete()
                            self._last_run_times[agent.name] = time.monotonic()

                            if did_work:
                                logger.info("Background task completed: %s", agent.name)

                        except asyncio.CancelledError:
                            logger.info("Background task cancelled: %s", agent.name)
                        except Exception as e:
                            logger.exception("Background task failed: %s - %s", agent.name, e)
                        finally:
                            self._active_task = None
                            self._current_task = None

                        # Only run one task per tick
                        break

            await asyncio.sleep(self._tick_interval)

        logger.info("Background scheduler stopped")

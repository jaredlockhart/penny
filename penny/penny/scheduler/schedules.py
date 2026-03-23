"""Concrete schedule implementations."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from penny.scheduler.base import Schedule

if TYPE_CHECKING:
    from penny.agents import Agent

logger = logging.getLogger(__name__)


class PeriodicSchedule(Schedule):
    """Runs periodically while system is idle."""

    def __init__(
        self,
        agent: Agent,
        interval: float,
    ):
        """
        Initialize periodic schedule.

        Args:
            agent: The agent to execute on each interval
            interval: Time in seconds between executions while idle
        """
        self.agent = agent
        self._interval = interval
        self._last_run: float | None = None
        logger.info(
            "PeriodicSchedule created for %s with interval=%.0fs",
            agent.name,
            interval,
        )

    def should_run(self, is_idle: bool) -> bool:
        """Check if system is idle and interval has elapsed since last run."""
        if not is_idle:
            return False

        now = time.monotonic()
        if self._last_run is None:
            # First run when idle
            return True

        elapsed = now - self._last_run
        return elapsed >= self._interval

    def reset(self) -> None:
        """Reset last run time when a message arrives."""
        self._last_run = None

    def mark_complete(self) -> None:
        """Record completion time for next interval calculation."""
        self._last_run = time.monotonic()


class AlwaysRunSchedule(Schedule):
    """Runs periodically regardless of idle state."""

    def __init__(
        self,
        agent: Agent,
        interval: float,
    ):
        """
        Initialize always-run schedule.

        Args:
            agent: The agent to execute on each interval
            interval: Time in seconds between executions
        """
        self.agent = agent
        self._interval = interval
        self._last_run: float | None = None
        logger.info(
            "AlwaysRunSchedule created for %s with interval=%.0fs",
            agent.name,
            interval,
        )

    def should_run(self, is_idle: bool) -> bool:
        """Check if interval has elapsed since last run, regardless of idle state."""
        now = time.monotonic()
        if self._last_run is None:
            # First run immediately on startup
            return True

        elapsed = now - self._last_run
        return elapsed >= self._interval

    def reset(self) -> None:
        """No-op — this schedule ignores message arrivals."""
        pass

    def mark_complete(self) -> None:
        """Record completion time for next interval calculation."""
        self._last_run = time.monotonic()

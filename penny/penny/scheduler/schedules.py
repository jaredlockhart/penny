"""Concrete schedule implementations."""

from __future__ import annotations

import logging
import random
import time
from enum import StrEnum
from typing import TYPE_CHECKING

from penny.scheduler.base import Schedule

if TYPE_CHECKING:
    from penny.agents import Agent

logger = logging.getLogger(__name__)


class DelayedScheduleState(StrEnum):
    """State machine for delayed schedule."""

    WAITING_IDLE = "waiting_idle"
    WAITING_DELAY = "waiting_delay"


class DelayedSchedule(Schedule):
    """Runs after global idle threshold + random delay."""

    def __init__(
        self,
        agent: Agent,
        min_delay: float,
        max_delay: float,
    ):
        """
        Initialize delayed schedule.

        Args:
            agent: The agent to execute when schedule fires
            min_delay: Minimum random delay in seconds after system becomes idle
            max_delay: Maximum random delay in seconds after system becomes idle
        """
        self.agent = agent
        self._min_delay = min_delay
        self._max_delay = max_delay
        self._state = DelayedScheduleState.WAITING_IDLE
        self._delay_start: float | None = None
        self._delay_target: float | None = None
        logger.info(
            "DelayedSchedule created for %s with delay=%.0fs-%.0fs",
            agent.name,
            min_delay,
            max_delay,
        )

    def should_run(self, is_idle: bool) -> bool:
        """Check if system is idle and random delay has elapsed."""
        if self._state == DelayedScheduleState.WAITING_IDLE:
            if is_idle:
                # Transition to delay phase
                self._state = DelayedScheduleState.WAITING_DELAY
                self._delay_start = time.monotonic()
                self._delay_target = random.uniform(self._min_delay, self._max_delay)
                logger.info(
                    "DelayedSchedule: system idle, delay timer started (%.0fs)",
                    self._delay_target,
                )
            return False

        # In WAITING_DELAY state
        assert self._delay_start is not None
        assert self._delay_target is not None
        elapsed = time.monotonic() - self._delay_start
        return elapsed >= self._delay_target

    def reset(self) -> None:
        """Reset to initial state (waiting for idle)."""
        if self._state == DelayedScheduleState.WAITING_DELAY:
            logger.info("DelayedSchedule: reset by incoming message")
        self._state = DelayedScheduleState.WAITING_IDLE
        self._delay_start = None
        self._delay_target = None

    def mark_complete(self) -> None:
        """Reset after task execution."""
        self.reset()


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

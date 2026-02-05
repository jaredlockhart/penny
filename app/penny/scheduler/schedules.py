"""Concrete schedule implementations."""

from __future__ import annotations

import logging
import random
import time
from enum import StrEnum
from typing import TYPE_CHECKING

from penny.scheduler.base import Schedule

if TYPE_CHECKING:
    from penny.agent import Agent

logger = logging.getLogger(__name__)


class ImmediateSchedule(Schedule):
    """Runs immediately when system becomes idle, then waits for reset."""

    def __init__(self, agent: Agent):
        """
        Initialize immediate schedule.

        Args:
            agent: The agent to execute when schedule fires
        """
        self.agent = agent
        self._fired = False
        logger.info("ImmediateSchedule created for %s", agent.name)

    def should_run(self, is_idle: bool) -> bool:
        """Check if system is idle and hasn't fired yet."""
        if self._fired:
            return False
        return is_idle

    def reset(self) -> None:
        """Reset fired state when a message arrives."""
        self._fired = False

    def mark_complete(self) -> None:
        """Mark as fired so it won't run again until reset."""
        self._fired = True


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

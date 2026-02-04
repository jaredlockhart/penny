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


class IdleSchedule(Schedule):
    """Triggers after a fixed idle period."""

    def __init__(self, agent: Agent, idle_seconds: float):
        """
        Initialize idle schedule.

        Args:
            agent: The agent to execute when schedule fires
            idle_seconds: Seconds of idle time before triggering
        """
        self.agent = agent
        self._idle_threshold = idle_seconds
        logger.info(
            "IdleSchedule created for %s with threshold %.0fs",
            agent.name,
            idle_seconds,
        )

    def should_run(self, idle_seconds: float) -> bool:
        """Check if idle threshold has been reached."""
        return idle_seconds >= self._idle_threshold

    def reset(self) -> None:
        """No state to reset."""
        pass

    def mark_complete(self) -> None:
        """No post-completion action needed."""
        pass


class TwoPhaseState(StrEnum):
    """State machine for two-phase schedule."""

    WAITING_IDLE = "waiting_idle"
    WAITING_DELAY = "waiting_delay"


class TwoPhaseSchedule(Schedule):
    """Two-phase schedule: idle threshold, then random delay."""

    def __init__(
        self,
        agent: Agent,
        idle_seconds: float,
        min_delay: float,
        max_delay: float,
    ):
        """
        Initialize two-phase schedule.

        Args:
            agent: The agent to execute when schedule fires
            idle_seconds: Phase 1 - seconds of idle time before entering delay phase
            min_delay: Phase 2 - minimum random delay in seconds
            max_delay: Phase 2 - maximum random delay in seconds
        """
        self.agent = agent
        self._idle_threshold = idle_seconds
        self._min_delay = min_delay
        self._max_delay = max_delay
        self._state = TwoPhaseState.WAITING_IDLE
        self._delay_start: float | None = None
        self._delay_target: float | None = None
        logger.info(
            "TwoPhaseSchedule created for %s with idle=%.0fs, delay=%.0fs-%.0fs",
            agent.name,
            idle_seconds,
            min_delay,
            max_delay,
        )

    def should_run(self, idle_seconds: float) -> bool:
        """Check if both phases are complete."""
        if self._state == TwoPhaseState.WAITING_IDLE:
            if idle_seconds >= self._idle_threshold:
                # Transition to delay phase
                self._state = TwoPhaseState.WAITING_DELAY
                self._delay_start = time.monotonic()
                self._delay_target = random.uniform(self._min_delay, self._max_delay)
                logger.info(
                    "TwoPhaseSchedule: idle threshold reached, delay timer started (%.0fs)",
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
        if self._state == TwoPhaseState.WAITING_DELAY:
            logger.info("TwoPhaseSchedule: reset by incoming message")
        self._state = TwoPhaseState.WAITING_IDLE
        self._delay_start = None
        self._delay_target = None

    def mark_complete(self) -> None:
        """Reset after task execution."""
        self.reset()

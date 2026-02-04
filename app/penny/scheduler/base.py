"""Base classes for background task scheduling."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from penny.agent import Agent


class Schedule:
    """Base class for schedule policies."""

    agent: Agent

    def should_run(self, idle_seconds: float) -> bool:
        """
        Check if the schedule condition is met.

        Args:
            idle_seconds: How long since the last message was received

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

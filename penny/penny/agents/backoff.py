"""Exponential backoff state machine for background agents."""

from __future__ import annotations

from datetime import UTC, datetime


def _ensure_utc(dt: datetime) -> datetime:
    """Ensure a datetime has UTC timezone info."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


class BackoffState:
    """Exponential backoff that resets on user interaction.

    Shared by NotificationAgent (one instance per user) and
    EnrichAgent (single global instance).

    State machine:
        Initial (backoff=0) → first action sets backoff to `initial`.
        Each subsequent action doubles backoff up to `max`.
        User interaction (latest_interaction > last_action_time) resets to 0.
    """

    __slots__ = ("last_action_time", "backoff_seconds")

    def __init__(self) -> None:
        self.last_action_time: datetime | None = None
        self.backoff_seconds: float = 0.0

    def should_act(self, latest_interaction: datetime | None) -> bool:
        """Check if enough time has elapsed to act again.

        Args:
            latest_interaction: Most recent user interaction time.
                If more recent than last_action_time, backoff resets to 0.
        """
        if self.last_action_time is None:
            return True

        # User interaction since last action → reset backoff
        if latest_interaction is not None and _ensure_utc(latest_interaction) > _ensure_utc(
            self.last_action_time
        ):
            self.backoff_seconds = 0.0

        if self.backoff_seconds <= 0:
            return True

        elapsed = (datetime.now(UTC) - _ensure_utc(self.last_action_time)).total_seconds()
        return elapsed >= self.backoff_seconds

    def mark_done(self, initial_backoff: float, max_backoff: float) -> None:
        """Record that an action was performed and increase backoff."""
        self.last_action_time = datetime.now(UTC)
        if self.backoff_seconds <= 0:
            self.backoff_seconds = initial_backoff
        else:
            self.backoff_seconds = min(
                self.backoff_seconds * 2,
                max_backoff,
            )

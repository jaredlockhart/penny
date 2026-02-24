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

    Used by NotificationAgent (one instance per user).

    State machine:
        Initial (backoff=0, last_action_time=None) → first action fires immediately.
        After a notification, backoff is set to `initial` and doubles on each send up to `max`.
        User interaction (latest_interaction > last_action_time) resets the clock to the
        interaction time with backoff=`initial`, so the next notification waits `initial`
        seconds from the interaction — not immediately on the first idle tick.
    """

    __slots__ = ("last_action_time", "backoff_seconds")

    def __init__(self) -> None:
        self.last_action_time: datetime | None = None
        self.backoff_seconds: float = 0.0

    def should_act(self, latest_interaction: datetime | None, initial_backoff: float) -> bool:
        """Check if enough time has elapsed to act again.

        Args:
            latest_interaction: Most recent user interaction time.
                If more recent than last_action_time, resets to wait initial_backoff
                from the interaction time before firing.
            initial_backoff: Seconds to wait after user interaction before first notification.
        """
        if self.last_action_time is None:
            # Never acted — fire immediately on first opportunity
            return True

        # User interaction since last action → reset: wait initial_backoff from interaction time
        if latest_interaction is not None and _ensure_utc(latest_interaction) > _ensure_utc(
            self.last_action_time
        ):
            self.backoff_seconds = initial_backoff
            self.last_action_time = _ensure_utc(latest_interaction)

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

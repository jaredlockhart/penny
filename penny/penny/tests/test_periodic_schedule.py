"""Unit tests for PeriodicSchedule."""

from unittest.mock import MagicMock

import pytest

from penny.scheduler.schedules import PeriodicSchedule


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    agent = MagicMock()
    agent.name = "test_agent"
    return agent


def test_periodic_schedule_first_run_when_idle(mock_agent):
    """Test that PeriodicSchedule fires immediately on first idle."""
    schedule = PeriodicSchedule(agent=mock_agent, interval=lambda: 60.0)

    # Should fire on first idle check
    assert schedule.should_run(is_idle=True)


def test_periodic_schedule_not_run_when_not_idle(mock_agent):
    """Test that PeriodicSchedule does not fire when system is not idle."""
    schedule = PeriodicSchedule(agent=mock_agent, interval=lambda: 60.0)

    # Should not fire if not idle
    assert not schedule.should_run(is_idle=False)


def test_periodic_schedule_respects_interval(mock_agent):
    """Test that PeriodicSchedule respects the interval between runs."""
    schedule = PeriodicSchedule(agent=mock_agent, interval=lambda: 0.1)

    # First run should fire
    assert schedule.should_run(is_idle=True)
    schedule.mark_complete()

    # Immediately after, should not fire
    assert not schedule.should_run(is_idle=True)

    # Advance past the interval by backdating _last_run
    assert schedule._last_run is not None
    schedule._last_run = schedule._last_run - 0.2

    # Should fire again after interval
    assert schedule.should_run(is_idle=True)


def test_periodic_schedule_reset_clears_last_run(mock_agent):
    """Test that reset() clears the last run time."""
    schedule = PeriodicSchedule(agent=mock_agent, interval=lambda: 60.0)

    # Fire once and mark complete
    schedule.should_run(is_idle=True)
    schedule.mark_complete()

    # After reset, should fire again immediately when idle
    schedule.reset()
    assert schedule.should_run(is_idle=True)


def test_independent_schedule_runs_when_not_idle(mock_agent):
    """requires_idle=False schedules run regardless of idle state once interval elapses."""
    schedule = PeriodicSchedule(agent=mock_agent, interval=lambda: 0.1, requires_idle=False)

    # Not yet — clock just started
    assert not schedule.should_run(is_idle=False)

    # Backdate to simulate interval elapsed
    assert schedule._last_run is not None
    schedule._last_run -= 0.2
    assert schedule.should_run(is_idle=False)


def test_independent_schedule_does_not_fire_immediately(mock_agent):
    """requires_idle=False schedules don't fire on the first tick (unlike idle-gated)."""
    schedule = PeriodicSchedule(agent=mock_agent, interval=lambda: 99999.0, requires_idle=False)
    assert not schedule.should_run(is_idle=False)
    assert not schedule.should_run(is_idle=True)


def test_independent_schedule_reset_is_noop(mock_agent):
    """reset() does not clear last_run for idle-independent schedules."""
    schedule = PeriodicSchedule(agent=mock_agent, interval=lambda: 0.1, requires_idle=False)
    assert schedule._last_run is not None
    schedule._last_run -= 0.2  # Simulate interval elapsed
    schedule.mark_complete()

    schedule.reset()

    # Timer still set — should not fire until interval elapses again
    assert not schedule.should_run(is_idle=False)


def test_idle_required_schedule_reset_clears_timer(mock_agent):
    """reset() clears last_run for idle-gated schedules so they fire after next idle."""
    schedule = PeriodicSchedule(agent=mock_agent, interval=lambda: 60.0, requires_idle=True)
    schedule.mark_complete()

    schedule.reset()

    assert schedule.should_run(is_idle=True)

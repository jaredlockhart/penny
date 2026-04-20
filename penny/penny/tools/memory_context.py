"""Contextvar carrying the current agent name for memory-write attribution.

The memory tool layer stamps the ``author`` field on every write/move/append/
delete with ``current_agent()``. The orchestration layer (Stage 2b / Stage 4+)
is responsible for calling ``set_current_agent`` at the start of each agent
run so writes are correctly attributed.

Default value is ``"unknown"`` so that unit tests and ad-hoc usage of the
tools (e.g. from a REPL) don't crash — they just log with the fallback.
"""

from __future__ import annotations

from contextvars import ContextVar

_DEFAULT_AGENT = "unknown"

_current_agent: ContextVar[str] = ContextVar("current_agent", default=_DEFAULT_AGENT)


def current_agent() -> str:
    """Return the current agent name, or ``"unknown"`` if none is set."""
    return _current_agent.get()


def set_current_agent(name: str) -> None:
    """Set the current agent name for the lifetime of the current context."""
    _current_agent.set(name)

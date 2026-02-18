"""Runtime configuration parameter definitions."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ConfigParam:
    """Definition of a runtime-configurable parameter."""

    key: str
    description: str
    type: type  # int or float
    default_value: int | float
    validator: Callable[[str], int | float]  # Parses and validates value from string


def _validate_positive_int(value: str) -> int:
    """Validate positive integer."""
    try:
        parsed = int(value)
    except ValueError as e:
        raise ValueError("must be a positive integer") from e

    if parsed <= 0:
        raise ValueError("must be a positive integer")

    return parsed


def _validate_positive_float(value: str) -> float:
    """Validate positive float."""
    try:
        parsed = float(value)
    except ValueError as e:
        raise ValueError("must be a positive number") from e

    if parsed <= 0:
        raise ValueError("must be a positive number")

    return parsed


# Runtime configurable parameters
RUNTIME_CONFIG_PARAMS: dict[str, ConfigParam] = {
    "MESSAGE_MAX_STEPS": ConfigParam(
        key="MESSAGE_MAX_STEPS",
        description="Max agent loop steps per message",
        type=int,
        default_value=5,
        validator=_validate_positive_int,
    ),
    "IDLE_SECONDS": ConfigParam(
        key="IDLE_SECONDS",
        description="Global idle threshold in seconds",
        type=float,
        default_value=300.0,
        validator=_validate_positive_float,
    ),
    "MAINTENANCE_INTERVAL_SECONDS": ConfigParam(
        key="MAINTENANCE_INTERVAL_SECONDS",
        description="Interval for periodic maintenance tasks (summarize, profile) in seconds",
        type=float,
        default_value=300.0,
        validator=_validate_positive_float,
    ),
    "LEARN_LOOP_INTERVAL": ConfigParam(
        key="LEARN_LOOP_INTERVAL",
        description="Interval for learn loop in seconds (runs during idle)",
        type=float,
        default_value=300.0,
        validator=_validate_positive_float,
    ),
}

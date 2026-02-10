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
    "FOLLOWUP_MIN_SECONDS": ConfigParam(
        key="FOLLOWUP_MIN_SECONDS",
        description="Min delay for spontaneous followup in seconds",
        type=float,
        default_value=3600.0,
        validator=_validate_positive_float,
    ),
    "FOLLOWUP_MAX_SECONDS": ConfigParam(
        key="FOLLOWUP_MAX_SECONDS",
        description="Max delay for spontaneous followup in seconds",
        type=float,
        default_value=7200.0,
        validator=_validate_positive_float,
    ),
    "DISCOVERY_MIN_SECONDS": ConfigParam(
        key="DISCOVERY_MIN_SECONDS",
        description="Min delay for discovery agent in seconds",
        type=float,
        default_value=7200.0,
        validator=_validate_positive_float,
    ),
    "DISCOVERY_MAX_SECONDS": ConfigParam(
        key="DISCOVERY_MAX_SECONDS",
        description="Max delay for discovery agent in seconds",
        type=float,
        default_value=14400.0,
        validator=_validate_positive_float,
    ),
}

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


def _validate_unit_float(value: str) -> float:
    """Validate float in (0.0, 1.0] range for similarity thresholds."""
    try:
        parsed = float(value)
    except ValueError as e:
        raise ValueError("must be a number between 0 and 1") from e

    if not (0.0 < parsed <= 1.0):
        raise ValueError("must be a number between 0 and 1")

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
    # Extraction pipeline thresholds
    "EXTRACTION_ENTITY_SEMANTIC_THRESHOLD": ConfigParam(
        key="EXTRACTION_ENTITY_SEMANTIC_THRESHOLD",
        description="Cosine similarity threshold for semantic entity name validation",
        type=float,
        default_value=0.58,
        validator=_validate_unit_float,
    ),
    "EXTRACTION_PREFILTER_SIMILARITY_THRESHOLD": ConfigParam(
        key="EXTRACTION_PREFILTER_SIMILARITY_THRESHOLD",
        description="Cosine similarity threshold for entity pre-filtering",
        type=float,
        default_value=0.2,
        validator=_validate_unit_float,
    ),
    "EXTRACTION_PREFILTER_MIN_COUNT": ConfigParam(
        key="EXTRACTION_PREFILTER_MIN_COUNT",
        description="Minimum known-entity count before pre-filtering is applied",
        type=int,
        default_value=20,
        validator=_validate_positive_int,
    ),
    "EXTRACTION_ENTITY_DEDUP_TCR_THRESHOLD": ConfigParam(
        key="EXTRACTION_ENTITY_DEDUP_TCR_THRESHOLD",
        description="Token containment ratio threshold for entity deduplication",
        type=float,
        default_value=0.75,
        validator=_validate_unit_float,
    ),
    "EXTRACTION_ENTITY_DEDUP_EMBEDDING_THRESHOLD": ConfigParam(
        key="EXTRACTION_ENTITY_DEDUP_EMBEDDING_THRESHOLD",
        description="Embedding similarity threshold for entity deduplication",
        type=float,
        default_value=0.85,
        validator=_validate_unit_float,
    ),
    "EXTRACTION_FACT_DEDUP_SIMILARITY_THRESHOLD": ConfigParam(
        key="EXTRACTION_FACT_DEDUP_SIMILARITY_THRESHOLD",
        description="Embedding similarity threshold for fact deduplication",
        type=float,
        default_value=0.85,
        validator=_validate_unit_float,
    ),
    "EXTRACTION_MIN_MESSAGE_LENGTH": ConfigParam(
        key="EXTRACTION_MIN_MESSAGE_LENGTH",
        description="Minimum character length for messages to be processed for extraction",
        type=int,
        default_value=20,
        validator=_validate_positive_int,
    ),
    # Notification settings
    "NOTIFICATION_INITIAL_BACKOFF": ConfigParam(
        key="NOTIFICATION_INITIAL_BACKOFF",
        description="Initial backoff in seconds after sending a fact notification",
        type=float,
        default_value=60.0,
        validator=_validate_positive_float,
    ),
    "NOTIFICATION_MAX_BACKOFF": ConfigParam(
        key="NOTIFICATION_MAX_BACKOFF",
        description="Maximum backoff cap in seconds for fact notifications",
        type=float,
        default_value=3600.0,
        validator=_validate_positive_float,
    ),
    "NOTIFICATION_MIN_LENGTH": ConfigParam(
        key="NOTIFICATION_MIN_LENGTH",
        description="Minimum character length for a fact notification to be sent",
        type=int,
        default_value=75,
        validator=_validate_positive_int,
    ),
    # Learn loop tuning
    "LEARN_ENRICHMENT_FACT_THRESHOLD": ConfigParam(
        key="LEARN_ENRICHMENT_FACT_THRESHOLD",
        description="Fact count below which an entity enters enrichment mode",
        type=int,
        default_value=5,
        validator=_validate_positive_int,
    ),
    "LEARN_STALENESS_DAYS": ConfigParam(
        key="LEARN_STALENESS_DAYS",
        description="Days since last verification before an entity is considered stale",
        type=float,
        default_value=7.0,
        validator=_validate_positive_float,
    ),
    "LEARN_MIN_INTEREST_SCORE": ConfigParam(
        key="LEARN_MIN_INTEREST_SCORE",
        description="Minimum interest score for an entity to be considered for enrichment",
        type=float,
        default_value=0.1,
        validator=_validate_positive_float,
    ),
    "LEARN_RECENT_DAYS": ConfigParam(
        key="LEARN_RECENT_DAYS",
        description="Skip enrichment if entity was verified within this many days",
        type=float,
        default_value=1.0,
        validator=_validate_positive_float,
    ),
}

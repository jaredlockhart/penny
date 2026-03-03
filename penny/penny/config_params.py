"""Runtime configuration parameter definitions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from penny.database import Database

# Auto-populated by ConfigParam.__post_init__
RUNTIME_CONFIG_PARAMS: dict[str, ConfigParam] = {}

# Group names (display order)
GROUP_GLOBAL = "Global"
GROUP_SCHEDULE = "Schedule"
GROUP_KNOWLEDGE = "Knowledge"
GROUP_EXTRACTION = "Extraction"
GROUP_LEARN = "Learn"
GROUP_EVENTS = "Events"
GROUP_INNER_MONOLOGUE = "Inner Monologue"
GROUP_HISTORY = "History"

# Ordered list for display
CONFIG_GROUPS: list[str] = [
    GROUP_GLOBAL,
    GROUP_SCHEDULE,
    GROUP_KNOWLEDGE,
    GROUP_EXTRACTION,
    GROUP_LEARN,
    GROUP_EVENTS,
    GROUP_INNER_MONOLOGUE,
    GROUP_HISTORY,
]


@dataclass
class ConfigParam:
    """Definition of a runtime-configurable parameter.

    Automatically registers itself into RUNTIME_CONFIG_PARAMS on creation.
    """

    key: str
    description: str
    type: type  # int or float
    default: int | float  # Default value (single source of truth)
    validator: Callable[[str], int | float]  # Parses and validates value from string
    group: str = GROUP_GLOBAL  # Display group for /config listing

    def __post_init__(self) -> None:
        RUNTIME_CONFIG_PARAMS[self.key] = self


def get_params_by_group() -> list[tuple[str, list[ConfigParam]]]:
    """Return params grouped by category in display order.

    Within each group, params are sorted alphabetically by key.
    """
    groups: dict[str, list[ConfigParam]] = {g: [] for g in CONFIG_GROUPS}
    for param in RUNTIME_CONFIG_PARAMS.values():
        groups[param.group].append(param)
    return [(g, sorted(groups[g], key=lambda p: p.key)) for g in CONFIG_GROUPS if groups[g]]


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


# ── Global ────────────────────────────────────────────────────────────────────

ConfigParam(
    key="MESSAGE_MAX_STEPS",
    description="Max agent loop steps per message",
    type=int,
    default=5,
    validator=_validate_positive_int,
    group=GROUP_GLOBAL,
)

ConfigParam(
    key="IMAGE_DOWNLOAD_TIMEOUT",
    description="Timeout in seconds for image downloads",
    type=float,
    default=15.0,
    validator=_validate_positive_float,
    group=GROUP_GLOBAL,
)

ConfigParam(
    key="IMAGE_MAX_RESULTS",
    description="Max image search results to return",
    type=int,
    default=3,
    validator=_validate_positive_int,
    group=GROUP_GLOBAL,
)

ConfigParam(
    key="EMAIL_BODY_MAX_LENGTH",
    description="Max character length for email body content",
    type=int,
    default=4000,
    validator=_validate_positive_int,
    group=GROUP_GLOBAL,
)

ConfigParam(
    key="JMAP_REQUEST_TIMEOUT",
    description="Timeout in seconds for JMAP API requests",
    type=float,
    default=30.0,
    validator=_validate_positive_float,
    group=GROUP_GLOBAL,
)

ConfigParam(
    key="EMBEDDING_BACKFILL_BATCH_LIMIT",
    description="Max facts/entities per embedding backfill cycle",
    type=int,
    default=50,
    validator=_validate_positive_int,
    group=GROUP_GLOBAL,
)

# ── Schedule ──────────────────────────────────────────────────────────────────

ConfigParam(
    key="IDLE_SECONDS",
    description="Global idle threshold in seconds",
    type=float,
    default=60.0,
    validator=_validate_positive_float,
    group=GROUP_SCHEDULE,
)

ConfigParam(
    key="MAINTENANCE_INTERVAL_SECONDS",
    description="Interval for the knowledge pipeline cycle in seconds",
    type=float,
    default=10.0,
    validator=_validate_positive_float,
    group=GROUP_SCHEDULE,
)

# ── Knowledge ─────────────────────────────────────────────────────────────────

ConfigParam(
    key="ENTITY_CONTEXT_MAX_FACTS",
    description="Max facts per entity in message context",
    type=int,
    default=5,
    validator=_validate_positive_int,
    group=GROUP_KNOWLEDGE,
)

ConfigParam(
    key="ENTITY_CONTEXT_THRESHOLD",
    description="Cosine similarity threshold for entity context injection",
    type=float,
    default=0.3,
    validator=_validate_unit_float,
    group=GROUP_KNOWLEDGE,
)

ConfigParam(
    key="ENTITY_CONTEXT_TOP_K",
    description="Number of top similar entities to inject into message context",
    type=int,
    default=5,
    validator=_validate_positive_int,
    group=GROUP_KNOWLEDGE,
)

ConfigParam(
    key="KNOWLEDGE_SUFFICIENT_MIN_FACTS",
    description="Min facts for entity context to be considered sufficient",
    type=int,
    default=3,
    validator=_validate_positive_int,
    group=GROUP_KNOWLEDGE,
)

ConfigParam(
    key="KNOWLEDGE_SUFFICIENT_MIN_SCORE",
    description="Min similarity score for entity context to be considered sufficient",
    type=float,
    default=0.5,
    validator=_validate_unit_float,
    group=GROUP_KNOWLEDGE,
)

ConfigParam(
    key="MESSAGE_CONTEXT_LIMIT",
    description="Max recent conversation messages injected into message context",
    type=int,
    default=20,
    validator=_validate_positive_int,
    group=GROUP_KNOWLEDGE,
)

ConfigParam(
    key="EVENT_CONTEXT_TOP_K",
    description="Number of top similar events to inject into message context",
    type=int,
    default=3,
    validator=_validate_positive_int,
    group=GROUP_KNOWLEDGE,
)

ConfigParam(
    key="EVENT_CONTEXT_THRESHOLD",
    description="Cosine similarity threshold for event context injection",
    type=float,
    default=0.3,
    validator=_validate_unit_float,
    group=GROUP_KNOWLEDGE,
)

# ── Extraction ────────────────────────────────────────────────────────────────

ConfigParam(
    key="ENTITY_EXTRACTION_BATCH_LIMIT",
    description="Max search logs processed per extraction cycle",
    type=int,
    default=10,
    validator=_validate_positive_int,
    group=GROUP_EXTRACTION,
)

ConfigParam(
    key="MESSAGE_EXTRACTION_BATCH_LIMIT",
    description="Max messages/reactions processed per user per extraction cycle",
    type=int,
    default=20,
    validator=_validate_positive_int,
    group=GROUP_EXTRACTION,
)

ConfigParam(
    key="EXTRACTION_MIN_MESSAGE_LENGTH",
    description="Minimum character length for messages to be processed for extraction",
    type=int,
    default=20,
    validator=_validate_positive_int,
    group=GROUP_EXTRACTION,
)

ConfigParam(
    key="EXTRACTION_PREFILTER_MIN_COUNT",
    description="Minimum known-entity count before pre-filtering is applied",
    type=int,
    default=20,
    validator=_validate_positive_int,
    group=GROUP_EXTRACTION,
)

ConfigParam(
    key="EXTRACTION_PREFILTER_SIMILARITY_THRESHOLD",
    description="Cosine similarity threshold for entity pre-filtering",
    type=float,
    default=0.4,
    validator=_validate_unit_float,
    group=GROUP_EXTRACTION,
)

ConfigParam(
    key="EXTRACTION_ENTITY_SEMANTIC_THRESHOLD",
    description="Cosine similarity threshold for semantic entity name validation",
    type=float,
    default=0.50,
    validator=_validate_unit_float,
    group=GROUP_EXTRACTION,
)

ConfigParam(
    key="EXTRACTION_ENTITY_DEDUP_EMBEDDING_THRESHOLD",
    description="Embedding similarity threshold for entity deduplication",
    type=float,
    default=0.85,
    validator=_validate_unit_float,
    group=GROUP_EXTRACTION,
)

ConfigParam(
    key="EXTRACTION_ENTITY_DEDUP_TCR_THRESHOLD",
    description="Token containment ratio threshold for entity deduplication",
    type=float,
    default=0.6,
    validator=_validate_unit_float,
    group=GROUP_EXTRACTION,
)

ConfigParam(
    key="EXTRACTION_MAX_NEW_ENTITIES",
    description="Max new entity candidates per extraction (ranked by relevance)",
    type=int,
    default=10,
    validator=_validate_positive_int,
    group=GROUP_EXTRACTION,
)

ConfigParam(
    key="EXTRACTION_FACT_DEDUP_SIMILARITY_THRESHOLD",
    description="Embedding similarity threshold for fact deduplication",
    type=float,
    default=0.85,
    validator=_validate_unit_float,
    group=GROUP_EXTRACTION,
)

# ── Learn ─────────────────────────────────────────────────────────────────────

ConfigParam(
    key="LEARN_PROMPT_DEFAULT_SEARCHES",
    description="Number of searches performed per /learn command",
    type=int,
    default=3,
    validator=_validate_positive_int,
    group=GROUP_LEARN,
)

ConfigParam(
    key="LEARN_STATUS_DISPLAY_LIMIT",
    description="Max learn prompts shown in /learn status",
    type=int,
    default=10,
    validator=_validate_positive_int,
    group=GROUP_LEARN,
)

# ── Inner Monologue ──────────────────────────────────────────────────────────

ConfigParam(
    key="INNER_MONOLOGUE_INTERVAL",
    description="Interval in seconds between inner monologue cycles",
    type=float,
    default=600.0,
    validator=_validate_positive_float,
    group=GROUP_INNER_MONOLOGUE,
)

ConfigParam(
    key="INNER_MONOLOGUE_MAX_STEPS",
    description="Max agentic loop steps per inner monologue cycle",
    type=int,
    default=20,
    validator=_validate_positive_int,
    group=GROUP_INNER_MONOLOGUE,
)


# ── Events ───────────────────────────────────────────────────────────────────

ConfigParam(
    key="EVENT_DEDUP_SIMILARITY_THRESHOLD",
    description="Embedding cosine similarity threshold for headline dedup",
    type=float,
    default=0.60,
    validator=_validate_unit_float,
    group=GROUP_EVENTS,
)

ConfigParam(
    key="EVENT_DEDUP_TCR_THRESHOLD",
    description="Token containment ratio threshold for headline dedup",
    type=float,
    default=0.60,
    validator=_validate_unit_float,
    group=GROUP_EVENTS,
)

ConfigParam(
    key="EVENT_DEDUP_WINDOW_DAYS",
    description="Number of days to look back for dedup comparison",
    type=int,
    default=7,
    validator=_validate_positive_int,
    group=GROUP_EVENTS,
)

ConfigParam(
    key="EVENT_RELEVANCE_THRESHOLD",
    description="Cosine similarity threshold for article-to-topic relevance",
    type=float,
    default=0.40,
    validator=_validate_positive_float,
    group=GROUP_EVENTS,
)

ConfigParam(
    key="EVENT_MAX_PER_POLL",
    description="Maximum number of events to store per follow prompt poll cycle",
    type=int,
    default=5,
    validator=_validate_positive_int,
    group=GROUP_EVENTS,
)

ConfigParam(
    key="EVENT_POLL_INTERVAL_SECONDS",
    description="Fixed interval between polls for each follow prompt (seconds)",
    type=float,
    default=600.0,
    validator=_validate_positive_float,
    group=GROUP_EVENTS,
)


# ── History ──────────────────────────────────────────────────────────────────

ConfigParam(
    key="HISTORY_INTERVAL",
    description="Interval in seconds between history summarization runs",
    type=float,
    default=3600.0,
    validator=_validate_positive_float,
    group=GROUP_HISTORY,
)

ConfigParam(
    key="HISTORY_MAX_DAYS_PER_RUN",
    description="Max days to summarize per history agent execution",
    type=int,
    default=3,
    validator=_validate_positive_int,
    group=GROUP_HISTORY,
)

ConfigParam(
    key="HISTORY_CONTEXT_LIMIT",
    description="Max daily history entries to show in context",
    type=int,
    default=14,
    validator=_validate_positive_int,
    group=GROUP_HISTORY,
)


class RuntimeParams:
    """Accessor for runtime-configurable parameters.

    Lookup chain: DB override → env override → ConfigParam.default.
    Supports attribute access with uppercase keys: config.runtime.IDLE_SECONDS
    """

    def __init__(
        self,
        db: Database | None = None,
        env_overrides: dict[str, int | float] | None = None,
    ) -> None:
        self._db = db
        self._env_overrides = env_overrides or {}

    def __getattr__(self, name: str) -> int | float:
        key = name.upper()
        if key not in RUNTIME_CONFIG_PARAMS:
            raise AttributeError(f"No runtime config param: {name}")

        # 1. Check database
        if self._db is not None:
            db_value = self._get_db_value(key)
            if db_value is not None:
                return db_value

        # 2. Check env overrides (from Config.load)
        if key in self._env_overrides:
            return self._env_overrides[key]

        # 3. Fall back to default
        return RUNTIME_CONFIG_PARAMS[key].default

    def _get_db_value(self, key: str) -> int | float | None:
        """Look up a runtime config override from the database."""
        assert self._db is not None  # Caller guards with `if self._db is not None`
        from sqlmodel import Session, select

        from penny.database.models import RuntimeConfig

        with Session(self._db.engine) as session:
            result = session.exec(select(RuntimeConfig).where(RuntimeConfig.key == key)).first()

        if result is None:
            return None

        param = RUNTIME_CONFIG_PARAMS[key]
        try:
            return param.validator(result.value)
        except ValueError:
            return None

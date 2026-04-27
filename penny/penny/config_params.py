"""Runtime configuration parameter definitions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from penny.database import Database

# Auto-populated by ConfigParam.__post_init__
RUNTIME_CONFIG_PARAMS: dict[str, ConfigParam] = {}

# Group names (display order)
GROUP_CHAT = "Chat"
GROUP_THINKING = "Thinking"
GROUP_HISTORY = "History"
GROUP_NOTIFY = "Notify"
GROUP_EMAIL = "Email"

# Ordered list for display
CONFIG_GROUPS: list[str] = [
    GROUP_CHAT,
    GROUP_THINKING,
    GROUP_HISTORY,
    GROUP_NOTIFY,
    GROUP_EMAIL,
]


@dataclass
class ConfigParam:
    """Definition of a runtime-configurable parameter.

    Automatically registers itself into RUNTIME_CONFIG_PARAMS on creation.
    """

    key: str
    description: str
    type: type  # int, float, or str
    default: int | float | str  # Default value (single source of truth)
    validator: Callable[[str], int | float | str]  # Parses and validates value from string
    group: str = GROUP_CHAT  # Display group for /config listing

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


def _validate_non_empty_string(value: str) -> str:
    """Validate non-empty string."""
    stripped = value.strip()
    if not stripped:
        raise ValueError("must be a non-empty string")
    return stripped


DOMAIN_MODE_RESTRICT = "restrict"
DOMAIN_MODE_ALLOW_ALL = "allow_all"
_VALID_DOMAIN_MODES = {DOMAIN_MODE_RESTRICT, DOMAIN_MODE_ALLOW_ALL}


def _validate_domain_mode(value: str) -> str:
    """Validate domain permission mode."""
    stripped = value.strip().lower()
    if stripped not in _VALID_DOMAIN_MODES:
        raise ValueError(f"must be one of: {', '.join(sorted(_VALID_DOMAIN_MODES))}")
    return stripped


def _validate_unit_float(value: str) -> float:
    """Validate float in (0.0, 1.0] range for similarity thresholds."""
    try:
        parsed = float(value)
    except ValueError as e:
        raise ValueError("must be a number between 0 and 1") from e

    if not (0.0 < parsed <= 1.0):
        raise ValueError("must be a number between 0 and 1")

    return parsed


# ── Chat — foreground conversation, retrieval context, and browser ───────────

ConfigParam(
    key="MESSAGE_MAX_STEPS",
    description="Max agent loop steps per message",
    type=int,
    default=8,
    validator=_validate_positive_int,
    group=GROUP_CHAT,
)

ConfigParam(
    key="CHAT_MAX_QUERIES",
    description="Max parallel queries per chat tool call",
    type=int,
    default=3,
    validator=_validate_positive_int,
    group=GROUP_CHAT,
)

ConfigParam(
    key="MESSAGE_CONTEXT_LIMIT",
    description="Max recent conversation messages injected into message context",
    type=int,
    default=20,
    validator=_validate_positive_int,
    group=GROUP_CHAT,
)

ConfigParam(
    key="SEARCH_URL",
    description="Base URL for text searches (encoded query is appended)",
    type=str,
    default="https://duckduckgo.com/?q=",
    validator=_validate_non_empty_string,
    group=GROUP_CHAT,
)

ConfigParam(
    key="DOMAIN_PERMISSION_MODE",
    description="Domain mode: restrict (prompt) or allow_all (auto-allow unknown)",
    type=str,
    default=DOMAIN_MODE_RESTRICT,
    validator=_validate_domain_mode,
    group=GROUP_CHAT,
)

# ── Thinking — inner monologue ───────────────────────────────────────────────

ConfigParam(
    key="INNER_MONOLOGUE_INTERVAL",
    description="Interval in seconds between inner monologue cycles",
    type=float,
    default=1200.0,
    validator=_validate_positive_float,
    group=GROUP_THINKING,
)

ConfigParam(
    key="INNER_MONOLOGUE_MAX_STEPS",
    description="Max thinking loop steps per inner monologue cycle",
    type=int,
    default=5,
    validator=_validate_positive_int,
    group=GROUP_THINKING,
)

ConfigParam(
    key="INNER_MONOLOGUE_MAX_QUERIES",
    description="Max parallel queries per thinking tool call",
    type=int,
    default=3,
    validator=_validate_positive_int,
    group=GROUP_THINKING,
)

ConfigParam(
    key="THOUGHT_DEDUP_EMBEDDING_THRESHOLD",
    description="Content embedding similarity threshold for thought dedup (0-1)",
    type=float,
    default=0.70,
    validator=_validate_unit_float,
    group=GROUP_THINKING,
)

ConfigParam(
    key="THOUGHT_DEDUP_TCR_THRESHOLD",
    description="Title token containment ratio threshold for thought dedup (0-1)",
    type=float,
    default=0.50,
    validator=_validate_unit_float,
    group=GROUP_THINKING,
)

ConfigParam(
    key="MAX_UNNOTIFIED_THOUGHTS",
    description="Max unnotified thoughts before thinking agent pauses",
    type=int,
    default=20,
    validator=_validate_positive_int,
    group=GROUP_THINKING,
)

ConfigParam(
    key="FREE_THINKING_PROBABILITY",
    description="Target ratio of free-exploration thoughts (0-1, remainder is seeded)",
    type=float,
    default=0.2,
    validator=_validate_unit_float,
    group=GROUP_THINKING,
)

# ── History — background preference and knowledge extraction ─────────────────

ConfigParam(
    key="HISTORY_INTERVAL",
    description="Interval in seconds between history agent runs (preferences + knowledge)",
    type=float,
    default=900.0,
    validator=_validate_positive_float,
    group=GROUP_HISTORY,
)

ConfigParam(
    key="PREFERENCE_DEDUP_EMBEDDING_THRESHOLD",
    description="Embedding similarity threshold for preference deduplication",
    type=float,
    default=0.85,
    validator=_validate_unit_float,
    group=GROUP_HISTORY,
)

ConfigParam(
    key="PREFERENCE_DEDUP_TCR_THRESHOLD",
    description="Token containment ratio threshold for preference deduplication",
    type=float,
    default=0.6,
    validator=_validate_unit_float,
    group=GROUP_HISTORY,
)

ConfigParam(
    key="PREFERENCE_MENTION_THRESHOLD",
    description=(
        "Minimum mention count for a preference to qualify as a thinking"
        " seed and for sentiment scoring"
    ),
    type=int,
    default=2,
    validator=_validate_positive_int,
    group=GROUP_HISTORY,
)


ConfigParam(
    key="EMBEDDING_BACKFILL_BATCH_LIMIT",
    description="Max items per embedding backfill cycle (preferences, thoughts, messages)",
    type=int,
    default=50,
    validator=_validate_positive_int,
    group=GROUP_HISTORY,
)

# ── Notify — notification outreach and idle timing ───────────────────────────

ConfigParam(
    key="IDLE_SECONDS",
    description="Seconds of silence before background agents become eligible",
    type=float,
    default=60.0,
    validator=_validate_positive_float,
    group=GROUP_NOTIFY,
)

ConfigParam(
    key="NOTIFY_CHECK_INTERVAL",
    description="Interval in seconds between notification check cycles",
    type=float,
    default=300.0,
    validator=_validate_positive_float,
    group=GROUP_NOTIFY,
)

ConfigParam(
    key="NOTIFY_COOLDOWN_MIN",
    description="Initial cooldown in seconds between autonomous messages",
    type=float,
    default=600.0,
    validator=_validate_positive_float,
    group=GROUP_NOTIFY,
)

ConfigParam(
    key="NOTIFY_COOLDOWN_MAX",
    description="Max cooldown in seconds (ceiling for exponential backoff)",
    type=float,
    default=5400.0,
    validator=_validate_positive_float,
    group=GROUP_NOTIFY,
)

ConfigParam(
    key="NOTIFY_CANDIDATES",
    description="Number of candidate messages to generate per notification cycle",
    type=int,
    default=5,
    validator=_validate_positive_int,
    group=GROUP_NOTIFY,
)

# ── Email — email tool settings ──────────────────────────────────────────────

ConfigParam(
    key="EMAIL_BODY_MAX_LENGTH",
    description="Max character length for email body content",
    type=int,
    default=4000,
    validator=_validate_positive_int,
    group=GROUP_EMAIL,
)

ConfigParam(
    key="EMAIL_SEARCH_LIMIT",
    description="Max email results returned by the search_emails tool",
    type=int,
    default=10,
    validator=_validate_positive_int,
    group=GROUP_EMAIL,
)

ConfigParam(
    key="EMAIL_LIST_LIMIT",
    description="Max email results returned by the list_emails tool",
    type=int,
    default=10,
    validator=_validate_positive_int,
    group=GROUP_EMAIL,
)

ConfigParam(
    key="JMAP_REQUEST_TIMEOUT",
    description="Timeout in seconds for email API requests",
    type=float,
    default=30.0,
    validator=_validate_positive_float,
    group=GROUP_EMAIL,
)


class RuntimeParams:
    """Accessor for runtime-configurable parameters.

    Lookup chain: DB override → env override → ConfigParam.default.
    Supports attribute access with uppercase keys: config.runtime.IDLE_SECONDS
    """

    def __init__(
        self,
        db: Database | None = None,
        env_overrides: dict[str, Any] | None = None,
    ) -> None:
        self._db = db
        self._env_overrides = env_overrides or {}

    def __getattr__(self, name: str) -> Any:
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

    def _get_db_value(self, key: str) -> Any:
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

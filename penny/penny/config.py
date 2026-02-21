"""Configuration management for Penny."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from penny.config_params import RUNTIME_CONFIG_PARAMS

if TYPE_CHECKING:
    from penny.database import Database


@dataclass
class Config:
    """Application configuration loaded from .env file."""

    # Channel type: "signal" or "discord"
    channel_type: str

    # Signal configuration (required if channel_type is "signal")
    signal_number: str | None
    signal_api_url: str

    # Discord configuration (required if channel_type is "discord")
    discord_bot_token: str | None
    discord_channel_id: str | None

    # Ollama configuration
    ollama_api_url: str
    ollama_foreground_model: str  # Fast model for user-facing messages
    ollama_background_model: str  # Smart model for background tasks

    # API keys
    perplexity_api_key: str | None

    # Logging configuration
    log_level: str

    # Database configuration
    db_path: str

    # Optional fields with defaults
    log_file: str | None = None
    ollama_vision_model: str | None = None  # Vision model for image understanding
    ollama_image_model: str | None = None  # Image generation model (e.g., x/z-image-turbo)
    ollama_embedding_model: str | None = None  # Embedding model (e.g., nomic-embed-text)

    # GitHub App Configuration (optional, needed for /bug command)
    github_app_id: str | None = None
    github_app_private_key_path: str | None = None
    github_app_installation_id: str | None = None

    # Agent runtime configuration
    message_max_steps: int = 5

    # Ollama retry configuration
    ollama_max_retries: int = 3
    ollama_retry_delay: float = 0.5

    # Tool execution timeout (seconds)
    tool_timeout: float = 60.0

    # Scheduler tick interval (seconds) — how often the background scheduler checks
    scheduler_tick_interval: float = 1.0

    # Global idle threshold for background tasks
    idle_seconds: float = 300.0

    # Periodic maintenance interval while idle
    maintenance_interval_seconds: float = 300.0

    # Learn loop configuration
    learn_loop_interval: float = 300.0

    # Extraction pipeline thresholds
    extraction_entity_semantic_threshold: float = 0.58
    extraction_prefilter_similarity_threshold: float = 0.2
    extraction_prefilter_min_count: int = 20
    extraction_entity_dedup_tcr_threshold: float = 0.75
    extraction_entity_dedup_embedding_threshold: float = 0.85
    extraction_fact_dedup_similarity_threshold: float = 0.85
    extraction_min_message_length: int = 20

    # Fact discovery notification settings
    notification_initial_backoff: float = 60.0
    notification_max_backoff: float = 3600.0
    notification_min_length: int = 75

    # Learn loop tuning
    learn_enrichment_fact_threshold: int = 5
    learn_staleness_days: float = 7.0
    learn_min_interest_score: float = 0.1
    learn_recent_days: float = 1.0

    # Fastmail JMAP configuration (optional, enables /email command)
    fastmail_api_token: str | None = None
    email_max_steps: int = 5

    _db: Database | None = None

    @classmethod
    def load(cls, db: Database | None = None) -> Config:
        """Load configuration from .env file."""
        # Load .env file from project root or /penny/.env in container
        env_paths = [
            Path.cwd() / ".env",
            Path("/penny/.env"),
        ]

        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path)
                break

        # Determine channel type based on which credentials are configured
        signal_number = os.getenv("SIGNAL_NUMBER")
        discord_bot_token = os.getenv("DISCORD_BOT_TOKEN")
        discord_channel_id = os.getenv("DISCORD_CHANNEL_ID")

        # Explicit channel type or auto-detect
        channel_type = os.getenv("CHANNEL_TYPE", "").lower()
        if not channel_type:
            # Auto-detect based on which credentials are present
            has_discord = (
                discord_bot_token
                and discord_bot_token != "your-bot-token-here"
                and discord_channel_id
            )
            has_signal = signal_number and signal_number != "+1234567890"

            if has_discord and not has_signal:
                channel_type = "discord"
            elif has_signal:
                channel_type = "signal"
            else:
                raise ValueError(
                    "No channel configured. Set either SIGNAL_NUMBER or "
                    "DISCORD_BOT_TOKEN + DISCORD_CHANNEL_ID in .env"
                )

        # Validate required fields for the selected channel
        if channel_type == "signal" and not signal_number:
            raise ValueError("SIGNAL_NUMBER is required for Signal channel")
        if channel_type == "discord":
            if not discord_bot_token or discord_bot_token == "your-bot-token-here":
                raise ValueError(
                    "DISCORD_BOT_TOKEN is required for Discord channel. "
                    "Get your bot token from https://discord.com/developers/applications"
                )
            if not discord_channel_id:
                raise ValueError("DISCORD_CHANNEL_ID is required for Discord channel")

        # Optional fields with defaults
        signal_api_url = os.getenv("SIGNAL_API_URL", "http://localhost:8080")
        ollama_api_url = os.getenv("OLLAMA_API_URL", "http://host.docker.internal:11434")
        ollama_foreground_model = os.getenv("OLLAMA_FOREGROUND_MODEL", "gpt-oss:20b")
        # Background model defaults to foreground model if not specified
        ollama_background_model = os.getenv("OLLAMA_BACKGROUND_MODEL", ollama_foreground_model)
        ollama_vision_model = os.getenv("OLLAMA_VISION_MODEL")  # Optional
        ollama_image_model = os.getenv("OLLAMA_IMAGE_MODEL")  # Optional
        ollama_embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL")  # Optional
        perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")  # Optional
        log_level = os.getenv("LOG_LEVEL", "INFO")
        db_path = os.getenv("DB_PATH", "/penny/data/penny.db")
        log_file = os.getenv("LOG_FILE")  # Optional, defaults to None

        # GitHub App configuration (optional, needed for /bug command)
        github_app_id = os.getenv("GITHUB_APP_ID")  # Optional
        github_app_private_key_path = os.getenv("GITHUB_APP_PRIVATE_KEY_PATH")  # Optional
        github_app_installation_id = os.getenv("GITHUB_APP_INSTALLATION_ID")  # Optional

        # Fastmail JMAP configuration (optional, needed for /email command)
        fastmail_api_token = os.getenv("FASTMAIL_API_TOKEN")  # Optional

        # Global idle threshold for all background tasks
        idle_seconds = float(os.getenv("IDLE_SECONDS", "300"))

        # Periodic maintenance interval
        maintenance_interval_seconds = float(os.getenv("MAINTENANCE_INTERVAL_SECONDS", "300"))

        # Tool execution timeout
        tool_timeout = float(os.getenv("TOOL_TIMEOUT", "60.0"))

        # Extraction pipeline thresholds
        extraction_entity_semantic_threshold = float(
            os.getenv("EXTRACTION_ENTITY_SEMANTIC_THRESHOLD", "0.58")
        )
        extraction_prefilter_similarity_threshold = float(
            os.getenv("EXTRACTION_PREFILTER_SIMILARITY_THRESHOLD", "0.2")
        )
        extraction_prefilter_min_count = int(os.getenv("EXTRACTION_PREFILTER_MIN_COUNT", "20"))
        extraction_entity_dedup_tcr_threshold = float(
            os.getenv("EXTRACTION_ENTITY_DEDUP_TCR_THRESHOLD", "0.75")
        )
        extraction_entity_dedup_embedding_threshold = float(
            os.getenv("EXTRACTION_ENTITY_DEDUP_EMBEDDING_THRESHOLD", "0.85")
        )
        extraction_fact_dedup_similarity_threshold = float(
            os.getenv("EXTRACTION_FACT_DEDUP_SIMILARITY_THRESHOLD", "0.85")
        )
        extraction_min_message_length = int(os.getenv("EXTRACTION_MIN_MESSAGE_LENGTH", "20"))

        # Fact discovery notification settings
        notification_initial_backoff = float(os.getenv("NOTIFICATION_INITIAL_BACKOFF", "60.0"))
        notification_max_backoff = float(os.getenv("NOTIFICATION_MAX_BACKOFF", "3600.0"))
        notification_min_length = int(os.getenv("NOTIFICATION_MIN_LENGTH", "75"))

        # Learn loop tuning
        learn_enrichment_fact_threshold = int(os.getenv("LEARN_ENRICHMENT_FACT_THRESHOLD", "5"))
        learn_staleness_days = float(os.getenv("LEARN_STALENESS_DAYS", "7.0"))
        learn_min_interest_score = float(os.getenv("LEARN_MIN_INTEREST_SCORE", "0.1"))
        learn_recent_days = float(os.getenv("LEARN_RECENT_DAYS", "1.0"))

        config = cls(
            channel_type=channel_type,
            signal_number=signal_number,
            signal_api_url=signal_api_url,
            discord_bot_token=discord_bot_token,
            discord_channel_id=discord_channel_id,
            ollama_api_url=ollama_api_url,
            ollama_foreground_model=ollama_foreground_model,
            ollama_background_model=ollama_background_model,
            ollama_vision_model=ollama_vision_model,
            ollama_image_model=ollama_image_model,
            ollama_embedding_model=ollama_embedding_model,
            perplexity_api_key=perplexity_api_key,
            github_app_id=github_app_id,
            github_app_private_key_path=github_app_private_key_path,
            github_app_installation_id=github_app_installation_id,
            log_level=log_level,
            db_path=db_path,
            log_file=log_file,
            tool_timeout=tool_timeout,
            idle_seconds=idle_seconds,
            maintenance_interval_seconds=maintenance_interval_seconds,
            fastmail_api_token=fastmail_api_token,
            extraction_entity_semantic_threshold=extraction_entity_semantic_threshold,
            extraction_prefilter_similarity_threshold=extraction_prefilter_similarity_threshold,
            extraction_prefilter_min_count=extraction_prefilter_min_count,
            extraction_entity_dedup_tcr_threshold=extraction_entity_dedup_tcr_threshold,
            extraction_entity_dedup_embedding_threshold=extraction_entity_dedup_embedding_threshold,
            extraction_fact_dedup_similarity_threshold=extraction_fact_dedup_similarity_threshold,
            extraction_min_message_length=extraction_min_message_length,
            notification_initial_backoff=notification_initial_backoff,
            notification_max_backoff=notification_max_backoff,
            notification_min_length=notification_min_length,
            learn_enrichment_fact_threshold=learn_enrichment_fact_threshold,
            learn_staleness_days=learn_staleness_days,
            learn_min_interest_score=learn_min_interest_score,
            learn_recent_days=learn_recent_days,
        )

        # Store database reference for runtime config lookups
        config._db = db

        return config

    def _get_db_config(self, field_name: str) -> float | int | None:
        """
        Get a runtime config value from the database.

        Args:
            field_name: Field name to look up (lowercase with underscores)

        Returns:
            Parsed value from database, or None if not found
        """
        if self._db is None:
            return None

        # Map field name to config key (uppercase)
        key = field_name.upper()
        if key not in RUNTIME_CONFIG_PARAMS:
            return None

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
            # Invalid value in database, fall back to .env
            return None

    def __getattribute__(self, name: str) -> object:
        """
        Override attribute access to check database for runtime config overrides.

        For runtime-configurable fields, database values take precedence over .env values.
        """
        # Get the base value from the dataclass
        base_value = super().__getattribute__(name)

        # Don't intercept private attributes or methods
        if name.startswith("_") or callable(base_value):
            return base_value

        # Check if this is a runtime-configurable field
        key = name.upper()
        if key in RUNTIME_CONFIG_PARAMS:
            # Try to get value from database
            db = super().__getattribute__("_db")
            if db is not None:
                db_value = self._get_db_config(name)
                if db_value is not None:
                    return db_value

        # Fall back to .env value
        return base_value


def setup_logging(log_level: str, log_file: str | None = None) -> None:
    """
    Configure logging for the application.

    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file. If provided, logs to both file and console.
    """
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if log_file specified)
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        root_logger.info("Logging to file: %s", log_file)

    # Silence noisy third-party loggers
    for name in (
        "httpcore",
        "httpx",
        "websockets",
        "perplexity",
        "duckduckgo_search",
        "primp",
        "rquest",
        "rustls",
        "reqwest",
        "hyper_util",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

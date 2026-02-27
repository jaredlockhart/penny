"""Configuration management for Penny."""

from __future__ import annotations

import contextlib
import logging
import logging.handlers
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from penny.config_params import RUNTIME_CONFIG_PARAMS, RuntimeParams

if TYPE_CHECKING:
    from penny.database import Database


def _load_dotenv() -> None:
    """Load .env file from project root or container path."""
    env_paths = [
        Path.cwd() / ".env",
        Path("/penny/.env"),
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            break


def _detect_channel_type() -> str:
    """Detect or read channel type from environment."""
    signal_number = os.getenv("SIGNAL_NUMBER")
    discord_bot_token = os.getenv("DISCORD_BOT_TOKEN")
    discord_channel_id = os.getenv("DISCORD_CHANNEL_ID")

    channel_type = os.getenv("CHANNEL_TYPE", "").lower()
    if channel_type:
        return channel_type

    has_discord = (
        discord_bot_token and discord_bot_token != "your-bot-token-here" and discord_channel_id
    )
    has_signal = signal_number and signal_number != "+1234567890"

    if has_discord and not has_signal:
        return "discord"
    if has_signal:
        return "signal"
    raise ValueError(
        "No channel configured. Set either SIGNAL_NUMBER or "
        "DISCORD_BOT_TOKEN + DISCORD_CHANNEL_ID in .env"
    )


def _validate_channel_config(channel_type: str) -> None:
    """Validate required fields for the selected channel type."""
    if channel_type == "signal" and not os.getenv("SIGNAL_NUMBER"):
        raise ValueError("SIGNAL_NUMBER is required for Signal channel")
    if channel_type == "discord":
        discord_bot_token = os.getenv("DISCORD_BOT_TOKEN")
        if not discord_bot_token or discord_bot_token == "your-bot-token-here":
            raise ValueError(
                "DISCORD_BOT_TOKEN is required for Discord channel. "
                "Get your bot token from https://discord.com/developers/applications"
            )
        if not os.getenv("DISCORD_CHANNEL_ID"):
            raise ValueError("DISCORD_CHANNEL_ID is required for Discord channel")


def _collect_env_vars(channel_type: str) -> dict:
    """Read all config environment variables and return as constructor kwargs."""
    ollama_foreground_model = os.getenv("OLLAMA_FOREGROUND_MODEL", "gpt-oss:20b")
    return {
        "channel_type": channel_type,
        "signal_number": os.getenv("SIGNAL_NUMBER"),
        "signal_api_url": os.getenv("SIGNAL_API_URL", "http://localhost:8080"),
        "discord_bot_token": os.getenv("DISCORD_BOT_TOKEN"),
        "discord_channel_id": os.getenv("DISCORD_CHANNEL_ID"),
        "ollama_api_url": os.getenv("OLLAMA_API_URL", "http://host.docker.internal:11434"),
        "ollama_foreground_model": ollama_foreground_model,
        "ollama_background_model": os.getenv("OLLAMA_BACKGROUND_MODEL", ollama_foreground_model),
        "ollama_vision_model": os.getenv("OLLAMA_VISION_MODEL"),
        "ollama_image_model": os.getenv("OLLAMA_IMAGE_MODEL"),
        "ollama_embedding_model": os.getenv("OLLAMA_EMBEDDING_MODEL"),
        "perplexity_api_key": os.getenv("PERPLEXITY_API_KEY"),
        "serper_api_key": os.getenv("SERPER_API_KEY"),
        "github_app_id": os.getenv("GITHUB_APP_ID"),
        "github_app_private_key_path": os.getenv("GITHUB_APP_PRIVATE_KEY_PATH"),
        "github_app_installation_id": os.getenv("GITHUB_APP_INSTALLATION_ID"),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "db_path": os.getenv("DB_PATH", "/penny/data/penny/penny.db"),
        "log_file": os.getenv("LOG_FILE"),
        "log_max_bytes": int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024))),
        "log_backup_count": int(os.getenv("LOG_BACKUP_COUNT", "5")),
        "tool_timeout": float(os.getenv("TOOL_TIMEOUT", "60.0")),
        "fastmail_api_token": os.getenv("FASTMAIL_API_TOKEN"),
        "news_api_key": os.getenv("NEWS_API_KEY"),
    }


def _build_runtime_params(db: Database | None) -> RuntimeParams:
    """Build runtime params with env overrides."""
    env_overrides: dict[str, int | float] = {}
    for key, param in RUNTIME_CONFIG_PARAMS.items():
        env_val = os.getenv(key)
        if env_val is not None:
            with contextlib.suppress(ValueError):
                env_overrides[key] = param.validator(env_val)
    return RuntimeParams(db=db, env_overrides=env_overrides)


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
    log_max_bytes: int = 10 * 1024 * 1024  # 10 MB
    log_backup_count: int = 5
    ollama_vision_model: str | None = None  # Vision model for image understanding
    ollama_image_model: str | None = None  # Image generation model (e.g., x/z-image-turbo)
    ollama_embedding_model: str | None = None  # Embedding model (e.g., nomic-embed-text)
    serper_api_key: str | None = None  # Serper API key for image search

    # GitHub App Configuration (optional, needed for /bug command)
    github_app_id: str | None = None
    github_app_private_key_path: str | None = None
    github_app_installation_id: str | None = None

    # Ollama retry configuration
    ollama_max_retries: int = 3
    ollama_retry_delay: float = 0.5

    # Tool execution timeout (seconds)
    tool_timeout: float = 60.0

    # Scheduler tick interval (seconds)
    scheduler_tick_interval: float = 1.0

    # Fastmail JMAP configuration (optional, enables /email command)
    fastmail_api_token: str | None = None
    email_max_steps: int = 5

    # News API configuration (optional, enables /follow command and event monitoring)
    news_api_key: str | None = None

    # Runtime-configurable params (DB override → env override → default)
    runtime: RuntimeParams = field(default_factory=RuntimeParams)

    @classmethod
    def load(cls, db: Database | None = None) -> Config:
        """Load configuration from .env file."""
        _load_dotenv()
        channel_type = _detect_channel_type()
        _validate_channel_config(channel_type)
        return cls(**_collect_env_vars(channel_type), runtime=_build_runtime_params(db))


def setup_logging(
    log_level: str,
    log_file: str | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """
    Configure logging for the application.

    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file. If provided, logs to both file and console.
        max_bytes: Maximum log file size in bytes before rotation (default 10 MB).
        backup_count: Number of rotated backup files to keep (default 5).
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

    # Rotating file handler (if log_file specified)
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        root_logger.info("Logging to file: %s", log_file)

    # Silence noisy third-party loggers
    for name in (
        "httpcore",
        "httpx",
        "websockets",
        "perplexity",
        "primp",
        "rquest",
        "rustls",
        "reqwest",
        "hyper_util",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

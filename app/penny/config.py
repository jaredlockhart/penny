"""Configuration management for Penny."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


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

    # Agent runtime configuration
    message_max_steps: int = 5

    # Ollama retry configuration
    ollama_max_retries: int = 3
    ollama_retry_delay: float = 0.5

    # Global idle threshold for background tasks
    idle_seconds: float = 300.0

    # Spontaneous followup (random delay after idle)
    followup_min_seconds: float = 3600.0
    followup_max_seconds: float = 7200.0

    # Discovery (random delay after idle)
    discovery_min_seconds: float = 7200.0
    discovery_max_seconds: float = 14400.0

    @classmethod
    def load(cls) -> "Config":
        """Load configuration from .env file."""
        # Load .env file from project root or /app/.env in container
        env_paths = [
            Path.cwd() / ".env",
            Path("/app/.env"),
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
        perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")  # Optional
        log_level = os.getenv("LOG_LEVEL", "INFO")
        db_path = os.getenv("DB_PATH", "/app/data/penny.db")
        log_file = os.getenv("LOG_FILE")  # Optional, defaults to None

        # Global idle threshold for all background tasks
        idle_seconds = float(os.getenv("IDLE_SECONDS", "300"))

        # Followup timing (random delay after idle)
        followup_min_seconds = float(os.getenv("FOLLOWUP_MIN_SECONDS", "3600"))
        followup_max_seconds = float(os.getenv("FOLLOWUP_MAX_SECONDS", "7200"))

        # Discovery timing (random delay after idle)
        discovery_min_seconds = float(os.getenv("DISCOVERY_MIN_SECONDS", "7200"))
        discovery_max_seconds = float(os.getenv("DISCOVERY_MAX_SECONDS", "14400"))

        return cls(
            channel_type=channel_type,
            signal_number=signal_number,
            signal_api_url=signal_api_url,
            discord_bot_token=discord_bot_token,
            discord_channel_id=discord_channel_id,
            ollama_api_url=ollama_api_url,
            ollama_foreground_model=ollama_foreground_model,
            ollama_background_model=ollama_background_model,
            perplexity_api_key=perplexity_api_key,
            log_level=log_level,
            db_path=db_path,
            log_file=log_file,
            idle_seconds=idle_seconds,
            followup_min_seconds=followup_min_seconds,
            followup_max_seconds=followup_max_seconds,
            discovery_min_seconds=discovery_min_seconds,
            discovery_max_seconds=discovery_max_seconds,
        )


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
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

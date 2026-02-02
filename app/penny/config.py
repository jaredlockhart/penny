"""Configuration management for Penny."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Config:
    """Application configuration loaded from .env file."""

    # Signal/messaging configuration
    signal_number: str
    signal_api_url: str

    # Ollama configuration
    ollama_api_url: str
    ollama_model: str

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
    task_max_steps: int = 10
    idle_timeout_seconds: float = 5.0
    task_check_interval: float = 1.0
    conversation_history_limit: int = 200

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

        # Required fields
        signal_number = os.getenv("SIGNAL_NUMBER")
        if not signal_number:
            raise ValueError("SIGNAL_NUMBER environment variable is required")

        # Optional fields with defaults
        signal_api_url = os.getenv("SIGNAL_API_URL", "http://localhost:8080")
        ollama_api_url = os.getenv("OLLAMA_API_URL", "http://host.docker.internal:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
        perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")  # Optional
        log_level = os.getenv("LOG_LEVEL", "INFO")
        db_path = os.getenv("DB_PATH", "/app/data/penny.db")
        log_file = os.getenv("LOG_FILE")  # Optional, defaults to None

        return cls(
            signal_number=signal_number,
            signal_api_url=signal_api_url,
            ollama_api_url=ollama_api_url,
            ollama_model=ollama_model,
            perplexity_api_key=perplexity_api_key,
            log_level=log_level,
            db_path=db_path,
            log_file=log_file,
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

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        root_logger.info("Logging to file: %s", log_file)

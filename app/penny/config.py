"""Configuration management for Penny."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Config:
    """Application configuration loaded from .env file."""

    signal_number: str
    signal_api_url: str
    ollama_api_url: str
    ollama_model: str
    log_level: str
    db_path: str

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
        log_level = os.getenv("LOG_LEVEL", "INFO")
        db_path = os.getenv("DB_PATH", "/app/data/penny.db")

        return cls(
            signal_number=signal_number,
            signal_api_url=signal_api_url,
            ollama_api_url=ollama_api_url,
            ollama_model=ollama_model,
            log_level=log_level,
            db_path=db_path,
        )


def setup_logging(log_level: str) -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

"""Discord agent for Penny - Communicates via Discord using discord.py."""

import asyncio
import logging
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from penny.agent import AgentController
from penny.channels.discord import DiscordChannel
from penny.config import setup_logging
from penny.constants import SYSTEM_PROMPT, MessageDirection
from penny.database import Database
from penny.ollama import OllamaClient
from penny.tools import SearchTool, ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class DiscordConfig:
    """Discord-specific configuration."""

    bot_token: str
    channel_id: str
    ollama_api_url: str
    ollama_model: str
    db_path: str
    log_level: str
    log_file: str | None = None
    perplexity_api_key: str | None = None
    message_max_steps: int = 5
    ollama_max_retries: int = 3
    ollama_retry_delay: float = 0.5

    @classmethod
    def load(cls) -> "DiscordConfig":
        """Load Discord configuration from .env file."""
        env_paths = [
            Path.cwd() / ".env",
            Path("/app/.env"),
        ]

        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path)
                break

        bot_token = os.getenv("DISCORD_BOT_TOKEN")
        if not bot_token or bot_token == "your-bot-token-here":
            raise ValueError(
                "DISCORD_BOT_TOKEN environment variable is required. "
                "Get your bot token from https://discord.com/developers/applications"
            )

        channel_id = os.getenv("DISCORD_CHANNEL_ID")
        if not channel_id:
            raise ValueError("DISCORD_CHANNEL_ID environment variable is required")

        return cls(
            bot_token=bot_token,
            channel_id=channel_id,
            ollama_api_url=os.getenv("OLLAMA_API_URL", "http://host.docker.internal:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            db_path=os.getenv("DB_PATH", "/app/data/penny.db"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE"),
            perplexity_api_key=os.getenv("PERPLEXITY_API_KEY"),
        )


class DiscordAgent:
    """AI agent that responds via Discord and Ollama."""

    def __init__(self, config: DiscordConfig):
        """Initialize the Discord agent."""
        self.config = config

        # Initialize database
        self.db = Database(config.db_path)
        self.db.create_tables()

        # Initialize Ollama client with required parameters
        self.ollama_client = OllamaClient(
            config.ollama_api_url,
            config.ollama_model,
            db=self.db,
            max_retries=config.ollama_max_retries,
            retry_delay=config.ollama_retry_delay,
        )

        # Set up tool registry
        tool_registry = ToolRegistry()
        if config.perplexity_api_key:
            tool_registry.register(
                SearchTool(perplexity_api_key=config.perplexity_api_key, db=self.db)
            )
            logger.info("Search tool registered (Perplexity + image search)")
        else:
            logger.warning("No PERPLEXITY_API_KEY configured - agent will have no tools")

        # Initialize agent controller
        self.controller = AgentController(
            ollama_client=self.ollama_client,
            tool_registry=tool_registry,
            max_steps=config.message_max_steps,
        )

        # Create Discord channel with message callback
        self.channel = DiscordChannel(
            token=config.bot_token,
            channel_id=config.channel_id,
            on_message_callback=self.handle_message,
        )

        self.running = True

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        logger.info("Received shutdown signal, stopping agent...")
        self.running = False

    async def _typing_loop(self, interval: float = 4.0) -> None:
        """Send typing indicators on a loop until cancelled."""
        try:
            while True:
                await self.channel.send_typing(self.config.channel_id, True)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    async def handle_message(self, envelope_data: dict) -> None:
        """
        Process an incoming message from Discord.

        Args:
            envelope_data: Raw message data from Discord event
        """
        try:
            # Extract message content from channel data
            message = self.channel.extract_message(envelope_data)
            if message is None:
                return

            logger.info("Received message from %s: %s", message.sender, message.content)

            # Log incoming message
            incoming_id = self.db.log_message(
                MessageDirection.INCOMING, message.sender, message.content
            )

            # Start typing indicator loop
            typing_task = asyncio.create_task(self._typing_loop())

            try:
                # Run through agent controller
                response = await self.controller.run(
                    current_message=message.content,
                    system_prompt=SYSTEM_PROMPT,
                    history=None,
                )

                answer = (
                    response.answer.strip()
                    if response.answer
                    else "Sorry, I couldn't generate a response."
                )

                # Log outgoing message
                self.db.log_message(
                    MessageDirection.OUTGOING,
                    "penny",
                    answer,
                    parent_id=incoming_id,
                )

                # Send response to Discord
                await self.channel.send_message(
                    self.config.channel_id, answer, attachments=response.attachments or None
                )

            finally:
                typing_task.cancel()
                await self.channel.send_typing(self.config.channel_id, False)

        except Exception as e:
            logger.exception("Error handling message: %s", e)

    async def run(self) -> None:
        """Run the Discord agent."""
        logger.info("Starting Penny Discord agent...")
        logger.info("Discord channel ID: %s", self.config.channel_id)
        logger.info("Ollama model: %s", self.config.ollama_model)

        try:
            # Start the Discord client
            await self.channel.start()

            # Keep running until shutdown signal
            while self.running:
                await asyncio.sleep(1)

        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Clean shutdown of resources."""
        logger.info("Shutting down Discord agent...")
        await self.channel.close()
        await self.ollama_client.close()
        logger.info("Discord agent shutdown complete")


async def main() -> None:
    """Main entry point for Discord agent."""
    # Load configuration
    config = DiscordConfig.load()
    setup_logging(config.log_level, config.log_file)

    # Create and run agent
    agent = DiscordAgent(config)
    await agent.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Discord agent stopped by user")

"""Main agent loop for Penny."""

import asyncio
import logging
import signal
import sys
from typing import Any

from penny.agent import (
    Agent,
    DiscoveryAgent,
    FollowupAgent,
    MessageAgent,
    ProfileAgent,
    SummarizeAgent,
)
from penny.channels import MessageChannel, create_channel
from penny.config import Config, setup_logging
from penny.constants import PROFILE_PROMPT, SUMMARIZE_PROMPT, SYSTEM_PROMPT
from penny.database import Database
from penny.database.migrate import migrate
from penny.scheduler import BackgroundScheduler, DelayedSchedule, ImmediateSchedule
from penny.tools import SearchTool

logger = logging.getLogger(__name__)


class Penny:
    """AI agent powered by Ollama via an agent controller."""

    def __init__(self, config: Config, channel: MessageChannel | None = None):
        """Initialize the agent with configuration."""
        self.config = config
        self.db = Database(config.db_path)
        migrate(config.db_path)
        self.db.create_tables()

        def search_tools():
            if config.perplexity_api_key:
                return [SearchTool(perplexity_api_key=config.perplexity_api_key, db=self.db)]
            return []

        self.message_agent = MessageAgent(
            system_prompt=SYSTEM_PROMPT,
            model=config.ollama_foreground_model,
            ollama_api_url=config.ollama_api_url,
            tools=search_tools(),
            db=self.db,
            max_steps=config.message_max_steps,
            max_retries=config.ollama_max_retries,
            retry_delay=config.ollama_retry_delay,
        )

        self.followup_agent = FollowupAgent(
            system_prompt=SYSTEM_PROMPT,
            model=config.ollama_background_model,
            ollama_api_url=config.ollama_api_url,
            tools=search_tools(),
            db=self.db,
            max_steps=config.message_max_steps,
            max_retries=config.ollama_max_retries,
            retry_delay=config.ollama_retry_delay,
        )

        self.summarize_agent = SummarizeAgent(
            system_prompt=SUMMARIZE_PROMPT,
            model=config.ollama_background_model,
            ollama_api_url=config.ollama_api_url,
            tools=[],
            db=self.db,
            max_steps=1,
            max_retries=config.ollama_max_retries,
            retry_delay=config.ollama_retry_delay,
        )

        self.profile_agent = ProfileAgent(
            system_prompt=PROFILE_PROMPT,
            model=config.ollama_background_model,
            ollama_api_url=config.ollama_api_url,
            tools=[],
            db=self.db,
            max_steps=1,
            max_retries=config.ollama_max_retries,
            retry_delay=config.ollama_retry_delay,
        )

        self.discovery_agent = DiscoveryAgent(
            system_prompt=SYSTEM_PROMPT,
            model=config.ollama_background_model,
            ollama_api_url=config.ollama_api_url,
            tools=search_tools(),
            db=self.db,
            max_steps=config.message_max_steps,
            max_retries=config.ollama_max_retries,
            retry_delay=config.ollama_retry_delay,
        )

        # Create channel (needs message_agent and db)
        self.channel = channel or create_channel(
            config=config,
            message_agent=self.message_agent,
            db=self.db,
        )

        # Connect agents that send messages to channel
        self.followup_agent.set_channel(self.channel)
        self.discovery_agent.set_channel(self.channel)

        # Create schedules (priority order: summarize, profile, followup, discovery)
        schedules = [
            ImmediateSchedule(agent=self.summarize_agent),
            ImmediateSchedule(agent=self.profile_agent),
            DelayedSchedule(
                agent=self.followup_agent,
                min_delay=config.followup_min_seconds,
                max_delay=config.followup_max_seconds,
            ),
            DelayedSchedule(
                agent=self.discovery_agent,
                min_delay=config.discovery_min_seconds,
                max_delay=config.discovery_max_seconds,
            ),
        ]
        self.scheduler = BackgroundScheduler(
            schedules=schedules,
            idle_threshold=config.idle_seconds,
        )

        # Connect scheduler to channel for message notifications
        self.channel.set_scheduler(self.scheduler)

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        logger.info("Received shutdown signal, stopping agent...")
        self.scheduler.stop()

    async def run(self) -> None:
        """Run the agent."""
        logger.info("Starting Penny AI agent...")
        logger.info("Channel: %s (sender_id=%s)", self.config.channel_type, self.channel.sender_id)
        logger.info("Ollama model: %s (messages)", self.config.ollama_foreground_model)
        if self.config.ollama_background_model != self.config.ollama_foreground_model:
            logger.info("Ollama model: %s (background)", self.config.ollama_background_model)

        await self._send_startup_announcement()

        try:
            await asyncio.gather(
                self.channel.listen(),
                self.scheduler.run(),
            )
        finally:
            await self.shutdown()

    async def _send_startup_announcement(self) -> None:
        """Send a startup announcement to all known recipients."""
        try:
            senders = self.db.get_all_senders()
            if not senders:
                logger.info("No recipients found for startup announcement")
                return

            logger.info("Sending startup announcement to %d recipient(s)", len(senders))
            for sender in senders:
                try:
                    await self.channel.send_status_message(sender, "👋")
                except Exception as e:
                    logger.warning("Failed to send startup announcement to %s: %s", sender, e)
        except Exception as e:
            logger.warning("Failed to send startup announcement: %s", e)

    async def shutdown(self) -> None:
        """Clean shutdown of resources."""
        logger.info("Shutting down agent...")
        self.scheduler.stop()
        await self.channel.close()
        await Agent.close_all()
        logger.info("Agent shutdown complete")


async def main() -> None:
    """Main entry point."""
    config = Config.load()
    setup_logging(config.log_level, config.log_file)

    logger.info("Starting Penny with config:")
    logger.info("  channel_type: %s", config.channel_type)
    logger.info("  ollama_model: %s", config.ollama_foreground_model)
    logger.info("  ollama_background_model: %s", config.ollama_background_model)
    logger.info("  ollama_api_url: %s", config.ollama_api_url)
    logger.info("  idle_threshold: %.0fs", config.idle_seconds)
    logger.info(
        "  followup_delay: %.0fs-%.0fs",
        config.followup_min_seconds,
        config.followup_max_seconds,
    )
    logger.info(
        "  discovery_delay: %.0fs-%.0fs",
        config.discovery_min_seconds,
        config.discovery_max_seconds,
    )

    agent = Penny(config)
    await agent.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Agent stopped by user")
        sys.exit(0)

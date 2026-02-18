"""Main agent loop for Penny."""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Any

from penny.agents import (
    Agent,
    DiscoveryAgent,
    ExtractionPipeline,
    FollowupAgent,
    MessageAgent,
)
from penny.agents.entity_cleaner import EntityCleaner
from penny.agents.research import ResearchAgent
from penny.channels import MessageChannel, create_channel
from penny.commands import create_command_registry
from penny.config import Config, setup_logging
from penny.database import Database
from penny.database.migrate import migrate
from penny.ollama.client import OllamaClient
from penny.prompts import Prompt
from penny.scheduler import (
    AlwaysRunSchedule,
    BackgroundScheduler,
    DelayedSchedule,
    PeriodicSchedule,
)
from penny.scheduler.schedule_runner import ScheduleExecutor
from penny.startup import get_restart_message
from penny.tools import SearchTool

logger = logging.getLogger(__name__)


class Penny:
    """AI agent powered by Ollama via an agent controller."""

    def __init__(self, config: Config, channel: MessageChannel | None = None):
        """Initialize the agent with configuration."""
        self.config = config
        self.start_time = datetime.now()
        self.db = Database(config.db_path)
        migrate(config.db_path)
        self.db.create_tables()

        # Set database reference for runtime config lookups
        config._db = self.db

        def search_tools(db):
            if config.perplexity_api_key:
                return [SearchTool(perplexity_api_key=config.perplexity_api_key, db=db)]
            return []

        def create_message_agent(db):
            """Factory for creating MessageAgent with a given database."""
            return MessageAgent(
                system_prompt=Prompt.SEARCH_PROMPT,
                model=config.ollama_foreground_model,
                ollama_api_url=config.ollama_api_url,
                tools=search_tools(db),
                db=db,
                max_steps=config.message_max_steps,
                max_retries=config.ollama_max_retries,
                retry_delay=config.ollama_retry_delay,
                tool_timeout=config.tool_timeout,
                vision_model=config.ollama_vision_model,
                embedding_model=config.ollama_embedding_model,
            )

        # Create message agent for production use
        self.message_agent = create_message_agent(self.db)

        # Initialize GitHub client if configured
        github_api = None
        if (
            config.github_app_id
            and config.github_app_private_key_path
            and config.github_app_installation_id
        ):
            try:
                from pathlib import Path

                from github_api.api import GitHubAPI
                from github_api.auth import GitHubAuth

                from penny.constants import PennyConstants

                key_path = Path(config.github_app_private_key_path)
                if not key_path.is_absolute():
                    key_path = Path.cwd() / key_path

                github_auth = GitHubAuth(
                    app_id=int(config.github_app_id),
                    private_key_path=key_path,
                    installation_id=int(config.github_app_installation_id),
                )
                github_api = GitHubAPI(
                    github_auth.get_token,
                    PennyConstants.GITHUB_REPO_OWNER,
                    PennyConstants.GITHUB_REPO_NAME,
                )
                logger.info("GitHub API client initialized")
            except Exception:
                logger.exception("Failed to initialize GitHub client")

        # Create command registry with message agent factory for test command
        self.command_registry = create_command_registry(
            message_agent_factory=create_message_agent,
            github_api=github_api,
            ollama_image_model=config.ollama_image_model,
            fastmail_api_token=config.fastmail_api_token,
        )

        self.followup_agent = FollowupAgent(
            system_prompt=Prompt.SEARCH_PROMPT,
            model=config.ollama_background_model,
            ollama_api_url=config.ollama_api_url,
            tools=search_tools(self.db),
            db=self.db,
            max_steps=config.message_max_steps,
            max_retries=config.ollama_max_retries,
            retry_delay=config.ollama_retry_delay,
            tool_timeout=config.tool_timeout,
        )

        self.extraction_pipeline = ExtractionPipeline(
            system_prompt="",  # ExtractionPipeline uses ollama_client.generate() directly
            model=config.ollama_background_model,
            ollama_api_url=config.ollama_api_url,
            tools=[],
            db=self.db,
            max_steps=1,
            max_retries=config.ollama_max_retries,
            retry_delay=config.ollama_retry_delay,
            tool_timeout=config.tool_timeout,
            embedding_model=config.ollama_embedding_model,
        )

        self.entity_cleaner = EntityCleaner(
            system_prompt="",  # EntityCleaner uses ollama_client.generate() directly
            model=config.ollama_background_model,
            ollama_api_url=config.ollama_api_url,
            tools=[],
            db=self.db,
            max_steps=1,
            max_retries=config.ollama_max_retries,
            retry_delay=config.ollama_retry_delay,
            tool_timeout=config.tool_timeout,
            embedding_model=config.ollama_embedding_model,
        )

        self.discovery_agent = DiscoveryAgent(
            system_prompt=Prompt.SEARCH_PROMPT,
            model=config.ollama_background_model,
            ollama_api_url=config.ollama_api_url,
            tools=search_tools(self.db),
            db=self.db,
            max_steps=config.message_max_steps,
            max_retries=config.ollama_max_retries,
            retry_delay=config.ollama_retry_delay,
            tool_timeout=config.tool_timeout,
        )

        self.research_agent = ResearchAgent(
            config=config,
            system_prompt=Prompt.RESEARCH_PROMPT,
            model=config.ollama_background_model,
            ollama_api_url=config.ollama_api_url,
            tools=search_tools(self.db),
            db=self.db,
            max_steps=config.message_max_steps,
            max_retries=config.ollama_max_retries,
            retry_delay=config.ollama_retry_delay,
            tool_timeout=config.tool_timeout,
        )

        self.schedule_executor = ScheduleExecutor(
            system_prompt="",  # ScheduleExecutor delegates to message_agent.run()
            model=config.ollama_background_model,
            ollama_api_url=config.ollama_api_url,
            tools=[],  # Schedule executor doesn't need tools itself
            db=self.db,
            max_steps=1,  # Just executes schedules, doesn't need multi-step loop
            max_retries=config.ollama_max_retries,
            retry_delay=config.ollama_retry_delay,
            tool_timeout=config.tool_timeout,
        )

        # Create channel (needs message_agent and db)
        self.channel = channel or create_channel(
            config=config,
            message_agent=self.message_agent,
            db=self.db,
            command_registry=self.command_registry,
        )

        # Connect agents that send messages to channel
        self.followup_agent.set_channel(self.channel)
        self.discovery_agent.set_channel(self.channel)
        self.extraction_pipeline.set_channel(self.channel)
        self.research_agent.set_channel(self.channel)
        self.schedule_executor.set_channel(self.channel)

        # Create schedules (priority: schedule, research, extraction, cleaner, followup, discovery)
        # ScheduleExecutor runs every minute regardless of idle state to check for due schedules
        # ResearchAgent runs always (whenever scheduler ticks) to process in-progress research
        schedules = [
            AlwaysRunSchedule(
                agent=self.schedule_executor,
                interval=60.0,  # Check every minute for due schedules
            ),
            AlwaysRunSchedule(
                agent=self.research_agent,
                interval=config.research_schedule_interval,
            ),
            PeriodicSchedule(
                agent=self.extraction_pipeline,
                interval=config.maintenance_interval_seconds,
            ),
            PeriodicSchedule(
                agent=self.entity_cleaner,
                interval=config.maintenance_interval_seconds,
            ),
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
            tick_interval=config.scheduler_tick_interval,
        )

        # Connect scheduler to channel for message notifications
        self.channel.set_scheduler(self.scheduler)

        # Set command context on channel (must be after scheduler initialization)
        self.channel.set_command_context(
            config=config,
            channel_type=config.channel_type,
            start_time=self.start_time,
        )

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
        if self.config.ollama_vision_model:
            logger.info("Ollama model: %s (vision)", self.config.ollama_vision_model)
        if self.config.ollama_image_model:
            logger.info("Ollama model: %s (image generation)", self.config.ollama_image_model)

        # Validate channel connectivity before starting (if implemented)
        validate_fn = getattr(self.channel, "validate_connectivity", None)
        if validate_fn and callable(validate_fn):
            await validate_fn()

        await self._send_startup_announcement()
        await self._prompt_for_missing_profiles()

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

            # Create temporary Ollama client for restart message generation
            ollama_client = OllamaClient(
                api_url=self.config.ollama_api_url,
                model=self.config.ollama_foreground_model,
                db=self.db,
                max_retries=self.config.ollama_max_retries,
                retry_delay=self.config.ollama_retry_delay,
            )

            # Generate restart message
            restart_msg = await get_restart_message(ollama_client)
            await ollama_client.close()

            # Combine wave with restart message
            announcement = f"👋 {restart_msg}"

            logger.info("Sending startup announcement to %d recipient(s)", len(senders))
            for sender in senders:
                try:
                    await self.channel.send_status_message(sender, announcement)
                except Exception as e:
                    logger.warning("Failed to send startup announcement to %s: %s", sender, e)
        except Exception as e:
            logger.warning("Failed to send startup announcement: %s", e)

    async def _prompt_for_missing_profiles(self) -> None:
        """Prompt users who don't have a profile set up yet."""
        try:
            senders = self.db.get_all_senders()
            if not senders:
                logger.info("No recipients to check for missing profiles")
                return

            prompt_msg = (
                "Hey! I need to collect some basic info about you before we can chat. "
                "Please run `/profile <name> <location> <date of birth>` "
                "to set up your profile.\n\n"
                "For example: `/profile sam denver march 5 1990` 📝"
            )

            for sender in senders:
                try:
                    user_info = self.db.get_user_info(sender)
                    if not user_info:
                        logger.info("User %s has no profile, sending prompt", sender)
                        try:
                            await self.channel.send_status_message(sender, prompt_msg)
                        except Exception as e:
                            logger.warning("Failed to send profile prompt to %s: %s", sender, e)
                except Exception:
                    # Silently skip if userinfo table doesn't exist yet
                    pass
        except Exception as e:
            logger.warning("Failed to send profile prompts: %s", e)

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
    logger.info("  maintenance_interval: %.0fs", config.maintenance_interval_seconds)
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

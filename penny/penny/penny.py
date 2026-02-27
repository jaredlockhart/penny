"""Main agent loop for Penny."""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Any

from penny.agents import (
    Agent,
    EnrichAgent,
    ExtractionPipeline,
    LearnAgent,
    MessageAgent,
    NotificationAgent,
)
from penny.channels import MessageChannel, create_channel
from penny.commands import create_command_registry
from penny.config import Config, setup_logging
from penny.database import Database
from penny.database.migrate import migrate
from penny.ollama.client import OllamaClient
from penny.ollama.embeddings import build_entity_embed_text, serialize_embedding
from penny.prompts import Prompt
from penny.scheduler import (
    AlwaysRunSchedule,
    BackgroundScheduler,
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
        config.runtime._db = self.db

        # Shared Ollama model clients: foreground (fast, user-facing) and background (smart)
        self.foreground_model_client = OllamaClient(
            api_url=config.ollama_api_url,
            model=config.ollama_foreground_model,
            db=self.db,
            max_retries=config.ollama_max_retries,
            retry_delay=config.ollama_retry_delay,
        )
        self.background_model_client = OllamaClient(
            api_url=config.ollama_api_url,
            model=config.ollama_background_model,
            db=self.db,
            max_retries=config.ollama_max_retries,
            retry_delay=config.ollama_retry_delay,
        )

        # Optional model clients: vision (image understanding) and embedding (similarity)
        self.vision_model_client: OllamaClient | None = None
        if config.ollama_vision_model:
            self.vision_model_client = OllamaClient(
                api_url=config.ollama_api_url,
                model=config.ollama_vision_model,
                db=self.db,
                max_retries=config.ollama_max_retries,
                retry_delay=config.ollama_retry_delay,
            )

        self.embedding_model_client: OllamaClient | None = None
        if config.ollama_embedding_model:
            self.embedding_model_client = OllamaClient(
                api_url=config.ollama_api_url,
                model=config.ollama_embedding_model,
                db=self.db,
                max_retries=config.ollama_max_retries,
                retry_delay=config.ollama_retry_delay,
            )

        self.image_model_client: OllamaClient | None = None
        if config.ollama_image_model:
            self.image_model_client = OllamaClient(
                api_url=config.ollama_api_url,
                model=config.ollama_image_model,
                db=self.db,
                max_retries=config.ollama_max_retries,
                retry_delay=config.ollama_retry_delay,
            )

        def search_tools(db):
            if config.perplexity_api_key:
                return [
                    SearchTool(
                        perplexity_api_key=config.perplexity_api_key,
                        db=db,
                        serper_api_key=config.serper_api_key,
                        image_max_results=int(config.runtime.IMAGE_MAX_RESULTS),
                        image_download_timeout=config.runtime.IMAGE_DOWNLOAD_TIMEOUT,
                    )
                ]
            return []

        def create_message_agent(db):
            """Factory for creating MessageAgent with a given database.

            Creates its own OllamaClient because the /test command needs
            prompt logging against a separate test database.
            """
            client = OllamaClient(
                api_url=config.ollama_api_url,
                model=config.ollama_foreground_model,
                db=db,
                max_retries=config.ollama_max_retries,
                retry_delay=config.ollama_retry_delay,
            )
            return MessageAgent(
                system_prompt=Prompt.SEARCH_PROMPT,
                background_model_client=client,
                foreground_model_client=client,
                tools=search_tools(db),
                db=db,
                config=config,
                max_steps=int(config.runtime.MESSAGE_MAX_STEPS),
                tool_timeout=config.tool_timeout,
                vision_model_client=self.vision_model_client,
                embedding_model_client=self.embedding_model_client,
            )

        # Create message agent for production use
        self.message_agent = MessageAgent(
            system_prompt=Prompt.SEARCH_PROMPT,
            background_model_client=self.foreground_model_client,
            foreground_model_client=self.foreground_model_client,
            tools=search_tools(self.db),
            db=self.db,
            config=config,
            max_steps=int(config.runtime.MESSAGE_MAX_STEPS),
            tool_timeout=config.tool_timeout,
            vision_model_client=self.vision_model_client,
            embedding_model_client=self.embedding_model_client,
        )

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

        # Shared search tool for commands and agents that call SearchTool directly
        shared_search_tools = search_tools(self.db)
        shared_search_tool = shared_search_tools[0] if shared_search_tools else None

        # Create command registry with message agent factory for test command
        self.command_registry = create_command_registry(
            message_agent_factory=create_message_agent,
            github_api=github_api,
            image_model_client=self.image_model_client,
            fastmail_api_token=config.fastmail_api_token,
        )

        # Learn agent processes /learn prompts one search step at a time.
        # Scheduled after extraction; gated by unextracted learn search logs so topics
        # flow through the pipeline one at a time (search → extract → notify).
        self.learn_agent = LearnAgent(
            search_tool=shared_search_tool,
            system_prompt="",
            background_model_client=self.background_model_client,
            foreground_model_client=self.foreground_model_client,
            tools=[],
            db=self.db,
            max_steps=1,
            tool_timeout=config.tool_timeout,
            config=config,
        )

        self.extraction_pipeline = ExtractionPipeline(
            system_prompt="",  # No agent-specific prompt; identity added by _build_messages
            background_model_client=self.background_model_client,
            foreground_model_client=self.foreground_model_client,
            tools=[],
            db=self.db,
            max_steps=1,
            tool_timeout=config.tool_timeout,
            embedding_model_client=self.embedding_model_client,
            config=config,
        )

        self.notification_agent = NotificationAgent(
            system_prompt="",  # No agent-specific prompt; identity added by _build_messages
            background_model_client=self.background_model_client,
            foreground_model_client=self.foreground_model_client,
            tools=[],
            db=self.db,
            max_steps=1,
            tool_timeout=config.tool_timeout,
            config=config,
        )

        # Enrich agent autonomously researches entities based on interest scores.
        # Lowest priority — only runs when notification, extraction, and learn have no work.
        self.enrich_agent = EnrichAgent(
            search_tool=shared_search_tool,
            system_prompt="",  # No agent-specific prompt; identity added by _build_messages
            background_model_client=self.background_model_client,
            foreground_model_client=self.foreground_model_client,
            tools=[],
            db=self.db,
            max_steps=1,
            tool_timeout=config.tool_timeout,
            embedding_model_client=self.embedding_model_client,
            config=config,
        )

        self.schedule_executor = ScheduleExecutor(
            system_prompt="",  # ScheduleExecutor delegates to message_agent.run()
            background_model_client=self.background_model_client,
            foreground_model_client=self.foreground_model_client,
            tools=[],  # Schedule executor doesn't need tools itself
            db=self.db,
            config=config,
            max_steps=1,  # Just executes schedules, doesn't need multi-step loop
            tool_timeout=config.tool_timeout,
        )

        # Create channel (needs message_agent and db)
        self.channel = channel or create_channel(
            config=config,
            message_agent=self.message_agent,
            db=self.db,
            command_registry=self.command_registry,
        )

        # Connect agents that send proactive messages to channel
        self.notification_agent.set_channel(self.channel)
        self.schedule_executor.set_channel(self.channel)

        # Schedules (priority: schedule executor → notification → extraction → learn → enrich)
        # Agents with no work are skipped, so lower-priority agents get a turn each tick.
        # ScheduleExecutor runs every minute regardless of idle state
        # NotificationAgent announces completed topics before new work starts
        # ExtractionPipeline processes search logs before learn creates more
        # LearnAgent runs gated by unextracted learn search logs
        # EnrichAgent runs only when all higher-priority agents have no work
        schedules = [
            AlwaysRunSchedule(
                agent=self.schedule_executor,
                interval=60.0,  # Check every minute for due schedules
            ),
            PeriodicSchedule(
                agent=self.notification_agent,
                interval=config.runtime.MAINTENANCE_INTERVAL_SECONDS,
            ),
            PeriodicSchedule(
                agent=self.extraction_pipeline,
                interval=config.runtime.MAINTENANCE_INTERVAL_SECONDS,
            ),
            PeriodicSchedule(
                agent=self.learn_agent,
                interval=config.runtime.MAINTENANCE_INTERVAL_SECONDS,
            ),
            PeriodicSchedule(
                agent=self.enrich_agent,
                interval=config.runtime.MAINTENANCE_INTERVAL_SECONDS,
            ),
        ]
        self.scheduler = BackgroundScheduler(
            schedules=schedules,
            idle_threshold=config.runtime.IDLE_SECONDS,
            tick_interval=config.scheduler_tick_interval,
        )

        # Connect scheduler to channel for message notifications
        self.channel.set_scheduler(self.scheduler)

        # Set command context on channel (must be after scheduler initialization)
        self.channel.set_command_context(
            config=config,
            channel_type=config.channel_type,
            start_time=self.start_time,
            foreground_model_client=self.foreground_model_client,
            embedding_model_client=self.embedding_model_client,
            image_model_client=self.image_model_client,
        )

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        logger.info("Received shutdown signal, stopping agent...")
        self.scheduler.stop()

    async def _validate_optional_models(self) -> None:
        """
        Check that all configured optional Ollama models are available on the host.

        Logs a warning for each model that is configured but not yet pulled.
        This surfaces misconfigured model names immediately at startup rather than
        letting background tasks fail repeatedly with opaque 404 errors.
        """
        optional_models: list[tuple[str, str]] = []
        if self.config.ollama_vision_model:
            optional_models.append((self.config.ollama_vision_model, "OLLAMA_VISION_MODEL"))
        if self.config.ollama_image_model:
            optional_models.append((self.config.ollama_image_model, "OLLAMA_IMAGE_MODEL"))
        if self.config.ollama_embedding_model:
            optional_models.append((self.config.ollama_embedding_model, "OLLAMA_EMBEDDING_MODEL"))

        if not optional_models:
            return

        available = await self.foreground_model_client.list_models()
        for model_name, env_var in optional_models:
            # Strip tag for comparison since some models report without tag
            base_name = model_name.split(":")[0]
            is_available = any(m == model_name or m.split(":")[0] == base_name for m in available)
            if not is_available:
                logger.warning(
                    "Configured model %r (%s) is not available on the Ollama host. "
                    "Run `ollama pull %s` to download it. "
                    "Features depending on this model will be disabled until it is pulled.",
                    model_name,
                    env_var,
                    model_name,
                )

    async def _backfill_all_embeddings(self) -> None:
        """Backfill all missing embeddings at startup.

        Loops through facts and entities with null embeddings in batches
        until none remain. This ensures a clean slate after switching
        embedding models without waiting for the per-cycle backfill.
        """
        if not self.embedding_model_client:
            return

        batch_limit = int(self.config.runtime.EMBEDDING_BACKFILL_BATCH_LIMIT)

        # Backfill facts
        total_facts = 0
        while True:
            facts = self.db.get_facts_without_embeddings(limit=batch_limit)
            if not facts:
                break
            try:
                fact_texts = [f.content for f in facts]
                vecs = await self.embedding_model_client.embed(fact_texts)
                for fact, vec, text in zip(facts, vecs, fact_texts, strict=True):
                    assert fact.id is not None
                    self.db.update_fact_embedding(fact.id, serialize_embedding(vec))
                    logger.info("Embedded fact %d: %s", fact.id, text[:120])
                total_facts += len(facts)
            except Exception as e:
                logger.warning("Startup embedding backfill failed for facts: %s", e)
                break

        # Backfill entities
        total_entities = 0
        while True:
            entities = self.db.get_entities_without_embeddings(limit=batch_limit)
            if not entities:
                break
            try:
                texts = []
                for entity in entities:
                    assert entity.id is not None
                    entity_facts = self.db.get_entity_facts(entity.id)
                    texts.append(
                        build_entity_embed_text(
                            entity.name, [f.content for f in entity_facts], entity.tagline
                        )
                    )
                vecs = await self.embedding_model_client.embed(texts)
                for entity, vec, text in zip(entities, vecs, texts, strict=True):
                    assert entity.id is not None
                    self.db.update_entity_embedding(entity.id, serialize_embedding(vec))
                    logger.info("Embedded entity %d: %s", entity.id, text[:120])
                total_entities += len(entities)
            except Exception as e:
                logger.warning("Startup embedding backfill failed for entities: %s", e)
                break

        if total_facts or total_entities:
            logger.info(
                "Startup embedding backfill complete: %d facts, %d entities",
                total_facts,
                total_entities,
            )

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

        await self._validate_optional_models()
        await self._backfill_all_embeddings()

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

            # Generate restart message
            restart_msg = await get_restart_message(self.foreground_model_client)

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
        await self.foreground_model_client.close()
        await self.background_model_client.close()
        if self.vision_model_client:
            await self.vision_model_client.close()
        if self.embedding_model_client:
            await self.embedding_model_client.close()
        if self.image_model_client:
            await self.image_model_client.close()
        logger.info("Agent shutdown complete")


async def main() -> None:
    """Main entry point."""
    config = Config.load()
    setup_logging(config.log_level, config.log_file, config.log_max_bytes, config.log_backup_count)

    logger.info("Starting Penny with config:")
    logger.info("  channel_type: %s", config.channel_type)
    logger.info("  ollama_model: %s", config.ollama_foreground_model)
    logger.info("  ollama_background_model: %s", config.ollama_background_model)
    logger.info("  ollama_api_url: %s", config.ollama_api_url)
    logger.info("  idle_threshold: %.0fs", config.runtime.IDLE_SECONDS)
    logger.info("  maintenance_interval: %.0fs", config.runtime.MAINTENANCE_INTERVAL_SECONDS)

    agent = Penny(config)
    await agent.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Agent stopped by user")
        sys.exit(0)

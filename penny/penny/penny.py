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
    EventAgent,
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
from penny.interest import HeatEngine
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
from penny.tools import SearchTool, Tool
from penny.tools.news import NewsTool

logger = logging.getLogger(__name__)


class Penny:
    """AI agent powered by Ollama via an agent controller."""

    def __init__(self, config: Config, channel: MessageChannel | None = None):
        """Initialize Penny — summary method."""
        self.config = config
        self.start_time = datetime.now()
        self._init_database(config)
        self._init_ollama_clients(config)
        self._init_agents(config)
        self._init_commands(config)
        self._init_channel(config, channel)
        self._init_scheduler(config)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _init_database(self, config: Config) -> None:
        """Set up database, run migrations, and connect to runtime config."""
        self.db = Database(config.db_path)
        migrate(config.db_path)
        self.db.create_tables()
        config.runtime._db = self.db

    def _create_ollama_client(self, model: str) -> OllamaClient:
        """Create an OllamaClient with standard configuration."""
        return OllamaClient(
            api_url=self.config.ollama_api_url,
            model=model,
            db=self.db,
            max_retries=self.config.ollama_max_retries,
            retry_delay=self.config.ollama_retry_delay,
        )

    def _init_ollama_clients(self, config: Config) -> None:
        """Create shared Ollama model clients."""
        self.foreground_model_client = self._create_ollama_client(config.ollama_foreground_model)
        self.background_model_client = self._create_ollama_client(config.ollama_background_model)
        self.vision_model_client = (
            self._create_ollama_client(config.ollama_vision_model)
            if config.ollama_vision_model
            else None
        )
        self.embedding_model_client = (
            self._create_ollama_client(config.ollama_embedding_model)
            if config.ollama_embedding_model
            else None
        )
        self.image_model_client = (
            self._create_ollama_client(config.ollama_image_model)
            if config.ollama_image_model
            else None
        )

    def _create_search_tools(self, db: Database) -> list[Tool]:
        """Build search tools list for a given database."""
        if not self.config.perplexity_api_key:
            return []
        return [
            SearchTool(
                perplexity_api_key=self.config.perplexity_api_key,
                db=db,
                serper_api_key=self.config.serper_api_key,
                image_max_results=int(self.config.runtime.IMAGE_MAX_RESULTS),
                image_download_timeout=self.config.runtime.IMAGE_DOWNLOAD_TIMEOUT,
            )
        ]

    def _create_message_agent(self, db: Database) -> MessageAgent:
        """Factory for creating MessageAgent with a given database.

        Creates its own OllamaClient because the /test command needs
        prompt logging against a separate test database.
        """
        client = OllamaClient(
            api_url=self.config.ollama_api_url,
            model=self.config.ollama_foreground_model,
            db=db,
            max_retries=self.config.ollama_max_retries,
            retry_delay=self.config.ollama_retry_delay,
        )
        return MessageAgent(
            system_prompt=Prompt.SEARCH_PROMPT,
            background_model_client=client,
            foreground_model_client=client,
            tools=self._create_search_tools(db),
            db=db,
            config=self.config,
            max_steps=int(self.config.runtime.MESSAGE_MAX_STEPS),
            tool_timeout=self.config.tool_timeout,
            vision_model_client=self.vision_model_client,
            embedding_model_client=self.embedding_model_client,
        )

    def _init_agents(self, config: Config) -> None:
        """Create message agent and background processing agents."""
        self.message_agent = MessageAgent(
            system_prompt=Prompt.SEARCH_PROMPT,
            background_model_client=self.foreground_model_client,
            foreground_model_client=self.foreground_model_client,
            tools=self._create_search_tools(self.db),
            db=self.db,
            config=config,
            max_steps=int(config.runtime.MESSAGE_MAX_STEPS),
            tool_timeout=config.tool_timeout,
            vision_model_client=self.vision_model_client,
            embedding_model_client=self.embedding_model_client,
        )
        shared_search_tools = self._create_search_tools(self.db)
        self._shared_search_tool = shared_search_tools[0] if shared_search_tools else None
        self._init_background_agents(config)

    def _background_agent_kwargs(self, config: Config) -> dict:
        """Common kwargs shared by all background processing agents."""
        return {
            "system_prompt": "",
            "background_model_client": self.background_model_client,
            "foreground_model_client": self.foreground_model_client,
            "tools": [],
            "db": self.db,
            "max_steps": 1,
            "tool_timeout": config.tool_timeout,
            "config": config,
        }

    def _init_background_agents(self, config: Config) -> None:
        """Create learn, extraction, notification, enrich, event, and schedule agents."""
        kwargs = self._background_agent_kwargs(config)
        search_tool = self._shared_search_tool
        self.learn_agent = LearnAgent(search_tool=search_tool, **kwargs)
        self.extraction_pipeline = ExtractionPipeline(
            embedding_model_client=self.embedding_model_client, **kwargs
        )
        self.heat_engine = HeatEngine(db=self.db, runtime=config.runtime)
        self.extraction_pipeline.set_heat_engine(self.heat_engine)
        self.notification_agent = NotificationAgent(**kwargs)
        self.notification_agent.set_heat_engine(self.heat_engine)
        self.enrich_agent = EnrichAgent(
            search_tool=search_tool,
            embedding_model_client=self.embedding_model_client,
            **kwargs,
        )
        self.enrich_agent.set_heat_engine(self.heat_engine)
        self.event_agent = EventAgent(
            news_tool=self._create_news_tool(config),
            embedding_model_client=self.embedding_model_client,
            **kwargs,
        )
        self.schedule_executor = ScheduleExecutor(**kwargs)

    def _create_news_tool(self, config: Config) -> NewsTool | None:
        """Create NewsTool if NEWS_API_KEY is configured."""
        if not config.news_api_key:
            return None
        return NewsTool(api_key=config.news_api_key)

    def _init_github_client(self, config: Config) -> Any:
        """Initialize GitHub API client if configured. Returns GitHubAPI or None."""
        if not (
            config.github_app_id
            and config.github_app_private_key_path
            and config.github_app_installation_id
        ):
            return None
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
            return github_api
        except Exception:
            logger.exception("Failed to initialize GitHub client")
            return None

    def _init_commands(self, config: Config) -> None:
        """Create command registry with GitHub client and message agent factory."""
        github_api = self._init_github_client(config)
        self.command_registry = create_command_registry(
            message_agent_factory=self._create_message_agent,
            github_api=github_api,
            image_model_client=self.image_model_client,
            fastmail_api_token=config.fastmail_api_token,
        )

    def _init_channel(self, config: Config, channel: MessageChannel | None) -> None:
        """Create channel and connect agents that send proactive messages."""
        self.channel = channel or create_channel(
            config=config,
            message_agent=self.message_agent,
            db=self.db,
            command_registry=self.command_registry,
        )
        self.notification_agent.set_channel(self.channel)
        self.event_agent.set_channel(self.channel)
        self.channel.set_heat_engine(self.heat_engine)
        self.schedule_executor.set_channel(self.channel)

    def _init_scheduler(self, config: Config) -> None:
        """Create background scheduler with prioritized schedules."""
        schedules = [
            AlwaysRunSchedule(agent=self.schedule_executor, interval=60.0),
            PeriodicSchedule(
                agent=self.notification_agent,
                interval=config.runtime.MAINTENANCE_INTERVAL_SECONDS,
            ),
            PeriodicSchedule(
                agent=self.extraction_pipeline,
                interval=config.runtime.MAINTENANCE_INTERVAL_SECONDS,
            ),
            PeriodicSchedule(
                agent=self.event_agent,
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
        self._connect_scheduler(config)

    def _connect_scheduler(self, config: Config) -> None:
        """Connect scheduler to channel and set command context."""
        self.channel.set_scheduler(self.scheduler)
        self.channel.set_command_context(
            config=config,
            channel_type=config.channel_type,
            start_time=self.start_time,
            foreground_model_client=self.foreground_model_client,
            embedding_model_client=self.embedding_model_client,
            image_model_client=self.image_model_client,
        )

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
        """Backfill all missing embeddings at startup."""
        if not self.embedding_model_client:
            return
        batch_limit = int(self.config.runtime.EMBEDDING_BACKFILL_BATCH_LIMIT)
        total_facts = await self._backfill_fact_embeddings(batch_limit)
        total_entities = await self._backfill_entity_embeddings(batch_limit)
        if total_facts or total_entities:
            logger.info(
                "Startup embedding backfill complete: %d facts, %d entities",
                total_facts,
                total_entities,
            )

    async def _backfill_fact_embeddings(self, batch_limit: int) -> int:
        """Backfill facts with missing embeddings. Returns count embedded."""
        assert self.embedding_model_client is not None
        total = 0
        while True:
            facts = self.db.facts.get_without_embeddings(limit=batch_limit)
            if not facts:
                break
            try:
                fact_texts = [f.content for f in facts]
                vecs = await self.embedding_model_client.embed(fact_texts)
                for fact, vec, text in zip(facts, vecs, fact_texts, strict=True):
                    assert fact.id is not None
                    self.db.facts.update_embedding(fact.id, serialize_embedding(vec))
                    logger.info("Embedded fact %d: %s", fact.id, text[:120])
                total += len(facts)
            except Exception as e:
                logger.warning("Startup embedding backfill failed for facts: %s", e)
                break
        return total

    async def _backfill_entity_embeddings(self, batch_limit: int) -> int:
        """Backfill entities with missing embeddings. Returns count embedded."""
        assert self.embedding_model_client is not None
        total = 0
        while True:
            entities = self.db.entities.get_without_embeddings(limit=batch_limit)
            if not entities:
                break
            try:
                texts = []
                for entity in entities:
                    assert entity.id is not None
                    entity_facts = self.db.facts.get_for_entity(entity.id)
                    texts.append(
                        build_entity_embed_text(
                            entity.name, [f.content for f in entity_facts], entity.tagline
                        )
                    )
                vecs = await self.embedding_model_client.embed(texts)
                for entity, vec, text in zip(entities, vecs, texts, strict=True):
                    assert entity.id is not None
                    self.db.entities.update_embedding(entity.id, serialize_embedding(vec))
                    logger.info("Embedded entity %d: %s", entity.id, text[:120])
                total += len(entities)
            except Exception as e:
                logger.warning("Startup embedding backfill failed for entities: %s", e)
                break
        return total

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
            senders = self.db.users.get_all_senders()
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
            senders = self.db.users.get_all_senders()
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
                    user_info = self.db.users.get_info(sender)
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

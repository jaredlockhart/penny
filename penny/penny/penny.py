"""Main agent loop for Penny."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Any

from penny.agents import (
    Agent,
    ChatAgent,
    HistoryAgent,
    NotifyAgent,
    ThinkingAgent,
)
from penny.channels import MessageChannel, create_channel_manager
from penny.channels.browser import BrowserChannel
from penny.channels.manager import ChannelManager
from penny.channels.permission_manager import PermissionManager
from penny.channels.signal.channel import SignalChannel
from penny.commands import create_command_registry
from penny.config import Config, setup_logging
from penny.constants import ChannelType, PennyConstants
from penny.database import Database
from penny.database.migrate import migrate
from penny.ollama.client import OllamaClient
from penny.ollama.embeddings import serialize_embedding
from penny.prompts import Prompt
from penny.responses import PennyResponse
from penny.scheduler import (
    AlwaysRunSchedule,
    BackgroundScheduler,
    PeriodicSchedule,
    Schedule,
)
from penny.scheduler.schedule_runner import ScheduleExecutor
from penny.startup import get_restart_message
from penny.tools import SearchTool
from penny.tools.browse_url import BrowseUrlTool
from penny.tools.fetch_news import FetchNewsTool
from penny.tools.multi import MultiTool
from penny.tools.news import NewsTool
from penny.zoho.models import ZohoCredentials

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
        self.model_client = self._create_ollama_client(config.ollama_model)
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

    def _create_search_tool(self, db: Database) -> SearchTool | None:
        """Build a search tool for a given database."""
        if not self.config.perplexity_api_key:
            return None
        return SearchTool(
            perplexity_api_key=self.config.perplexity_api_key,
            db=db,
        )

    def _create_multi_tool(
        self, search_tool: SearchTool | None, max_calls: int | None = None
    ) -> MultiTool:
        """Build a MultiTool wrapper for the given search tool."""
        calls = max_calls or int(self.config.runtime.MESSAGE_MAX_TOOL_CALLS)
        return MultiTool(
            search_tool=search_tool,
            news_tool=self._news_tool,
            max_calls=calls,
        )

    def _create_chat_agent(self, db: Database) -> ChatAgent:
        """Factory for creating ChatAgent with a given database.

        Creates its own OllamaClient because the /test command needs
        prompt logging against a separate test database.
        """
        client = OllamaClient(
            api_url=self.config.ollama_api_url,
            model=self.config.ollama_model,
            db=db,
            max_retries=self.config.ollama_max_retries,
            retry_delay=self.config.ollama_retry_delay,
        )
        search_tool = self._create_search_tool(db)
        multi_tool = self._create_multi_tool(search_tool)
        return ChatAgent(
            system_prompt=Prompt.CONVERSATION_PROMPT,
            multi_tool=multi_tool,
            model_client=client,
            tools=[],
            db=db,
            config=self.config,
            tool_timeout=self.config.tool_timeout,
            vision_model_client=self.vision_model_client,
            embedding_model_client=self.embedding_model_client,
        )

    def _create_news_tool(self, config: Config) -> FetchNewsTool | None:
        """Create FetchNewsTool if NEWS_API_KEY is configured."""
        if not config.news_api_key:
            return None
        return FetchNewsTool(news_tool=NewsTool(api_key=config.news_api_key))

    def _init_agents(self, config: Config) -> None:
        """Create message agent and background processing agents."""
        self._shared_search_tool = self._create_search_tool(self.db)
        self._news_tool = self._create_news_tool(config)
        self._multi_tool = self._create_multi_tool(self._shared_search_tool)
        self.chat_agent = ChatAgent(
            system_prompt=Prompt.CONVERSATION_PROMPT,
            multi_tool=self._multi_tool,
            model_client=self.model_client,
            tools=[],
            db=self.db,
            config=config,
            tool_timeout=config.tool_timeout,
            vision_model_client=self.vision_model_client,
            embedding_model_client=self.embedding_model_client,
        )
        self.notify_agent = NotifyAgent(
            search_tool=self._shared_search_tool,
            news_tool=self._news_tool,
            system_prompt=Prompt.NOTIFY_SYSTEM_PROMPT,
            model_client=self.model_client,
            tools=[],
            db=self.db,
            config=config,
            tool_timeout=config.tool_timeout,
            embedding_model_client=self.embedding_model_client,
        )
        self._init_background_agents(config)

    def _background_agent_kwargs(self, config: Config) -> dict:
        """Common kwargs shared by all background processing agents."""
        return {
            "system_prompt": "",
            "model_client": self.model_client,
            "tools": [],
            "db": self.db,
            "tool_timeout": config.tool_timeout,
            "config": config,
        }

    def _init_background_agents(self, config: Config) -> None:
        """Create monologue, history, and schedule agents."""
        kwargs = self._background_agent_kwargs(config)
        thinking_search_tool = self._create_search_tool(self.db)
        thinking_max_queries = int(config.runtime.INNER_MONOLOGUE_MAX_QUERIES)
        self._thinking_multi_tool = self._create_multi_tool(
            thinking_search_tool, max_calls=thinking_max_queries
        )
        self.thinking_agent = ThinkingAgent(
            multi_tool=self._thinking_multi_tool,
            embedding_model_client=self.embedding_model_client,
            **kwargs,
        )
        self.history_agent = HistoryAgent(
            embedding_model_client=self.embedding_model_client, **kwargs
        )
        self.schedule_executor = ScheduleExecutor(**kwargs)

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
        zoho_credentials = self._get_zoho_credentials(config)
        self.command_registry = create_command_registry(
            message_agent_factory=self._create_chat_agent,
            github_api=github_api,
            image_model_client=self.image_model_client,
            fastmail_api_token=config.fastmail_api_token,
            zoho_credentials=zoho_credentials,
        )

    def _get_zoho_credentials(self, config: Config) -> ZohoCredentials | None:
        """Get Zoho credentials if all required values are configured."""
        if config.zoho_api_id and config.zoho_api_secret and config.zoho_refresh_token:
            return ZohoCredentials(
                client_id=config.zoho_api_id,
                client_secret=config.zoho_api_secret,
                refresh_token=config.zoho_refresh_token,
            )
        return None

    def _init_channel(self, config: Config, channel: MessageChannel | None) -> None:
        """Create channel manager and connect agents that send notifications."""
        self.channel = channel or create_channel_manager(
            config=config,
            message_agent=self.chat_agent,
            db=self.db,
            command_registry=self.command_registry,
        )
        self.schedule_executor.set_channel(self.channel)
        self.notify_agent.set_channel(self.channel)
        self._wire_browser_tools(config)

    def _wire_browser_tools(self, config: Config) -> None:
        """Connect browser tools to agents when a browser channel is available."""
        if not isinstance(self.channel, ChannelManager):
            return
        browser_ch = self.channel.get_channel(ChannelType.BROWSER)
        if not isinstance(browser_ch, BrowserChannel):
            return

        # Wire up permission manager
        perm_mgr = PermissionManager(db=self.db, channel_manager=self.channel, config=config)
        browser_ch.set_permission_manager(perm_mgr)
        signal_ch = self.channel.get_channel(ChannelType.SIGNAL)
        if isinstance(signal_ch, SignalChannel):
            signal_ch.set_permission_manager(perm_mgr)

        # Chat agent uses MultiTool — give it a provider checked at dispatch time.
        def browse_tool_provider() -> BrowseUrlTool | None:
            if not browser_ch.has_tool_connection:
                return None
            return BrowseUrlTool(
                request_fn=browser_ch.send_tool_request,
                permission_manager=perm_mgr,
            )

        self._multi_tool.set_browse_url_provider(browse_tool_provider)
        self._thinking_multi_tool.set_browse_url_provider(browse_tool_provider)
        self.thinking_agent._on_tool_start_factory = browser_ch.make_background_tool_callback

    def _init_scheduler(self, config: Config) -> None:
        """Create background scheduler with prioritized schedules."""
        schedules: list[Schedule] = [
            AlwaysRunSchedule(agent=self.schedule_executor, interval=60.0),
            PeriodicSchedule(
                agent=self.history_agent,
                interval=lambda: config.runtime.HISTORY_INTERVAL,
                requires_idle=False,
            ),
            PeriodicSchedule(
                agent=self.notify_agent,
                interval=lambda: config.runtime.NOTIFY_CHECK_INTERVAL,
            ),
            PeriodicSchedule(
                agent=self.thinking_agent,
                interval=lambda: config.runtime.INNER_MONOLOGUE_INTERVAL,
                requires_idle=False,
            ),
        ]
        self.scheduler = BackgroundScheduler(
            schedules=schedules,
            idle_threshold=lambda: config.runtime.IDLE_SECONDS,
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
            model_client=self.model_client,
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

        available = await self.model_client.list_models()
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
        total_prefs = await self._backfill_preference_embeddings(batch_limit)
        total_thoughts = await self._backfill_thought_embeddings(batch_limit)
        total_messages = await self._backfill_message_embeddings(batch_limit)
        total = total_prefs + total_thoughts + total_messages
        if total:
            logger.info(
                "Startup embedding backfill complete: %d preferences, %d thoughts, %d messages",
                total_prefs,
                total_thoughts,
                total_messages,
            )

    async def _backfill_thought_embeddings(self, batch_limit: int) -> int:
        """Backfill thoughts with missing embeddings. Returns count embedded."""
        assert self.embedding_model_client is not None
        total = 0
        while True:
            thoughts = self.db.thoughts.get_without_embeddings(limit=batch_limit)
            if not thoughts:
                break
            try:
                texts = [t.content for t in thoughts]
                vecs = await self.embedding_model_client.embed(texts)
                for thought, vec in zip(thoughts, vecs, strict=True):
                    assert thought.id is not None
                    self.db.thoughts.update_embedding(thought.id, serialize_embedding(vec))
                    logger.info("Embedded thought %d: %s", thought.id, thought.content[:80])
                total += len(thoughts)
            except Exception as e:
                logger.warning("Startup embedding backfill failed for thoughts: %s", e)
                break
        return total

    async def _backfill_message_embeddings(self, batch_limit: int) -> int:
        """Backfill outgoing messages with missing embeddings. Returns count embedded."""
        assert self.embedding_model_client is not None
        total = 0
        while True:
            messages = self.db.messages.get_outgoing_without_embeddings(limit=batch_limit)
            if not messages:
                break
            try:
                texts = [m.content for m in messages]
                vecs = await self.embedding_model_client.embed(texts)
                for msg, vec in zip(messages, vecs, strict=True):
                    assert msg.id is not None
                    self.db.messages.update_embedding(msg.id, serialize_embedding(vec))
                total += len(messages)
            except Exception as e:
                logger.warning("Startup embedding backfill failed for messages: %s", e)
                break
        return total

    async def _backfill_preference_embeddings(self, batch_limit: int) -> int:
        """Backfill preferences with missing embeddings. Returns count embedded."""
        assert self.embedding_model_client is not None
        total = 0
        while True:
            prefs = self.db.preferences.get_without_embeddings(limit=batch_limit)
            if not prefs:
                break
            try:
                texts = [p.content for p in prefs]
                vecs = await self.embedding_model_client.embed(texts)
                for pref, vec in zip(prefs, vecs, strict=True):
                    assert pref.id is not None
                    self.db.preferences.update_embedding(pref.id, serialize_embedding(vec))
                    logger.info("Embedded preference %d: %s", pref.id, pref.content[:120])
                total += len(prefs)
            except Exception as e:
                logger.warning("Startup embedding backfill failed for preferences: %s", e)
                break
        return total

    async def run(self) -> None:
        """Run the agent."""
        logger.info("Starting Penny AI agent...")
        logger.info("Channel: %s (sender_id=%s)", self.config.channel_type, self.channel.sender_id)
        logger.info("Ollama model: %s", self.config.ollama_model)
        if self.config.ollama_vision_model:
            logger.info("Ollama model: %s (vision)", self.config.ollama_vision_model)
        if self.config.ollama_image_model:
            logger.info("Ollama model: %s (image generation)", self.config.ollama_image_model)

        # Validate channel connectivity before starting
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
        """Send a startup announcement to the user's default device."""
        try:
            sender = self.db.users.get_primary_sender()
            if not sender:
                logger.info("No user profile found for startup announcement")
                return

            # Only announce if the user has chatted before (not a fresh profile)
            if not self.db.messages.get_latest_incoming_time(sender):
                logger.info("No message history yet, skipping startup announcement")
                return

            restart_msg = await get_restart_message(self.model_client)
            announcement = f"👋 {restart_msg}"

            logger.info("Sending startup announcement to %s", sender)
            await self.channel.send_status_message(sender, announcement)
        except Exception as e:
            logger.warning("Failed to send startup announcement: %s", e)

    async def _prompt_for_missing_profiles(self) -> None:
        """Prompt the user if they don't have a profile set up yet (single-user)."""
        try:
            if self.db.users.get_primary_sender():
                return  # Profile exists, nothing to do

            # No profile — send prompt to any known sender from message history
            senders = self.db.users.get_all_senders()
            for sender in senders:
                try:
                    logger.info("User %s has no profile, sending prompt", sender)
                    await self.channel.send_status_message(sender, PennyResponse.PROFILE_REQUIRED)
                except Exception as e:
                    logger.warning("Failed to send profile prompt to %s: %s", sender, e)
        except Exception as e:
            logger.warning("Failed to send profile prompts: %s", e)

    async def shutdown(self) -> None:
        """Clean shutdown of resources."""
        logger.info("Shutting down agent...")
        self.scheduler.stop()
        await self.channel.close()
        await Agent.close_all()
        await self.model_client.close()
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
    logger.info("  ollama_model: %s", config.ollama_model)
    logger.info("  ollama_api_url: %s", config.ollama_api_url)
    logger.info("  idle_threshold: %.0fs", config.runtime.IDLE_SECONDS)

    agent = Penny(config)
    await agent.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Agent stopped by user")
        sys.exit(0)

"""Main agent loop for Penny."""

import asyncio
import json
import logging
import random
import signal
import sys
import time
from typing import Any

import websockets

from penny.agent import AgentController
from penny.agent.models import MessageRole
from penny.channels import MessageChannel, SignalChannel
from penny.config import Config, setup_logging
from penny.constants import CONTINUE_PROMPT, SUMMARIZE_PROMPT, SYSTEM_PROMPT, MessageDirection
from penny.database import Database
from penny.ollama import OllamaClient
from penny.tools import SearchTool, ToolRegistry

logger = logging.getLogger(__name__)


class PennyAgent:
    """AI agent powered by Ollama via an agent controller."""

    def __init__(self, config: Config, channel: MessageChannel | None = None):
        """Initialize the agent with configuration."""
        self.config = config
        self.channel = channel or SignalChannel(config.signal_api_url, config.signal_number)
        self.db = Database(config.db_path)
        self.db.create_tables()
        self.ollama_client = OllamaClient(
            config.ollama_api_url,
            config.ollama_model,
            db=self.db,
            max_retries=config.ollama_max_retries,
            retry_delay=config.ollama_retry_delay,
        )

        tool_registry = ToolRegistry()
        if config.perplexity_api_key:
            tool_registry.register(
                SearchTool(perplexity_api_key=config.perplexity_api_key, db=self.db)
            )
            logger.info("Search tool registered (Perplexity + image search)")
        else:
            logger.warning("No PERPLEXITY_API_KEY configured - agent will have no tools")

        self.controller = AgentController(
            ollama_client=self.ollama_client,
            tool_registry=tool_registry,
            max_steps=config.message_max_steps,
        )

        self.running = True
        self.last_message_time = time.monotonic()

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        logger.info("Received shutdown signal, stopping agent...")
        self.running = False

    async def handle_message(self, envelope_data: dict) -> None:
        """Process an incoming message through the agent controller."""
        try:
            self.last_message_time = time.monotonic()

            message = self.channel.extract_message(envelope_data)
            if message is None:
                return

            logger.info("Received message from %s: %s", message.sender, message.content)

            # Find parent message and build thread context if this is a quoted reply
            parent_id = None
            history = None
            if message.quoted_text:
                parent_id, history = self.db.get_thread_context(message.quoted_text)

            # Log incoming message linked to parent
            incoming_id = self.db.log_message(
                MessageDirection.INCOMING, message.sender, message.content, parent_id=parent_id
            )

            await self.channel.send_typing(message.sender, True)
            try:
                response = await self.controller.run(
                    current_message=message.content,
                    system_prompt=SYSTEM_PROMPT,
                    history=history,
                    on_step=lambda _: self.channel.send_typing(message.sender, True),
                )

                answer = (
                    response.answer.strip()
                    if response.answer
                    else "Sorry, I couldn't generate a response."
                )
                self.db.log_message(
                    MessageDirection.OUTGOING,
                    self.config.signal_number,
                    answer,
                    parent_id=incoming_id,
                )
                await self.channel.send_message(
                    message.sender, answer, attachments=response.attachments or None
                )
            finally:
                await self.channel.send_typing(message.sender, False)

        except Exception as e:
            logger.exception("Error handling message: %s", e)

    async def listen_for_messages(self) -> None:
        """Listen for incoming messages from the channel."""
        connection_url = self.channel.get_connection_url()

        while self.running:
            try:
                logger.info("Connecting to channel: %s", connection_url)

                async with websockets.connect(connection_url) as websocket:
                    logger.info("Connected to Signal WebSocket")

                    while self.running:
                        try:
                            message = await asyncio.wait_for(
                                websocket.recv(),
                                timeout=30.0,
                            )

                            logger.debug("Received raw WebSocket message: %s", message[:200])

                            envelope = json.loads(message)
                            logger.info("Parsed envelope with keys: %s", envelope.keys())

                            asyncio.create_task(self.handle_message(envelope))

                        except TimeoutError:
                            logger.debug("WebSocket receive timeout, continuing...")
                            continue

                        except json.JSONDecodeError as e:
                            logger.warning("Failed to parse message JSON: %s", e)
                            continue

            except websockets.exceptions.WebSocketException as e:
                logger.error("WebSocket error: %s", e)
                if self.running:
                    logger.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)

            except Exception as e:
                logger.exception("Unexpected error in message listener: %s", e)
                if self.running:
                    logger.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)

        logger.info("Message listener stopped")

    async def summarize_threads(self) -> None:
        """Background loop: summarize unsummarized threads when idle."""
        while self.running:
            await asyncio.sleep(1)

            idle_seconds = time.monotonic() - self.last_message_time
            if idle_seconds < self.config.summarize_idle_seconds:
                continue

            unsummarized = self.db.get_unsummarized_messages()
            if not unsummarized:
                continue

            logger.info("Idle for %.0fs, summarizing %d threads", idle_seconds, len(unsummarized))

            for msg in unsummarized:
                if not self.running:
                    break

                # Skip if no longer idle (new message came in)
                if time.monotonic() - self.last_message_time < self.config.summarize_idle_seconds:
                    logger.info("No longer idle, pausing summarization")
                    break

                assert msg.id is not None
                thread = self.db._walk_thread(msg.id)
                if len(thread) < 2:
                    # Single message, just mark it with empty summary to skip next time
                    self.db.set_parent_summary(msg.id, "")
                    continue

                # Format thread for summarization
                thread_text = "\n".join(
                    "{}: {}".format(
                        MessageRole.USER
                        if m.direction == MessageDirection.INCOMING
                        else MessageRole.ASSISTANT,
                        m.content,
                    )
                    for m in thread
                )

                try:
                    response = await self.ollama_client.generate(f"{SUMMARIZE_PROMPT}{thread_text}")
                    summary = response.content.strip()
                    self.db.set_parent_summary(msg.id, summary)
                    logger.info(
                        "Summarized thread for message %d (length: %d)", msg.id, len(summary)
                    )
                except Exception as e:
                    logger.error("Failed to summarize thread for message %d: %s", msg.id, e)

    async def continue_conversations(self) -> None:
        """Background loop: spontaneously continue a dangling conversation at random intervals."""
        while self.running:
            delay = random.uniform(
                self.config.continue_min_seconds, self.config.continue_max_seconds
            )
            logger.debug("Next spontaneous continuation in %.0fs", delay)
            await asyncio.sleep(delay)

            if not self.running:
                break

            # Only continue conversations when idle
            idle_seconds = time.monotonic() - self.last_message_time
            if idle_seconds < self.config.continue_idle_seconds:
                continue

            leaves = self.db.get_conversation_leaves()
            if not leaves:
                logger.debug("No conversation leaves to continue")
                continue

            leaf = random.choice(leaves)
            assert leaf.id is not None

            # Walk thread to find the recipient (sender of an incoming message)
            thread = self.db._walk_thread(leaf.id)
            recipient = None
            for msg in thread:
                if msg.direction == MessageDirection.INCOMING:
                    recipient = msg.sender
                    break

            if not recipient:
                logger.debug("Could not find recipient for leaf message %d", leaf.id)
                continue

            logger.info(
                "Spontaneously continuing conversation (leaf=%d, recipient=%s)", leaf.id, recipient
            )

            # Build thread history
            history = [
                (
                    MessageRole.USER
                    if m.direction == MessageDirection.INCOMING
                    else MessageRole.ASSISTANT,
                    m.content,
                )
                for m in thread
            ]

            try:
                await self.channel.send_typing(recipient, True)
                response = await self.controller.run(
                    current_message=CONTINUE_PROMPT,
                    system_prompt=SYSTEM_PROMPT,
                    history=history,
                )

                answer = response.answer.strip() if response.answer else None
                if not answer:
                    continue

                self.db.log_message(
                    MessageDirection.OUTGOING,
                    self.config.signal_number,
                    answer,
                    parent_id=leaf.id,
                )
                await self.channel.send_message(
                    recipient, answer, attachments=response.attachments or None
                )
                await self.channel.send_typing(recipient, False)
            except Exception as e:
                logger.exception("Error in spontaneous continuation: %s", e)

    async def run(self) -> None:
        """Run the agent."""
        logger.info("Starting Penny AI agent...")
        logger.info("Signal number: %s", self.config.signal_number)
        logger.info("Ollama model: %s", self.config.ollama_model)

        try:
            await asyncio.gather(
                self.listen_for_messages(),
                self.summarize_threads(),
                self.continue_conversations(),
            )
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Clean shutdown of resources."""
        logger.info("Shutting down agent...")
        await self.channel.close()
        await self.ollama_client.close()
        logger.info("Agent shutdown complete")


async def main() -> None:
    """Main entry point."""
    config = Config.load()
    setup_logging(config.log_level, config.log_file)

    agent = PennyAgent(config)
    await agent.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Agent stopped by user")
        sys.exit(0)

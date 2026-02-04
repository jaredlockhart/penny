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

from penny.agent import Agent
from penny.agent.models import MessageRole
from penny.channels import MessageChannel, SignalChannel
from penny.config import Config, setup_logging
from penny.constants import CONTINUE_PROMPT, SUMMARIZE_PROMPT, SYSTEM_PROMPT, MessageDirection
from penny.database import Database
from penny.tools import SearchTool

logger = logging.getLogger(__name__)


class Penny:
    """AI agent powered by Ollama via an agent controller."""

    def __init__(self, config: Config, channel: MessageChannel | None = None):
        """Initialize the agent with configuration."""
        self.config = config
        self.channel = channel or SignalChannel(config.signal_api_url, config.signal_number)
        self.db = Database(config.db_path)
        self.db.create_tables()

        def search_tools():
            if config.perplexity_api_key:
                return [SearchTool(perplexity_api_key=config.perplexity_api_key, db=self.db)]
            return []

        self.message_agent = Agent(
            system_prompt=SYSTEM_PROMPT,
            model=config.ollama_model,
            ollama_api_url=config.ollama_api_url,
            tools=search_tools(),
            db=self.db,
            max_steps=config.message_max_steps,
            max_retries=config.ollama_max_retries,
            retry_delay=config.ollama_retry_delay,
        )

        self.continue_agent = Agent(
            system_prompt=SYSTEM_PROMPT,
            model=config.ollama_model,
            ollama_api_url=config.ollama_api_url,
            tools=search_tools(),
            db=self.db,
            max_steps=config.message_max_steps,
            max_retries=config.ollama_max_retries,
            retry_delay=config.ollama_retry_delay,
        )

        self.summarize_agent = Agent(
            system_prompt=SUMMARIZE_PROMPT,
            model=config.ollama_model,
            ollama_api_url=config.ollama_api_url,
            tools=[],
            db=self.db,
            max_steps=1,
            max_retries=config.ollama_max_retries,
            retry_delay=config.ollama_retry_delay,
        )

        self.running = True
        self.last_message_time = time.monotonic()
        self._continue_cancel: asyncio.Event = asyncio.Event()

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        logger.info("Received shutdown signal, stopping agent...")
        self.running = False

    async def _typing_loop(self, recipient: str, interval: float = 4.0) -> None:
        """Send typing indicators on a loop until cancelled."""
        try:
            while True:
                await self.channel.send_typing(recipient, True)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    async def handle_message(self, envelope_data: dict) -> None:
        """Process an incoming message through the agent controller."""
        try:
            self.last_message_time = time.monotonic()
            self._continue_cancel.set()

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

            typing_task = asyncio.create_task(self._typing_loop(message.sender))
            try:
                response = await self.message_agent.run(
                    prompt=message.content,
                    history=history,
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
                typing_task.cancel()
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
                        MessageRole.USER.value
                        if m.direction == MessageDirection.INCOMING
                        else MessageRole.ASSISTANT.value,
                        m.content,
                    )
                    for m in thread
                )

                try:
                    response = await self.summarize_agent.run(prompt=thread_text)
                    summary = response.answer.strip()
                    self.db.set_parent_summary(msg.id, summary)
                    logger.info(
                        "Summarized thread for message %d (length: %d)", msg.id, len(summary)
                    )
                except Exception as e:
                    logger.error("Failed to summarize thread for message %d: %s", msg.id, e)

    async def continue_conversations(self) -> None:
        """Background loop: spontaneously continue a dangling conversation.

        Logic:
        1. Wait until idle for idle_time (reset if message received)
        2. Start random timer between min and max (cancel if message received)
        3. When timer completes, send continuation
        """
        while self.running:
            # Phase 1: Wait for idle threshold
            logger.info("Continuation: waiting for %.0fs idle", self.config.continue_idle_seconds)
            while self.running:
                self._continue_cancel.clear()
                idle_seconds = time.monotonic() - self.last_message_time
                if idle_seconds >= self.config.continue_idle_seconds:
                    break
                remaining = self.config.continue_idle_seconds - idle_seconds
                try:
                    await asyncio.wait_for(self._continue_cancel.wait(), timeout=remaining)
                    # Message received, loop back to check idle again
                    logger.info("Continuation: idle reset by incoming message")
                    continue
                except TimeoutError:
                    # Idle threshold reached
                    break

            if not self.running:
                break

            # Phase 2: Random delay timer (cancellable)
            self._continue_cancel.clear()
            delay = random.uniform(
                self.config.continue_min_seconds, self.config.continue_max_seconds
            )
            logger.info("Continuation: idle threshold reached, timer started (%.0fs)", delay)
            try:
                await asyncio.wait_for(self._continue_cancel.wait(), timeout=delay)
                # Message received, go back to phase 1
                logger.info("Continuation: timer cancelled by incoming message")
                continue
            except TimeoutError:
                # Timer completed
                logger.info("Continuation: timer completed, sending message")
                pass

            if not self.running:
                break

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
                typing_task = asyncio.create_task(self._typing_loop(recipient))
                try:
                    response = await self.continue_agent.run(
                        prompt=CONTINUE_PROMPT,
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
                finally:
                    typing_task.cancel()
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
        await Agent.close_all()
        logger.info("Agent shutdown complete")


async def main() -> None:
    """Main entry point."""
    config = Config.load()
    setup_logging(config.log_level, config.log_file)

    logger.info("Starting Penny with config:")
    logger.info("  ollama_model: %s", config.ollama_model)
    logger.info("  ollama_api_url: %s", config.ollama_api_url)
    logger.info("  signal_api_url: %s", config.signal_api_url)
    logger.info(
        "  continue_idle: %.0fs, continue_range: %.0fs-%.0fs",
        config.continue_idle_seconds,
        config.continue_min_seconds,
        config.continue_max_seconds,
    )

    agent = Penny(config)
    await agent.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Agent stopped by user")
        sys.exit(0)

"""Main agent loop for Penny - Tool-based task system."""

import asyncio
import json
import logging
import signal
import sys
import time
from typing import Any

import websockets

from penny.agentic import AgenticController
from penny.channels import MessageChannel, SignalChannel
from penny.config import Config, setup_logging
from penny.constants import ErrorMessages, SystemPrompts
from penny.memory import Database
from penny.memory.models import MessageDirection
from penny.ollama import OllamaClient
from penny.tools import (
    CompleteTaskTool,
    CreateTaskTool,
    GetCurrentTimeTool,
    ListTasksTool,
    PerplexitySearchTool,
    StoreMemoryTool,
    ToolRegistry,
)

logger = logging.getLogger(__name__)


class PennyAgent:
    """AI agent that responds via Ollama with tool-based task management."""

    def __init__(self, config: Config, channel: MessageChannel | None = None):
        """Initialize the agent with configuration."""
        self.config = config
        self.channel = channel or SignalChannel(config.signal_api_url, config.signal_number)
        self.ollama_client = OllamaClient(config.ollama_api_url, config.ollama_model)

        # Initialize database
        self.db = Database(config.db_path)
        self.db.create_tables()

        # Auto-discover user number from first message
        self.user_number = None

        # Message handler registry: store_memory and create_task (static)
        self.message_registry = ToolRegistry()
        self.message_registry.register(StoreMemoryTool(db=self.db))
        self.message_registry.register(CreateTaskTool(db=self.db, agent=self))

        # Task processor registry: all task-related tools
        self.task_registry = ToolRegistry()
        self.task_registry.register(StoreMemoryTool(db=self.db))
        self.task_registry.register(GetCurrentTimeTool())
        self.task_registry.register(ListTasksTool(db=self.db))
        self.task_registry.register(CompleteTaskTool(db=self.db))

        # Add Perplexity search if configured
        if config.perplexity_api_key:
            self.task_registry.register(PerplexitySearchTool(api_key=config.perplexity_api_key))
            logger.info("Perplexity search tool registered for task processing")
        else:
            logger.info("Perplexity API key not configured, skipping search tool")

        # Create separate controllers for each context
        self.message_controller = AgenticController(
            ollama_client=self.ollama_client,
            tool_registry=self.message_registry,
            db=self.db,
            max_steps=self.config.message_max_steps,
        )

        self.task_controller = AgenticController(
            ollama_client=self.ollama_client,
            tool_registry=self.task_registry,
            db=self.db,
            max_steps=self.config.task_max_steps,
        )

        # Track last message time for idle detection
        self.last_message_time = time.time()

        self.running = True

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        logger.info("Received shutdown signal, stopping agent...")
        self.running = False

    async def _send_response(
        self, recipient: str, answer: str, thinking: str | None = None
    ) -> None:
        """
        Send a response to a user with typing indicators and proper formatting.

        Args:
            recipient: Phone number to send to
            answer: Response text to send
            thinking: Optional thinking text (logged only on first chunk)
        """
        # Start typing indicator
        await self.channel.send_typing(recipient, True)

        try:
            # Split answer by newlines and send each line separately
            lines = [line.strip() for line in answer.split("\n") if line.strip()]
            for idx, line in enumerate(lines):
                await self.channel.send_message(recipient, line)
                self.db.log_message(
                    MessageDirection.OUTGOING.value,
                    self.config.signal_number,
                    recipient,
                    line,
                    chunk_index=idx,
                    thinking=thinking if idx == 0 else None,
                )
        finally:
            # Always stop typing indicator
            await self.channel.send_typing(recipient, False)

    async def handle_message(self, envelope_data: dict) -> None:
        """
        Process an incoming message from the channel.

        Args:
            envelope_data: Raw message data from the channel
        """
        try:
            # Extract message content from channel data
            message = self.channel.extract_message(envelope_data)
            if message is None:
                return

            logger.info("Received message from %s: %s", message.sender, message.content)

            # Auto-discover user number from first message
            if self.user_number is None:
                self.user_number = message.sender
                logger.info("Auto-discovered user number: %s", self.user_number)

            # Update last message time for idle detection
            self.last_message_time = time.time()

            # Log incoming message
            self.db.log_message(
                MessageDirection.INCOMING.value,
                message.sender,
                self.config.signal_number,
                message.content,
            )

            try:
                # Get conversation history
                history = self.db.get_conversation_history(
                    message.sender,
                    self.config.signal_number,
                    limit=self.config.conversation_history_limit,
                )

                # Run agentic loop using message controller (only has store_memory and create_task)
                response = await self.message_controller.run(
                    history,
                    message.content,
                    system_prompt=SystemPrompts.MESSAGE_HANDLER,
                )

                # Get answer with fallback
                answer = response.answer.strip() if response.answer else ""
                if not answer:
                    answer = ErrorMessages.NO_RESPONSE

                # Send response using shared helper
                await self._send_response(message.sender, answer, response.thinking)

            except Exception as e:
                logger.exception("Error processing message: %s", e)
                await self._send_response(message.sender, ErrorMessages.PROCESSING_ERROR)

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
                            # Receive message with timeout
                            message = await asyncio.wait_for(
                                websocket.recv(),
                                timeout=30.0,
                            )

                            logger.debug("Received raw WebSocket message: %s", message[:200])

                            # Parse JSON envelope
                            envelope = json.loads(message)
                            logger.info("Parsed envelope with keys: %s", envelope.keys())

                            # Handle message in background
                            asyncio.create_task(self.handle_message(envelope))

                        except asyncio.TimeoutError:
                            # Timeout is expected - just continue listening
                            logger.debug("WebSocket receive timeout, continuing...")
                            continue

                        except json.JSONDecodeError as e:
                            logger.warning("Failed to parse message JSON: %s", e)
                            logger.debug("Raw message: %s", message)
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

    async def process_tasks(self) -> None:
        """Background task processor that works on tasks during idle time."""
        logger.info("Starting background task processor...")

        while self.running:
            try:
                # Check if we've been idle
                idle_time = time.time() - self.last_message_time

                if idle_time >= self.config.idle_timeout_seconds:
                    # Check database directly for pending tasks (avoid model call if empty)
                    pending_tasks = self.db.get_pending_tasks()

                    if pending_tasks:
                        logger.info("Found %d pending task(s), processing...", len(pending_tasks))

                        # Track task IDs before processing
                        pending_task_ids = {task.id for task in pending_tasks}

                        # Run task controller to work on tasks
                        try:
                            response = await self.task_controller.run(
                                [], SystemPrompts.TASK_PROCESSOR
                            )
                            logger.info("Task processing response: %s", response.answer[:100])

                            # Check for newly completed tasks and send results to requester
                            from penny.memory.models import Task, TaskStatus

                            with self.db.get_session() as session:
                                completed_tasks = (
                                    session.query(Task)
                                    .filter(
                                        Task.id.in_(pending_task_ids),
                                        Task.status == TaskStatus.COMPLETED.value,
                                    )
                                    .all()
                                )

                                logger.info(
                                    "Found %d completed tasks from %d pending",
                                    len(completed_tasks),
                                    len(pending_task_ids),
                                )

                                for task in completed_tasks:
                                    logger.info(
                                        "Task %d - status: %s, result: %s",
                                        task.id,
                                        task.status,
                                        "present" if task.result else "missing",
                                    )
                                    if task.result:
                                        logger.info(
                                            "Sending task %d result to %s",
                                            task.id,
                                            task.requester,
                                        )
                                        # Send task result using shared helper
                                        await self._send_response(task.requester, task.result)

                        except Exception as e:
                            logger.exception("Error during task processing: %s", e)
                    else:
                        logger.debug("No pending tasks, skipping model call")

                # Check periodically
                await asyncio.sleep(self.config.task_check_interval)

            except Exception as e:
                logger.exception("Error in task processor: %s", e)
                await asyncio.sleep(self.config.idle_timeout_seconds)

        logger.info("Task processor stopped")

    async def run(self) -> None:
        """Run the agent."""
        logger.info("Starting Penny AI agent...")
        logger.info("Signal number: %s", self.config.signal_number)
        logger.info("Ollama model: %s", self.config.ollama_model)

        try:
            # Start both message listener and task processor in parallel
            await asyncio.gather(
                self.listen_for_messages(),
                self.process_tasks(),
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
    # Load configuration
    config = Config.load()
    setup_logging(config.log_level, config.log_file)

    # Create and run agent
    agent = PennyAgent(config)
    await agent.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Agent stopped by user")
        sys.exit(0)

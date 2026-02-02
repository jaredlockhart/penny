"""Main agent loop for Penny - Simple Signal echo test."""

import asyncio
import json
import logging
import signal
import sys
from typing import Any

import websockets

from penny.agentic import AgenticController
from penny.channels import MessageChannel, SignalChannel
from penny.config import Config, setup_logging
from penny.memory import Database
from penny.ollama import OllamaClient
from penny.tools import GetCurrentTimeTool, PerplexitySearchTool, StoreMemoryTool, ToolRegistry

logger = logging.getLogger(__name__)


class PennyAgent:
    """AI agent that responds via Ollama."""

    def __init__(self, config: Config, channel: MessageChannel | None = None):
        """Initialize the agent with configuration."""
        self.config = config
        self.channel = channel or SignalChannel(config.signal_api_url, config.signal_number)
        self.ollama_client = OllamaClient(config.ollama_api_url, config.ollama_model)

        # Initialize database
        self.db = Database(config.db_path)
        self.db.create_tables()

        # Initialize tools
        self.tool_registry = ToolRegistry()
        self.tool_registry.register(GetCurrentTimeTool())
        self.tool_registry.register(StoreMemoryTool(db=self.db))

        # Register Perplexity search tool if API key is configured
        if config.perplexity_api_key:
            self.tool_registry.register(PerplexitySearchTool(api_key=config.perplexity_api_key))
            logger.info("Registered Perplexity search tool")
        else:
            logger.info("Perplexity API key not configured, skipping search tool")

        # Initialize agentic controller
        self.controller = AgenticController(
            ollama_client=self.ollama_client,
            tool_registry=self.tool_registry,
            db=self.db,
            max_steps=5,
        )

        self.running = True

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        logger.info("Received shutdown signal, stopping agent...")
        self.running = False

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

            # Log incoming message
            self.db.log_message("incoming", message.sender, self.config.signal_number, message.content)

            # Send typing indicator
            await self.channel.send_typing(message.sender, True)

            try:
                # Get conversation history
                history = self.db.get_conversation_history(message.sender, self.config.signal_number, limit=20)
                logger.debug("Got %d history messages", len(history))

                # Run agentic loop to get final answer
                response = await self.controller.run(history, message.content)
                logger.info("Received from controller - answer length: %d, thinking: %s",
                           len(response.answer),
                           "present" if response.thinking else "None")

                # Ensure we have a non-empty answer
                answer = response.answer.strip() if response.answer else ""
                if not answer:
                    answer = "Sorry, I couldn't generate a response."

                # Split answer by newlines and send each line separately
                lines = [line.strip() for line in answer.split('\n') if line.strip()]
                for idx, line in enumerate(lines):
                    await self.channel.send_message(message.sender, line)
                    # Log each line to database, but only include thinking on the first one
                    self.db.log_message(
                        "outgoing",
                        self.config.signal_number,
                        message.sender,
                        line,
                        chunk_index=idx,
                        thinking=response.thinking if idx == 0 else None,
                    )

            except Exception as e:
                logger.exception("Error in message handling: %s", e)
                error_msg = "Sorry, I encountered an error processing your message."
                await self.channel.send_message(message.sender, error_msg)
                self.db.log_message("outgoing", self.config.signal_number, message.sender, error_msg)

            finally:
                # Always stop typing indicator
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

    async def run(self) -> None:
        """Run the agent."""
        logger.info("Starting Penny AI agent...")
        logger.info("Signal number: %s", self.config.signal_number)
        logger.info("Ollama model: %s", self.config.ollama_model)

        try:
            await self.listen_for_messages()
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

"""Main agent loop for Penny - Simple Signal echo test."""

import asyncio
import json
import logging
import signal
import sys
from typing import Any

import websockets

from penny.channels import MessageChannel, SignalChannel
from penny.config import Config, setup_logging
from penny.memory import Database, build_context
from penny.ollama import OllamaClient

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

        self.running = True

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        logger.info("Received shutdown signal, stopping agent...")
        self.running = False

    async def _stream_and_send_response(self, sender: str, context: str) -> None:
        """
        Stream response from Ollama and send chunks to Signal.

        Args:
            sender: Phone number of the recipient
            context: Conversation context to send to Ollama
        """
        logger.info("Generating streaming response with Ollama...")
        logger.debug("Context length: %d chars", len(context))
        chunk_count = 0

        try:
            # Stream response lines from Ollama
            async for chunk in self.ollama_client.stream_response(context):
                # Turn off typing indicator before sending
                await self.channel.send_typing(sender, False)

                logger.debug("Sending chunk %d: %s...", chunk_count, chunk.line[:50])
                await self.channel.send_message(sender, chunk.line)

                # Log to database (thinking included in first chunk)
                self.db.log_message("outgoing", self.config.signal_number, sender, chunk.line, chunk_count, thinking=chunk.thinking)
                chunk_count += 1

                # Turn typing indicator back on for next chunk
                await self.channel.send_typing(sender, True)

            logger.info("Sent %d chunks to %s", chunk_count, sender)

        except Exception as e:
            logger.error("Error during streaming generation: %s", e)
            error_msg = "Sorry, I encountered an error generating a response."
            await self.channel.send_message(sender, error_msg)
            self.db.log_message("outgoing", self.config.signal_number, sender, error_msg)

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
                # Build context from conversation history
                history = self.db.get_conversation_history(message.sender, self.config.signal_number, limit=20)
                context = build_context(history, message.content)

                # Stream and send response
                await self._stream_and_send_response(message.sender, context)
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
    setup_logging(config.log_level)

    # Create and run agent
    agent = PennyAgent(config)
    await agent.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Agent stopped by user")
        sys.exit(0)

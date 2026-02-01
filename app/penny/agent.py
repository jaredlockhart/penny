"""Main agent loop for Penny - Simple Signal echo test."""

import asyncio
import json
import logging
import signal
import sys
from typing import Any

import websockets

from penny.config import Config, setup_logging
from penny.ollama import OllamaClient
from penny.signal import SignalClient

logger = logging.getLogger(__name__)


class PennyAgent:
    """AI agent that responds via Ollama."""

    def __init__(self, config: Config):
        """Initialize the agent with configuration."""
        self.config = config
        self.signal_client = SignalClient(config.signal_api_url, config.signal_number)
        self.ollama_client = OllamaClient(config.ollama_api_url, config.ollama_model)
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
        Process an incoming Signal message.

        Args:
            envelope_data: Signal message envelope from WebSocket
        """
        try:
            # Parse envelope using SignalClient
            envelope = self.signal_client.parse_envelope(envelope_data)
            if envelope is None:
                return

            logger.debug("Processing envelope from: %s", envelope.envelope.source)

            # Check if this is a data message (not typing indicator, etc.)
            if envelope.envelope.dataMessage is None:
                logger.debug("Ignoring non-data message")
                return

            sender = envelope.envelope.source
            content = envelope.envelope.dataMessage.message.strip()

            logger.info("Extracted - sender: %s, content: '%s'", sender, content)

            if not content:
                logger.debug("Ignoring empty message from %s", sender)
                return

            logger.info("Received message from %s: %s", sender, content)

            # Send typing indicator
            await self.signal_client.send_typing(sender, True)

            # Generate response using Ollama
            logger.info("Generating response with Ollama...")
            response = await self.ollama_client.generate(content)

            if response is None:
                logger.error("Failed to generate response from Ollama")
                response = "Sorry, I'm having trouble generating a response right now."

            # Stop typing indicator
            await self.signal_client.send_typing(sender, False)

            logger.info("Sending response to %s: %s...", sender, response[:50])

            success = await self.signal_client.send_message(sender, response)

            if success:
                logger.info("Successfully sent response")
            else:
                logger.error("Failed to send response")

        except Exception as e:
            logger.exception("Error handling message: %s", e)

    async def listen_for_messages(self) -> None:
        """Listen for incoming Signal messages via WebSocket."""
        ws_url = self.signal_client.get_websocket_url()

        while self.running:
            try:
                logger.info("Connecting to Signal WebSocket: %s", ws_url)

                async with websockets.connect(ws_url) as websocket:
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
        await self.signal_client.close()
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

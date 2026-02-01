"""Main agent loop for Penny - Simple Signal echo test."""

import asyncio
import json
import logging
import signal
import sys
from typing import Any

import websockets

from penny.config import Config, setup_logging
from penny.memory import Database, Message
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

    def _build_context(self, history: list, current_message: str) -> str:
        """
        Build conversation context from history.

        Args:
            history: List of Message objects from database
            current_message: The current incoming message

        Returns:
            Formatted context string for Ollama
        """
        context_parts = ["You are Penny, a helpful AI assistant communicating via Signal messages."]

        if history:
            context_parts.append("\nRecent conversation history:")
            for msg in history:
                # Skip the current message if it's already in history (shouldn't be, but just in case)
                if msg.direction == "incoming":
                    context_parts.append(f"User: {msg.content}")
                else:
                    # For outgoing chunks, only include non-chunk messages or first chunk
                    if msg.chunk_index is None or msg.chunk_index == 0:
                        context_parts.append(f"Penny: {msg.content}")

        context_parts.append(f"\nUser: {current_message}")
        context_parts.append("Penny:")

        return "\n".join(context_parts)

    def log_message(self, direction: str, sender: str, recipient: str, content: str, chunk_index: int | None = None) -> None:
        """
        Log a message to the database.

        Args:
            direction: "incoming" or "outgoing"
            sender: Phone number of sender
            recipient: Phone number of recipient
            content: Message content
            chunk_index: Optional chunk index for streaming responses
        """
        try:
            with self.db.get_session() as session:
                message = Message(
                    direction=direction,
                    sender=sender,
                    recipient=recipient,
                    content=content,
                    chunk_index=chunk_index,
                )
                session.add(message)
                session.commit()
                logger.debug("Logged %s message: %s -> %s", direction, sender, recipient)
        except Exception as e:
            logger.error("Failed to log message: %s", e)

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

            # Log incoming message
            self.log_message("incoming", sender, self.config.signal_number, content)

            # Send typing indicator
            await self.signal_client.send_typing(sender, True)

            # Build context from conversation history
            history = self.db.get_conversation_history(sender, self.config.signal_number, limit=20)
            context = self._build_context(history, content)

            # Generate response using Ollama streaming
            logger.info("Generating streaming response with Ollama...")
            logger.debug("Context length: %d chars", len(context))
            buffer = ""
            chunk_count = 0

            try:
                async for chunk in self.ollama_client.generate_stream(context):
                    buffer += chunk

                    # Send accumulated text when we hit a newline or paragraph break
                    if "\n" in buffer:
                        # Split on newlines, keep the last incomplete part in buffer
                        parts = buffer.split("\n")
                        buffer = parts[-1]  # Keep last part (might be incomplete)

                        # Send all complete parts
                        for part in parts[:-1]:
                            if part.strip():  # Only send non-empty lines
                                # Turn off typing indicator before sending
                                await self.signal_client.send_typing(sender, False)

                                logger.debug("Sending chunk %d: %s...", chunk_count, part[:50])
                                await self.signal_client.send_message(sender, part)

                                # Log outgoing chunk
                                self.log_message("outgoing", self.config.signal_number, sender, part, chunk_count)
                                chunk_count += 1

                                # Turn typing indicator back on for next chunk
                                await self.signal_client.send_typing(sender, True)

                # Send any remaining buffer
                if buffer.strip():
                    await self.signal_client.send_typing(sender, False)
                    logger.debug("Sending final chunk: %s...", buffer[:50])
                    await self.signal_client.send_message(sender, buffer)

                    # Log final chunk
                    self.log_message("outgoing", self.config.signal_number, sender, buffer, chunk_count)
                    chunk_count += 1

                logger.info("Sent %d chunks to %s", chunk_count, sender)

            except Exception as e:
                logger.error("Error during streaming generation: %s", e)
                error_msg = "Sorry, I encountered an error generating a response."
                await self.signal_client.send_message(sender, error_msg)
                self.log_message("outgoing", self.config.signal_number, sender, error_msg)

            # Stop typing indicator
            await self.signal_client.send_typing(sender, False)

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

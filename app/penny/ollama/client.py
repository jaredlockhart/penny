"""Ollama API client for LLM inference."""

import logging
import time

import ollama

from penny.ollama.models import ChatResponse

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API using the official SDK."""

    def __init__(self, api_url: str, model: str, db=None):
        """
        Initialize Ollama client.

        Args:
            api_url: Base URL for Ollama API (e.g., http://localhost:11434)
            model: Model name to use (e.g., llama3.2)
            db: Optional Database instance for logging prompts
        """
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.db = db

        # Initialize the official Ollama client
        self.client = ollama.AsyncClient(host=api_url)

        logger.info("Initialized Ollama client: url=%s, model=%s", api_url, model)

    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
    ) -> ChatResponse:
        """
        Generate a chat completion with optional tool calling.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions in Ollama format

        Returns:
            ChatResponse with message, thinking, tool calls, etc.
        """
        try:
            logger.debug("Sending chat request to Ollama")

            start = time.time()

            raw = await self.client.chat(
                model=self.model,
                messages=messages,
                tools=tools,
            )

            duration_ms = int((time.time() - start) * 1000)

            # Convert raw response to dict then parse with pydantic
            if hasattr(raw, "model_dump"):
                raw_dict = raw.model_dump()
            elif hasattr(raw, "__dict__"):
                raw_dict = dict(raw)
            else:
                raw_dict = dict(raw)

            response = ChatResponse(**raw_dict)

            # Extract thinking from either top-level or message
            thinking = response.thinking or response.message.thinking

            if response.has_tool_calls:
                logger.info("Received %d tool call(s)", len(response.message.tool_calls or []))
            if thinking:
                logger.debug("Model thinking: %s", thinking[:200])

            logger.debug("Response content length: %d", len(response.content))

            # Log to database
            if self.db:
                self.db.log_prompt(
                    model=self.model,
                    messages=messages,
                    response=raw_dict,
                    tools=tools,
                    thinking=thinking,
                    duration_ms=duration_ms,
                )

            return response

        except Exception as e:
            logger.exception("Ollama chat error: %s", e)
            raise

    async def generate(self, prompt: str, tools: list[dict] | None = None) -> ChatResponse:
        """
        Generate a completion for a prompt (converts to chat format internally).

        Args:
            prompt: The prompt to generate from
            tools: Optional list of tool definitions

        Returns:
            ChatResponse
        """
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, tools)

    async def close(self) -> None:
        """Close the client (SDK handles cleanup automatically)."""
        logger.info("Ollama client closed")

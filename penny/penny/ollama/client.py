"""Ollama API client for LLM inference."""

import asyncio
import logging
import time

import httpx
import ollama

from penny.ollama.models import ChatResponse, GenerateResponse

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API using the official SDK."""

    def __init__(self, api_url: str, model: str, db=None, *, max_retries: int, retry_delay: float):
        """
        Initialize Ollama client.

        Args:
            api_url: Base URL for Ollama API (e.g., http://localhost:11434)
            model: Model name to use (e.g., gpt-oss:20b)
            db: Optional Database instance for logging prompts
            max_retries: Number of retry attempts on failure
            retry_delay: Seconds between retries
        """
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.db = db
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize the official Ollama client
        self.client = ollama.AsyncClient(host=api_url)

        logger.info("Initialized Ollama client: url=%s, model=%s", api_url, model)

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        format: dict | str | None = None,
        model: str | None = None,
    ) -> ChatResponse:
        """
        Generate a chat completion with optional tool calling.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions in Ollama format
            format: Optional format specification (JSON schema dict, "json", or None)
            model: Optional model override for this call (e.g., vision model)

        Returns:
            ChatResponse with message, thinking, tool calls, etc.
        """
        effective_model = model or self.model
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    "Sending chat request to Ollama (attempt %d/%d)", attempt + 1, self.max_retries
                )
                logger.debug("Prompt messages: %s", messages)

                start = time.time()

                # Snapshot messages before the call so the log captures
                # exactly what was sent, not the mutated list.
                messages_snapshot = list(messages)

                # Build kwargs for chat call
                chat_kwargs: dict = {
                    "model": effective_model,
                    "messages": messages,
                }
                if tools is not None:
                    chat_kwargs["tools"] = tools
                if format is not None:
                    chat_kwargs["format"] = format

                raw = await self.client.chat(**chat_kwargs)

                duration_ms = int((time.time() - start) * 1000)

                raw_dict = raw.model_dump()

                response = ChatResponse(**raw_dict)

                # Extract thinking from either top-level or message
                thinking = response.thinking or response.message.thinking

                if response.has_tool_calls:
                    logger.info("Received %d tool call(s)", len(response.message.tool_calls or []))
                if thinking:
                    logger.debug("Model thinking: %s", thinking[:200])

                logger.debug("Response content: %s", response.content)
                if response.has_tool_calls:
                    logger.debug("Response tool calls: %s", response.message.tool_calls)

                # Log to database
                if self.db:
                    self.db.log_prompt(
                        model=effective_model,
                        messages=messages_snapshot,
                        response=raw_dict,
                        tools=tools,
                        thinking=thinking,
                        duration_ms=duration_ms,
                    )

                return response

            except Exception as e:
                last_error = e
                logger.warning(
                    "Ollama chat error (attempt %d/%d): %s", attempt + 1, self.max_retries, e
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)

        logger.error("Ollama chat failed after %d attempts: %s", self.max_retries, last_error)
        raise last_error  # type: ignore[misc]

    async def generate(
        self, prompt: str, tools: list[dict] | None = None, format: dict | str | None = None
    ) -> ChatResponse:
        """
        Generate a completion for a prompt (converts to chat format internally).

        Args:
            prompt: The prompt to generate from
            tools: Optional list of tool definitions
            format: Optional format specification (JSON schema dict, "json", or None)

        Returns:
            ChatResponse
        """
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, tools, format)

    async def generate_image(self, prompt: str, model: str) -> str:
        """
        Generate an image from a text prompt using an image generation model.

        Args:
            prompt: Text description of the image to generate
            model: Image generation model name (e.g., x/z-image-turbo)

        Returns:
            Base64-encoded PNG image data

        Raises:
            RuntimeError: If the model does not return image data
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    "Sending image generation request to Ollama (attempt %d/%d)",
                    attempt + 1,
                    self.max_retries,
                )

                # Use httpx directly — the ollama SDK's GenerateResponse
                # Pydantic model drops the 'image' field from the response.
                async with httpx.AsyncClient(timeout=120) as http_client:
                    resp = await http_client.post(
                        f"{self.api_url}/api/generate",
                        json={"model": model, "prompt": prompt, "stream": False},
                    )
                    resp.raise_for_status()
                    response = GenerateResponse(**resp.json())

                image_data = response.image
                if not image_data:
                    raise RuntimeError("Model did not return image data")

                logger.info("Image generated successfully with model %s", model)
                return image_data

            except Exception as e:
                last_error = e
                logger.warning(
                    "Ollama image generation error (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)

        logger.error(
            "Ollama image generation failed after %d attempts: %s", self.max_retries, last_error
        )
        raise last_error  # type: ignore[misc]

    async def embed(self, text: str | list[str], model: str) -> list[list[float]]:
        """
        Generate embeddings for one or more texts.

        Args:
            text: Single text or list of texts to embed
            model: Embedding model name (e.g., nomic-embed-text)

        Returns:
            List of embedding vectors (one per input text)
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    "Sending embed request to Ollama (attempt %d/%d)", attempt + 1, self.max_retries
                )

                response = await self.client.embed(model=model, input=text)
                embeddings = [list(e) for e in response.embeddings]

                logger.debug(
                    "Generated %d embedding(s), dim=%d", len(embeddings), len(embeddings[0])
                )
                return embeddings

            except Exception as e:
                last_error = e
                logger.warning(
                    "Ollama embed error (attempt %d/%d): %s", attempt + 1, self.max_retries, e
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)

        logger.error("Ollama embed failed after %d attempts: %s", self.max_retries, last_error)
        raise last_error  # type: ignore[misc]

    async def close(self) -> None:
        """Close the client (SDK handles cleanup automatically)."""
        logger.info("Ollama client closed")

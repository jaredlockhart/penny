"""Ollama API client for LLM inference."""

import logging

import httpx
from pydantic import ValidationError

from penny.ollama.models import GenerateRequest, GenerateResponse

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(self, api_url: str, model: str):
        """
        Initialize Ollama client.

        Args:
            api_url: Base URL for Ollama API (e.g., http://localhost:11434)
            model: Model name to use (e.g., llama3.2)
        """
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.http_client = httpx.AsyncClient(timeout=120.0)
        self.last_thinking = ""  # Stores thinking from most recent stream
        logger.info("Initialized Ollama client: url=%s, model=%s", api_url, model)

    async def generate(self, prompt: str) -> str | None:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The prompt to generate from

        Returns:
            Generated text or None if generation fails
        """
        try:
            url = f"{self.api_url}/api/generate"

            # Create request
            request = GenerateRequest(
                model=self.model,
                prompt=prompt,
                stream=False,
            )

            logger.debug("Sending to Ollama: %s", url)
            logger.debug("Prompt length: %d chars", len(prompt))

            response = await self.http_client.post(
                url,
                json=request.model_dump(),
            )
            response.raise_for_status()

            # Parse response
            data = response.json()
            result = GenerateResponse.model_validate(data)

            logger.info(
                "Generated response: %d chars (eval_count: %s, eval_duration: %sms)",
                len(result.response),
                result.eval_count,
                result.eval_duration // 1_000_000 if result.eval_duration else None,
            )

            return result.response

        except httpx.HTTPError as e:
            logger.error("Ollama API error: %s", e)
            if hasattr(e, "response") and e.response is not None:
                logger.error(
                    "Response status: %d, body: %s",
                    e.response.status_code,
                    e.response.text,
                )
            return None

        except ValidationError as e:
            logger.error("Failed to parse Ollama response: %s", e)
            return None

        except Exception as e:
            logger.exception("Unexpected error during generation: %s", e)
            return None

    async def generate_stream(self, prompt: str):
        """
        Generate a completion as a stream of text chunks.

        Args:
            prompt: The prompt to generate from

        Yields:
            Dict with 'type' ('thinking' or 'response') and 'content' (text chunk)
        """
        try:
            import json

            url = f"{self.api_url}/api/generate"

            # Create streaming request
            request = GenerateRequest(
                model=self.model,
                prompt=prompt,
                stream=True,
            )

            logger.debug("Sending streaming request to Ollama: %s", url)
            logger.debug("Prompt length: %d chars", len(prompt))

            async with self.http_client.stream(
                "POST",
                url,
                json=request.model_dump(),
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        data = json.loads(line)

                        # Check for thinking chunk (from thinking models)
                        thinking_chunk = data.get("thinking", "")
                        if thinking_chunk:
                            yield {"type": "thinking", "content": thinking_chunk}

                        # Check for response chunk
                        response_chunk = data.get("response", "")
                        if response_chunk:
                            yield {"type": "response", "content": response_chunk}

                        # Check if done
                        if data.get("done", False):
                            logger.info("Stream completed")
                            break

                    except json.JSONDecodeError as e:
                        logger.warning("Failed to parse stream chunk: %s", e)
                        continue

        except httpx.HTTPError as e:
            logger.error("Ollama streaming API error: %s", e)
            if hasattr(e, "response") and e.response is not None:
                logger.error(
                    "Response status: %d, body: %s",
                    e.response.status_code,
                    e.response.text,
                )

        except Exception as e:
            logger.exception("Unexpected error during streaming generation: %s", e)

    async def stream_response(self, prompt: str):
        """
        Stream response lines with thinking data.

        Buffers response chunks and yields complete lines (split on newlines).
        The first yielded dict includes accumulated thinking; subsequent dicts have thinking as None.

        Args:
            prompt: The prompt to generate from

        Yields:
            Dict with 'line' (response text) and 'thinking' (reasoning, only in first yield)
        """
        # Reset thinking for this stream
        self.last_thinking = ""
        response_buffer = ""
        first_line = True

        async for chunk_data in self.generate_stream(prompt):
            chunk_type = chunk_data["type"]
            chunk_content = chunk_data["content"]

            if chunk_type == "thinking":
                # Accumulate thinking internally
                self.last_thinking += chunk_content

            elif chunk_type == "response":
                # Accumulate response
                response_buffer += chunk_content

                # Yield complete lines when we hit newlines
                if "\n" in response_buffer:
                    parts = response_buffer.split("\n")
                    response_buffer = parts[-1]  # Keep incomplete part

                    # Yield all complete lines
                    for line in parts[:-1]:
                        if line.strip():  # Only yield non-empty lines
                            thinking = self.last_thinking.strip() if first_line and self.last_thinking else None
                            yield {"line": line, "thinking": thinking}
                            first_line = False

        # Yield any remaining buffer
        if response_buffer.strip():
            thinking = self.last_thinking.strip() if first_line and self.last_thinking else None
            yield {"line": response_buffer, "thinking": thinking}

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.http_client.aclose()
        logger.info("Ollama client closed")

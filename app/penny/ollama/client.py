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

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.http_client.aclose()
        logger.info("Ollama client closed")

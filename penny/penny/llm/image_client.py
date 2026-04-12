"""Ollama-specific client for image generation and model listing.

These endpoints are Ollama-specific (not OpenAI-compatible) and use
raw httpx requests to the Ollama REST API.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class _GenerateResponse(BaseModel):
    """Response from Ollama's /api/generate endpoint."""

    model: str
    created_at: str = Field(alias="created_at")
    response: str | None = None
    done: bool
    image: str | None = None

    class Config:
        populate_by_name = True


class OllamaImageClient:
    """Client for Ollama-specific image generation and model listing."""

    def __init__(
        self,
        api_url: str,
        model: str,
        *,
        max_retries: int,
        retry_delay: float,
    ):
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    async def generate_image(self, prompt: str) -> str:
        """Generate an image from a text prompt.

        Returns base64-encoded PNG image data.
        Raises RuntimeError if the model does not return image data.
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    "Sending image generation request (attempt %d/%d)",
                    attempt + 1,
                    self.max_retries,
                )

                async with httpx.AsyncClient(timeout=120) as http_client:
                    response = await http_client.post(
                        f"{self.api_url}/api/generate",
                        json={"model": self.model, "prompt": prompt, "stream": False},
                    )
                    response.raise_for_status()
                    parsed = _GenerateResponse(**response.json())

                image_data = parsed.image
                if not image_data:
                    raise RuntimeError("Model did not return image data")

                logger.info("Image generated successfully with model %s", self.model)
                return image_data

            except httpx.HTTPStatusError as error:
                last_error = error
                if error.response.status_code == 404:
                    logger.error("Image generation failed (model not found, no retry): %s", error)
                    raise
                logger.warning(
                    "Image generation error (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    error,
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
            except Exception as error:
                last_error = error
                logger.warning(
                    "Image generation error (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    error,
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)

        logger.error("Image generation failed after %d attempts: %s", self.max_retries, last_error)
        assert last_error is not None
        raise last_error

    async def list_models(self) -> list[str]:
        """List all locally available Ollama models."""
        try:
            async with httpx.AsyncClient(timeout=10) as http_client:
                response = await http_client.get(f"{self.api_url}/api/tags")
                response.raise_for_status()
                data = response.json()
            return [m["name"] for m in data.get("models", []) if m.get("name")]
        except Exception as error:
            logger.warning("Failed to list Ollama models: %s", error)
            return []

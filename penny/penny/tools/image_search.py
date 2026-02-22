"""Image search via Serper (Google Images)."""

from __future__ import annotations

import base64
import logging
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

SERPER_IMAGES_URL = "https://google.serper.dev/images"
ALLOWED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp"})
ALLOWED_IMAGE_MIMES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})


class SerperImageResult(BaseModel):
    """A single image result from the Serper API."""

    imageUrl: str = ""
    imageWidth: int = 0
    imageHeight: int = 0
    thumbnailUrl: str = ""
    title: str = ""
    source: str = ""
    domain: str = ""
    link: str = ""
    position: int = 0


class SerperImageResponse(BaseModel):
    """Response from the Serper image search API."""

    images: list[SerperImageResult] = []


async def search_image(
    query: str,
    api_key: str | None = None,
    *,
    max_results: int,
    timeout: float,
) -> str | None:
    """Search for an image via Serper and return base64 data URI.

    Returns a data URI string (e.g., 'data:image/jpeg;base64,...') or None
    if no suitable image is found. Failures are logged and return None.
    """
    if not api_key:
        return None

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            # Search for images via Serper
            resp = await client.post(
                SERPER_IMAGES_URL,
                json={"q": query, "num": max_results},
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            )
            resp.raise_for_status()
            response = SerperImageResponse.model_validate(resp.json())

            if not response.images:
                return None

            # Download first valid image (skip non-raster formats like SVG)
            for result in response.images:
                if not result.imageUrl:
                    continue
                # Filter by URL extension before downloading
                path = urlparse(result.imageUrl).path.lower()
                ext = path[path.rfind(".") :] if "." in path else ""
                if ext and ext not in ALLOWED_IMAGE_EXTENSIONS:
                    logger.debug("Skipping disallowed extension %s: %s", ext, result.imageUrl)
                    continue
                try:
                    img_resp = await client.get(result.imageUrl)
                    img_resp.raise_for_status()
                    content_type = img_resp.headers.get("content-type", "")
                    mime = content_type.split(";")[0].strip()
                    if mime not in ALLOWED_IMAGE_MIMES:
                        logger.debug("Skipping disallowed MIME %s: %s", mime, result.imageUrl)
                        continue
                    image_b64 = base64.b64encode(img_resp.content).decode()
                    return f"data:{mime};base64,{image_b64}"
                except httpx.HTTPError:
                    logger.debug("Failed to download image: %s", result.imageUrl)
                    continue

        return None
    except Exception as e:
        logger.warning("Image search failed: %s", e)
        return None

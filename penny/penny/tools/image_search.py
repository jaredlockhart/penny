"""Standalone image search via DuckDuckGo."""

from __future__ import annotations

import asyncio
import base64
import logging
from functools import partial

import httpx
from duckduckgo_search import DDGS

from penny.constants import PennyConstants

logger = logging.getLogger(__name__)


async def search_image(query: str) -> str | None:
    """Search for an image via DuckDuckGo and return base64 data URI.

    Returns a data URI string (e.g., 'data:image/jpeg;base64,...') or None
    if no suitable image is found. Failures are logged and return None.
    """
    try:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None, partial(DDGS().images, query, max_results=PennyConstants.IMAGE_MAX_RESULTS)
        )

        if not results:
            return None

        async with httpx.AsyncClient(
            timeout=PennyConstants.IMAGE_DOWNLOAD_TIMEOUT, follow_redirects=True
        ) as client:
            for result in results:
                image_url = result.get("image", "")
                if not image_url:
                    continue
                try:
                    resp = await client.get(image_url)
                    resp.raise_for_status()
                    content_type = resp.headers.get("content-type", "")
                    if "image" not in content_type:
                        continue
                    image_b64 = base64.b64encode(resp.content).decode()
                    mime = content_type.split(";")[0].strip()
                    return f"data:{mime};base64,{image_b64}"
                except httpx.HTTPError:
                    logger.debug("Failed to download image: %s", image_url)
                    continue

        return None
    except Exception as e:
        logger.warning("Image search failed: %s", e)
        return None

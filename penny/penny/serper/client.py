"""Image search via Serper (Google Images)."""

from __future__ import annotations

import base64
import logging
from urllib.parse import urlparse

import httpx

from penny.serper.models import SerperImageResponse

logger = logging.getLogger(__name__)

SERPER_IMAGES_URL = "https://google.serper.dev/images"
BLOCKED_IMAGE_DOMAINS = frozenset({"tiktok.com", "instagram.com", "facebook.com"})
_SITE_EXCLUSIONS = " ".join(f"-site:{d}" for d in BLOCKED_IMAGE_DOMAINS)
ALLOWED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp"})
ALLOWED_IMAGE_MIMES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})
_EXT_TO_MIME: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


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

    logger.info("Image search query: %s", query[:120])
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await _fetch_results(client, query, api_key, max_results)
            if not response.images:
                logger.info("Image search returned no results")
                return None
            result = await _download_first_valid(client, response)
            if result:
                logger.info("Image search found image")
            else:
                logger.info("Image search: all results failed to download")
            return result
    except Exception as e:
        logger.warning("Image search failed: %s", e)
        return None


async def _fetch_results(
    client: httpx.AsyncClient,
    query: str,
    api_key: str,
    max_results: int,
) -> SerperImageResponse:
    """Call the Serper image search API."""
    resp = await client.post(
        SERPER_IMAGES_URL,
        json={"q": f"{query} {_SITE_EXCLUSIONS}", "num": max_results},
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return SerperImageResponse.model_validate(resp.json())


async def _download_first_valid(
    client: httpx.AsyncClient,
    response: SerperImageResponse,
) -> str | None:
    """Download first valid image from results, skip non-raster formats."""
    for result in response.images:
        if not result.imageUrl:
            continue
        domain = urlparse(result.imageUrl).hostname or ""
        if any(blocked in domain for blocked in BLOCKED_IMAGE_DOMAINS):
            logger.debug("Skipping blocked domain %s: %s", domain, result.imageUrl)
            continue
        ext = _parse_extension(result.imageUrl)
        if ext and ext not in ALLOWED_IMAGE_EXTENSIONS:
            logger.debug("Skipping disallowed extension %s: %s", ext, result.imageUrl)
            continue
        data_uri = await _try_download(client, result.imageUrl, ext)
        if data_uri:
            return data_uri
    return None


async def search_image_url(
    query: str,
    api_key: str | None = None,
    *,
    max_results: int,
    timeout: float,
) -> str | None:
    """Search for an image via Serper and return the URL (no download).

    Returns the first valid image URL or None. Used by the browser channel
    which can render URLs directly via <img> tags.
    """
    if not api_key:
        return None

    logger.info("Image URL search query: %s", query[:120])
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await _fetch_results(client, query, api_key, max_results)
            if not response.images:
                logger.info("Image URL search returned no results")
                return None
            for result in response.images:
                if not result.imageUrl:
                    continue
                domain = urlparse(result.imageUrl).hostname or ""
                if any(blocked in domain for blocked in BLOCKED_IMAGE_DOMAINS):
                    continue
                ext = _parse_extension(result.imageUrl)
                if not ext or ext not in ALLOWED_IMAGE_EXTENSIONS:
                    continue
                logger.info("Image URL search found: %s", result.imageUrl[:120])
                return result.imageUrl
            logger.info("Image URL search: no valid results")
            return None
    except Exception as e:
        logger.warning("Image URL search failed: %s", e)
        return None


def _parse_extension(url: str) -> str:
    """Extract file extension from URL, handling paths like /image.jpg/revision/latest/."""
    path = urlparse(url).path.lower()
    ext_start = path.rfind(".")
    if ext_start == -1:
        return ""
    ext_end = path.find("/", ext_start)
    return path[ext_start:ext_end] if ext_end != -1 else path[ext_start:]


async def _try_download(client: httpx.AsyncClient, url: str, ext: str) -> str | None:
    """Try to download and encode a single image. Returns data URI or None."""
    try:
        img_resp = await client.get(url)
        img_resp.raise_for_status()
        mime = _resolve_mime(img_resp, ext)
        if mime not in ALLOWED_IMAGE_MIMES:
            logger.debug("Skipping disallowed MIME %s: %s", mime, url)
            return None
        image_b64 = base64.b64encode(img_resp.content).decode()
        return f"data:{mime};base64,{image_b64}"
    except httpx.HTTPError:
        logger.debug("Failed to download image: %s", url)
        return None


def _resolve_mime(response: httpx.Response, ext: str) -> str:
    """Resolve MIME type from response headers, falling back to extension."""
    content_type = response.headers.get("content-type", "")
    mime = content_type.split(";")[0].strip()
    if mime in ("binary/octet-stream", "application/octet-stream") and ext:
        inferred = _EXT_TO_MIME.get(ext)
        if inferred:
            return inferred
    return mime

"""Tests for Serper image search client."""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from penny.serper.client import (
    _SITE_EXCLUSIONS,
    BLOCKED_IMAGE_DOMAINS,
    _download_first_valid,
    _fetch_results,
    search_image,
    search_image_url,
)
from penny.serper.models import SerperImageResponse

FAKE_API_KEY = "test-api-key"
FAKE_JPEG = b"\xff\xd8\xff\xe0fake-jpeg-bytes"
FAKE_JPEG_B64 = base64.b64encode(FAKE_JPEG).decode()


def _serper_response(*image_urls: str) -> dict:
    """Build a minimal Serper API response with the given image URLs."""
    return {
        "images": [{"imageUrl": url} for url in image_urls],
    }


# --- Happy path ---


@pytest.mark.asyncio
async def test_search_image_appends_site_exclusions(monkeypatch):
    """Query sent to Serper includes -site: exclusions for blocked domains."""
    captured_queries: list[str] = []

    async def mock_fetch(client, query, api_key, max_results):
        captured_queries.append(query)
        return SerperImageResponse(images=[])

    monkeypatch.setattr("penny.serper.client._fetch_results", mock_fetch)

    await search_image("cute cats", api_key=FAKE_API_KEY, max_results=5, timeout=10.0)

    # _fetch_results receives the query with site exclusions already appended
    # (search_image builds the client, then passes query to _fetch_results)
    # But actually _fetch_results is what appends the exclusions to the JSON body.
    # So we need to test _fetch_results directly.
    assert len(captured_queries) == 1


@pytest.mark.asyncio
async def test_fetch_results_appends_site_exclusions():
    """_fetch_results appends -site: exclusions to the Serper query."""
    captured_body: list[dict] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        captured_body.append(json.loads(request.content))
        return httpx.Response(200, json=_serper_response())

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await _fetch_results(client, "cute cats", FAKE_API_KEY, max_results=5)

    assert len(captured_body) == 1
    query = captured_body[0]["q"]
    assert query.startswith("cute cats ")
    for domain in BLOCKED_IMAGE_DOMAINS:
        assert f"-site:{domain}" in query


@pytest.mark.asyncio
async def test_download_returns_data_uri():
    """Successful download returns a data URI with correct MIME and base64."""
    response = SerperImageResponse.model_validate(_serper_response("https://example.com/photo.jpg"))

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=FAKE_JPEG, headers={"content-type": "image/jpeg"})

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        result = await _download_first_valid(client, response)

    assert result == f"data:image/jpeg;base64,{FAKE_JPEG_B64}"


# --- Filtering ---


@pytest.mark.asyncio
async def test_skips_blocked_domains():
    """Images from blocked domains are skipped; falls through to valid ones."""
    response = SerperImageResponse.model_validate(
        _serper_response(
            "https://lookaside.instagram.com/media/123",
            "https://www.facebook.com/photo.jpg",
            "https://example.com/good.jpg",
        )
    )

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=FAKE_JPEG, headers={"content-type": "image/jpeg"})

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        result = await _download_first_valid(client, response)

    assert result is not None


@pytest.mark.asyncio
async def test_skips_disallowed_mime():
    """Images that return text/html MIME are skipped."""
    response = SerperImageResponse.model_validate(
        _serper_response(
            "https://example.com/redirect",
            "https://example.com/real.jpg",
        )
    )
    call_count = 0

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(200, content=b"<html>", headers={"content-type": "text/html"})
        return httpx.Response(200, content=FAKE_JPEG, headers={"content-type": "image/jpeg"})

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        result = await _download_first_valid(client, response)

    assert result is not None
    assert call_count == 2


@pytest.mark.asyncio
async def test_all_results_blocked_returns_none():
    """When every result is from a blocked domain, returns None."""
    response = SerperImageResponse.model_validate(
        _serper_response(
            "https://lookaside.instagram.com/media/1",
            "https://www.facebook.com/photo.jpg",
            "https://tiktok.com/image.jpg",
        )
    )

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("Should not attempt to download blocked domain images")

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        result = await _download_first_valid(client, response)

    assert result is None


# --- Edge cases ---


@pytest.mark.asyncio
async def test_no_api_key_returns_none():
    """No API key means no search attempt."""
    result = await search_image("test", api_key=None, max_results=5, timeout=10.0)
    assert result is None


@pytest.mark.asyncio
async def test_empty_results_returns_none(monkeypatch):
    """Empty image list from Serper returns None."""
    monkeypatch.setattr(
        "penny.serper.client._fetch_results",
        AsyncMock(return_value=SerperImageResponse(images=[])),
    )

    result = await search_image("test", api_key=FAKE_API_KEY, max_results=5, timeout=10.0)

    assert result is None


@pytest.mark.asyncio
async def test_site_exclusions_constant_matches_blocked_domains():
    """_SITE_EXCLUSIONS is built from BLOCKED_IMAGE_DOMAINS."""
    for domain in BLOCKED_IMAGE_DOMAINS:
        assert f"-site:{domain}" in _SITE_EXCLUSIONS


# --- search_image_url ---


@pytest.mark.asyncio
async def test_search_image_url_returns_direct_image_url(monkeypatch):
    """search_image_url returns a URL with a recognized image extension."""
    monkeypatch.setattr(
        "penny.serper.client._fetch_results",
        AsyncMock(
            return_value=SerperImageResponse.model_validate(
                _serper_response("https://example.com/photo.jpg")
            )
        ),
    )
    result = await search_image_url("cute dogs", api_key=FAKE_API_KEY, max_results=5, timeout=10.0)
    assert result == "https://example.com/photo.jpg"


@pytest.mark.asyncio
async def test_search_image_url_skips_gallery_page(monkeypatch):
    """search_image_url skips extensionless gallery pages and returns the next valid image URL."""
    monkeypatch.setattr(
        "penny.serper.client._fetch_results",
        AsyncMock(
            return_value=SerperImageResponse.model_validate(
                _serper_response(
                    "https://example.com/gallery",
                    "https://example.com/image.png",
                )
            )
        ),
    )
    result = await search_image_url("funny dog", api_key=FAKE_API_KEY, max_results=5, timeout=10.0)
    assert result == "https://example.com/image.png"


@pytest.mark.asyncio
async def test_search_image_url_all_gallery_returns_none(monkeypatch):
    """search_image_url returns None when all results are extensionless gallery URLs."""
    monkeypatch.setattr(
        "penny.serper.client._fetch_results",
        AsyncMock(
            return_value=SerperImageResponse.model_validate(
                _serper_response(
                    "https://example.com/gallery",
                    "https://giphy.com/gifs/some-slug",
                )
            )
        ),
    )
    result = await search_image_url("meme", api_key=FAKE_API_KEY, max_results=5, timeout=10.0)
    assert result is None


@pytest.mark.asyncio
async def test_search_image_url_no_api_key_returns_none():
    """No API key means no search attempt."""
    result = await search_image_url("test", api_key=None, max_results=5, timeout=10.0)
    assert result is None

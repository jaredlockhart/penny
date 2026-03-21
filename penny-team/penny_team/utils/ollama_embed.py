"""Lightweight Ollama embedding client for dedup.

Calls Ollama's /api/embed endpoint via urllib (no external deps).
Returns None on failure so callers degrade to TCR-only matching.
"""

from __future__ import annotations

import json
import logging
import urllib.request

logger = logging.getLogger(__name__)

EMBED_ENDPOINT = "/api/embed"
EMBED_TIMEOUT = 30


def embed_batch(
    texts: list[str],
    ollama_url: str,
    model: str,
) -> list[list[float]] | None:
    """Embed a batch of texts via Ollama.

    Args:
        texts: List of texts to embed.
        ollama_url: Base Ollama API URL (e.g., http://host.docker.internal:11434).
        model: Embedding model name (e.g., embeddinggemma).

    Returns:
        List of embedding vectors (one per input text), or None on failure.
    """
    if not texts:
        return []

    url = f"{ollama_url.rstrip('/')}{EMBED_ENDPOINT}"
    payload = {"model": model, "input": texts}

    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=EMBED_TIMEOUT) as resp:
            result = json.loads(resp.read())
        embeddings = result.get("embeddings")
        if not embeddings or len(embeddings) != len(texts):
            logger.warning(
                "Ollama embed returned %s vectors for %d texts", len(embeddings or []), len(texts)
            )
            return None
        return [list(e) for e in embeddings]
    except (OSError, json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning("Ollama embed failed: %s", e)
        return None

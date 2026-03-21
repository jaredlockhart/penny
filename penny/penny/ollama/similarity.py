"""Penny-specific similarity operations.

Re-exports DedupStrategy and is_embedding_duplicate from the shared
similarity package for backward compatibility.  Keeps penny-specific
functions that depend on OllamaClient or penny's data model.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from similarity.dedup import DedupStrategy, is_embedding_duplicate
from similarity.embeddings import cosine_similarity, deserialize_embedding

if TYPE_CHECKING:
    from penny.ollama.client import OllamaClient

logger = logging.getLogger(__name__)

# Re-export so existing `from penny.ollama.similarity import DedupStrategy` works
__all__ = [
    "DedupStrategy",
    "compute_sentiment_score",
    "embed_text",
    "is_embedding_duplicate",
    "load_preference_vectors",
    "novelty_score",
]

# ── Safe embedding helper ─────────────────────────────────────────────────────


async def embed_text(
    client: OllamaClient | None,
    text: str,
) -> list[float] | None:
    """Embed a single text string.  Returns None if no client or on failure."""
    if client is None:
        return None
    try:
        vecs = await client.embed(text)
        return vecs[0]
    except Exception:
        logger.warning("Failed to embed text: %.60s", text)
        return None


# ── Sentiment scoring ────────────────────────────────────────────────────────


def compute_sentiment_score(
    vec: list[float],
    likes: list[list[float]],
    dislikes: list[list[float]],
) -> float:
    """Score = avg similarity to likes - avg similarity to dislikes.

    Higher scores indicate stronger alignment with liked topics.
    Returns 0.0 when both lists are empty.
    """
    like_score = 0.0
    if likes:
        like_score = sum(cosine_similarity(vec, lv) for lv in likes) / len(likes)
    dislike_score = 0.0
    if dislikes:
        dislike_score = sum(cosine_similarity(vec, dv) for dv in dislikes) / len(dislikes)
    return like_score - dislike_score


def load_preference_vectors(
    preferences: list,
    positive_valence: str,
    negative_valence: str,
) -> tuple[list[list[float]], list[list[float]]]:
    """Load like and dislike embedding vectors from preference records.

    Args:
        preferences: Preference records with .embedding and .valence attributes.
        positive_valence: Valence string for likes (e.g. "positive").
        negative_valence: Valence string for dislikes (e.g. "negative").

    Returns:
        (likes, dislikes) as lists of float vectors.
    """
    likes: list[list[float]] = []
    dislikes: list[list[float]] = []
    for p in preferences:
        if not p.embedding:
            continue
        vec = deserialize_embedding(p.embedding)
        if p.valence == positive_valence:
            likes.append(vec)
        elif p.valence == negative_valence:
            dislikes.append(vec)
    return likes, dislikes


def novelty_score(vec: list[float], recent_vecs: list[list[float]]) -> float:
    """1 - max similarity to any recent message. Higher = more novel."""
    if not recent_vecs:
        return 1.0
    max_sim = max(cosine_similarity(vec, rv) for rv in recent_vecs)
    return 1.0 - max_sim

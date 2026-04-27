"""Penny-specific similarity operations that depend on LlmClient or penny's data model.

Pure math primitives (cosine, TCR, dedup, serialization) live in the shared
`similarity/` package — import those directly, not via this module.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from similarity.embeddings import cosine_similarity, deserialize_embedding

from penny.llm.models import LlmError

if TYPE_CHECKING:
    from penny.llm.client import LlmClient

logger = logging.getLogger(__name__)

# ── Safe embedding helper ─────────────────────────────────────────────────────


async def embed_text(
    client: LlmClient | None,
    text: str,
) -> list[float] | None:
    """Embed a single text string.  Returns None if no client or on failure."""
    if client is None:
        return None
    try:
        vecs = await client.embed(text)
        return vecs[0]
    except LlmError:
        logger.warning("Failed to embed text: %.60s", text)
        return None


# ── Sentiment scoring ────────────────────────────────────────────────────────


def compute_mention_weighted_sentiment(
    vec: list[float],
    preferences: list,
    min_mentions: int,
) -> float:
    """Score a vector against mention-weighted positive and negative preferences.

    Only preferences with mention_count >= min_mentions are considered.
    Pass PREFERENCE_MENTION_THRESHOLD so the same gate controls both
    seed-topic eligibility and sentiment filtering.

    Returns weighted_avg_sim(positive) - weighted_avg_sim(negative).
    Returns 0.0 when no qualifying preferences exist.
    """
    pos_weighted: list[tuple[list[float], int]] = []
    neg_weighted: list[tuple[list[float], int]] = []
    for p in preferences:
        if not p.embedding or (p.mention_count or 0) < min_mentions:
            continue
        pref_vec = deserialize_embedding(p.embedding)
        weight = p.mention_count
        if p.valence == "positive":
            pos_weighted.append((pref_vec, weight))
        elif p.valence == "negative":
            neg_weighted.append((pref_vec, weight))

    def weighted_avg(items: list[tuple[list[float], int]]) -> float:
        if not items:
            return 0.0
        total_w = sum(w for _, w in items)
        return sum(cosine_similarity(vec, v) * w for v, w in items) / total_w

    return weighted_avg(pos_weighted) - weighted_avg(neg_weighted)


def novelty_score(vec: list[float], recent_vecs: list[list[float]]) -> float:
    """1 - max similarity to any recent message. Higher = more novel."""
    if not recent_vecs:
        return 1.0
    max_sim = max(cosine_similarity(vec, rv) for rv in recent_vecs)
    return 1.0 - max_sim


def centrality_score(vec: list[float], corpus_vecs: list[list[float]]) -> float:
    """Mean cosine similarity to a corpus. Higher = more centroid-like / generic.

    Inverse of novelty_score's intent: instead of measuring distance to the
    nearest neighbor, this measures how generally similar a message is to the
    corpus as a whole. Used to identify low-information centroid-magnet
    messages (greetings, "look at this" fillers, generic "Hey Penny what are
    some..." boilerplate) that match many unrelated queries equally well and
    should be down-weighted in similarity-based retrieval.
    """
    if not corpus_vecs:
        return 0.0
    return sum(cosine_similarity(vec, cv) for cv in corpus_vecs) / len(corpus_vecs)

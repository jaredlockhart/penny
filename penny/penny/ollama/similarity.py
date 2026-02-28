"""Shared similarity checking and deduplication operations.

Composes the low-level primitives from embeddings.py with OllamaClient
embedding calls.  All functions are stateless — thresholds and clients
are passed as parameters.
"""

from __future__ import annotations

import logging
import re
from enum import StrEnum
from typing import TYPE_CHECKING

from penny.ollama.embeddings import (
    cosine_similarity,
    deserialize_embedding,
    find_similar,
    token_containment_ratio,
    tokenize_entity_name,
)

if TYPE_CHECKING:
    from penny.ollama.client import OllamaClient

logger = logging.getLogger(__name__)

# ── Text normalization ────────────────────────────────────────────────────────

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_fact(fact: str) -> str:
    """Normalize a fact string for dedup comparison.

    Strips leading '- ', lowercases, and collapses whitespace so that
    near-duplicate facts with minor formatting differences are caught.
    """
    text = fact.strip().lstrip("-").strip()
    return _WHITESPACE_RE.sub(" ", text).lower()


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


# ── Relevance checking ────────────────────────────────────────────────────────


def check_relevance(
    candidate_vec: list[float],
    reference_vec: list[float],
    threshold: float,
) -> float | None:
    """Compare two pre-computed vectors and return score if relevant.

    Returns the cosine similarity score if >= threshold, else None.
    """
    score = cosine_similarity(candidate_vec, reference_vec)
    if score >= threshold:
        return score
    return None


# ── Embedding dedup ───────────────────────────────────────────────────────────


class DedupStrategy(StrEnum):
    """How to combine TCR and embedding signals for dedup."""

    EMBEDDING_ONLY = "embedding_only"
    TCR_AND_EMBEDDING = "tcr_and_embedding"
    TCR_OR_EMBEDDING = "tcr_or_embedding"


def is_embedding_duplicate(
    candidate_name: str,
    candidate_vec: list[float] | None,
    existing_items: list[tuple[str, bytes | None]],
    strategy: DedupStrategy,
    embedding_threshold: float,
    tcr_threshold: float = 0.0,
) -> int | None:
    """Check if a candidate is a semantic duplicate of any existing item.

    Args:
        candidate_name: Name/text of the candidate item.
        candidate_vec: Pre-computed embedding (None → no match possible).
        existing_items: List of (name, serialized_embedding_or_None).
        strategy: How to combine TCR and embedding signals.
        embedding_threshold: Cosine similarity threshold for embedding match.
        tcr_threshold: Token containment ratio threshold (ignored for
            EMBEDDING_ONLY).

    Returns:
        Index of the matching existing item, or None if no duplicate found.
    """
    candidate_tokens = tokenize_entity_name(candidate_name)

    for idx, (existing_name, existing_bytes) in enumerate(existing_items):
        tcr_pass = _check_tcr(
            candidate_name, candidate_tokens, existing_name, strategy, tcr_threshold
        )

        if strategy == DedupStrategy.TCR_AND_EMBEDDING and not tcr_pass:
            continue

        if candidate_vec is not None:
            embed_pass = _check_embedding(candidate_vec, existing_bytes, embedding_threshold)
        else:
            embed_pass = False

        if _is_match(strategy, tcr_pass, embed_pass):
            return idx

    return None


def _check_tcr(
    candidate_name: str,
    candidate_tokens: list[str],
    existing_name: str,
    strategy: DedupStrategy,
    tcr_threshold: float,
) -> bool:
    """Evaluate the TCR signal for one candidate–existing pair."""
    if strategy == DedupStrategy.EMBEDDING_ONLY:
        return False

    existing_tokens = tokenize_entity_name(existing_name)
    shorter_len = min(len(candidate_tokens), len(existing_tokens))

    # Single-token bypass: TCR is meaningless with one token (e.g. acronyms).
    if shorter_len <= 1 and strategy == DedupStrategy.TCR_AND_EMBEDDING:
        return True

    tcr = token_containment_ratio(candidate_name, existing_name)
    return tcr >= tcr_threshold


def _check_embedding(
    candidate_vec: list[float],
    existing_bytes: bytes | None,
    embedding_threshold: float,
) -> bool:
    """Evaluate the embedding similarity signal for one pair."""
    if existing_bytes is None:
        return False
    existing_vec = deserialize_embedding(existing_bytes)
    sim = cosine_similarity(candidate_vec, existing_vec)
    return sim >= embedding_threshold


def _is_match(strategy: DedupStrategy, tcr_pass: bool, embed_pass: bool) -> bool:
    """Combine TCR and embedding signals according to the strategy."""
    if strategy == DedupStrategy.EMBEDDING_ONLY:
        return embed_pass
    if strategy == DedupStrategy.TCR_AND_EMBEDDING:
        return embed_pass  # tcr_pass already enforced by caller skip
    # TCR_OR_EMBEDDING
    return tcr_pass or embed_pass


# ── Batch fact dedup ──────────────────────────────────────────────────────────


async def dedup_facts_by_embedding(
    client: OllamaClient | None,
    candidates: list[str],
    existing_with_embeddings: list[tuple[int, bytes]],
    threshold: float,
) -> tuple[list[str], list[int]]:
    """Deduplicate candidate facts against existing via embedding similarity.

    Args:
        client: Embedding model client (None → skip, return all candidates).
        candidates: New fact text strings to check.
        existing_with_embeddings: List of (id_or_index, serialized_embedding).
        threshold: Cosine similarity threshold for duplicate detection.

    Returns:
        Tuple of (surviving_candidates, matched_existing_ids).
    """
    if not client or not candidates or not existing_with_embeddings:
        return candidates, []

    existing_candidates = [
        (idx, deserialize_embedding(emb)) for idx, emb in existing_with_embeddings
    ]

    try:
        vecs = await client.embed(candidates)
    except Exception as e:
        logger.warning("Embedding dedup failed, keeping all candidates: %s", e)
        return candidates, []

    survivors: list[str] = []
    matched_ids: list[int] = []

    for fact_text, query_vec in zip(candidates, vecs, strict=True):
        matches = find_similar(query_vec, existing_candidates, top_k=1, threshold=threshold)
        if matches:
            matched_ids.append(matches[0][0])
            logger.debug("Skipping duplicate fact (embedding match): %s", fact_text[:50])
        else:
            survivors.append(fact_text)

    return survivors, matched_ids

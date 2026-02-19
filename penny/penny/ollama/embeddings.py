"""Embedding storage and similarity search utilities."""

from __future__ import annotations

import math
import struct
import unicodedata


def serialize_embedding(embedding: list[float]) -> bytes:
    """Serialize a float vector to a compact binary blob for SQLite storage."""
    return struct.pack(f"<{len(embedding)}f", *embedding)


def deserialize_embedding(data: bytes) -> list[float]:
    """Deserialize a binary blob back to a float vector."""
    count = len(data) // 4  # 4 bytes per float32
    return list(struct.unpack(f"<{count}f", data))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def find_similar(
    query: list[float],
    candidates: list[tuple[int, list[float]]],
    top_k: int = 5,
    threshold: float = 0.0,
) -> list[tuple[int, float]]:
    """
    Find the most similar candidates to a query embedding.

    Args:
        query: Query embedding vector
        candidates: List of (id, embedding) tuples to search
        top_k: Maximum number of results to return
        threshold: Minimum cosine similarity to include

    Returns:
        List of (id, similarity_score) tuples, sorted by descending similarity
    """
    scored = []
    for item_id, embedding in candidates:
        score = cosine_similarity(query, embedding)
        if score >= threshold:
            scored.append((item_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# Unicode punctuation that NFKD doesn't normalize to ASCII
_UNICODE_REPLACEMENTS = (
    ("\u2010", "-"),  # HYPHEN
    ("\u2011", "-"),  # NON-BREAKING HYPHEN
    ("\u2013", "-"),  # EN DASH
    ("\u2014", "-"),  # EM DASH
    ("\u2018", "'"),  # LEFT SINGLE QUOTATION MARK
    ("\u2019", "'"),  # RIGHT SINGLE QUOTATION MARK
    ("\u201c", '"'),  # LEFT DOUBLE QUOTATION MARK
    ("\u201d", '"'),  # RIGHT DOUBLE QUOTATION MARK
)


def normalize_unicode(text: str) -> str:
    """Normalize unicode variants to ASCII equivalents for token comparison.

    Applies NFKD decomposition, strips combining marks (accents), and replaces
    common unicode punctuation with ASCII equivalents.
    """
    normalized = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in normalized if not unicodedata.combining(c))
    for old, new in _UNICODE_REPLACEMENTS:
        stripped = stripped.replace(old, new)
    return stripped


def tokenize_entity_name(text: str) -> list[str]:
    """Tokenize an entity name for dedup comparison (unicode-normalized, lowercased)."""
    return normalize_unicode(text).lower().split()


def token_containment_ratio(name_a: str, name_b: str) -> float:
    """Fraction of the shorter name's tokens found in the longer name.

    Returns 1.0 when the shorter name is a complete token-subset of the longer.
    Used as a fast lexical signal for entity deduplication.
    """
    tokens_a = set(tokenize_entity_name(name_a))
    tokens_b = set(tokenize_entity_name(name_b))
    shorter, longer = (
        (tokens_a, tokens_b) if len(tokens_a) <= len(tokens_b) else (tokens_b, tokens_a)
    )
    if not shorter:
        return 0.0
    return len(shorter & longer) / len(shorter)


def build_entity_embed_text(name: str, facts: list[str]) -> str:
    """Build text for embedding an entity (name + facts).

    Args:
        name: Entity name
        facts: List of fact content strings

    Returns:
        Combined text suitable for embedding
    """
    if facts:
        return f"{name}: {'; '.join(facts)}"
    return name

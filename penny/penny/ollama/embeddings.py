"""Embedding storage and similarity search utilities."""

from __future__ import annotations

import math
import struct


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

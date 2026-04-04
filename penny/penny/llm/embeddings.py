"""Re-export shim — actual implementations live in the shared similarity package."""

from similarity.embeddings import (
    cosine_similarity,
    deserialize_embedding,
    serialize_embedding,
)

__all__ = [
    "cosine_similarity",
    "deserialize_embedding",
    "serialize_embedding",
]

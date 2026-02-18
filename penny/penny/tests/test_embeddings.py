"""Tests for embedding utilities and OllamaClient.embed()."""

import math

import pytest

from penny.ollama.embeddings import (
    cosine_similarity,
    deserialize_embedding,
    find_similar,
    serialize_embedding,
)


class TestSerializeDeserialize:
    """Tests for embedding serialization round-trip."""

    def test_round_trip(self):
        original = [0.1, 0.2, 0.3, 0.4, 0.5]
        blob = serialize_embedding(original)
        restored = deserialize_embedding(blob)
        assert len(restored) == len(original)
        for a, b in zip(original, restored, strict=True):
            assert a == pytest.approx(b, abs=1e-6)

    def test_compact_size(self):
        embedding = [0.0] * 768
        blob = serialize_embedding(embedding)
        assert len(blob) == 768 * 4  # 4 bytes per float32

    def test_empty_vector(self):
        blob = serialize_embedding([])
        assert deserialize_embedding(blob) == []


class TestCosineSimilarity:
    """Tests for cosine similarity computation."""

    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_similar_vectors(self):
        a = [1.0, 1.0, 0.0]
        b = [1.0, 0.0, 0.0]
        expected = 1.0 / math.sqrt(2)
        assert cosine_similarity(a, b) == pytest.approx(expected)

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestFindSimilar:
    """Tests for find_similar search function."""

    def test_returns_top_k(self):
        query = [1.0, 0.0, 0.0]
        candidates = [
            (1, [1.0, 0.0, 0.0]),  # identical
            (2, [0.9, 0.1, 0.0]),  # very similar
            (3, [0.0, 1.0, 0.0]),  # orthogonal
            (4, [0.5, 0.5, 0.0]),  # moderate
        ]
        results = find_similar(query, candidates, top_k=2)
        assert len(results) == 2
        assert results[0][0] == 1  # Most similar first
        assert results[1][0] == 2

    def test_threshold_filters(self):
        query = [1.0, 0.0]
        candidates = [
            (1, [1.0, 0.0]),  # similarity = 1.0
            (2, [0.0, 1.0]),  # similarity = 0.0
            (3, [-1.0, 0.0]),  # similarity = -1.0
        ]
        results = find_similar(query, candidates, threshold=0.5)
        assert len(results) == 1
        assert results[0][0] == 1

    def test_empty_candidates(self):
        assert find_similar([1.0], [], top_k=5) == []

    def test_descending_order(self):
        query = [1.0, 0.0, 0.0]
        candidates = [
            (1, [0.0, 1.0, 0.0]),
            (2, [0.5, 0.5, 0.0]),
            (3, [1.0, 0.0, 0.0]),
        ]
        results = find_similar(query, candidates)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


class TestOllamaClientEmbed:
    """Integration tests for OllamaClient.embed() with mock."""

    @pytest.mark.asyncio
    async def test_embed_single_text(self, mock_ollama):
        from penny.ollama.client import OllamaClient

        expected = [[0.1, 0.2, 0.3, 0.4]]
        mock_ollama.set_embed_handler(lambda model, input: expected)

        client = OllamaClient(
            api_url="http://localhost:11434",
            model="test-model",
            max_retries=1,
            retry_delay=0.0,
        )
        result = await client.embed("hello world", model="nomic-embed-text")

        assert result == expected
        assert len(mock_ollama.embed_requests) == 1
        assert mock_ollama.embed_requests[0]["model"] == "nomic-embed-text"
        assert mock_ollama.embed_requests[0]["input"] == "hello world"

    @pytest.mark.asyncio
    async def test_embed_batch(self, mock_ollama):
        from penny.ollama.client import OllamaClient

        expected = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_ollama.set_embed_handler(lambda model, input: expected)

        client = OllamaClient(
            api_url="http://localhost:11434",
            model="test-model",
            max_retries=1,
            retry_delay=0.0,
        )
        result = await client.embed(["a", "b", "c"], model="nomic-embed-text")

        assert len(result) == 3
        assert result == expected

    @pytest.mark.asyncio
    async def test_embed_default_mock(self, mock_ollama):
        """Default mock returns zero vectors."""
        from penny.ollama.client import OllamaClient

        client = OllamaClient(
            api_url="http://localhost:11434",
            model="test-model",
            max_retries=1,
            retry_delay=0.0,
        )
        result = await client.embed("test", model="nomic-embed-text")

        assert len(result) == 1
        assert all(v == 0.0 for v in result[0])

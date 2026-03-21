"""Tests for the shared similarity package (imported directly, not via penny shim).

Verifies that penny-team can import and use the shared similarity
primitives.  Penny's own test suite has comprehensive coverage of
edge cases — these tests focus on the functions used by penny-team
dedup (TCR, cosine similarity, serialization, dedup strategies).
"""

from __future__ import annotations

import json

import pytest

from similarity.dedup import DedupStrategy, is_embedding_duplicate
from similarity.embeddings import (
    cosine_similarity,
    deserialize_embedding,
    serialize_embedding,
    token_containment_ratio,
    tokenize_entity_name,
)


class TestTokenContainmentRatio:
    """TCR tests focused on the issue-dedup use case."""

    def test_identical_titles(self):
        a = "model returns empty content after tool calls"
        assert token_containment_ratio(a, a) == 1.0

    def test_subset_matches(self):
        short = "empty content after tool calls"
        long = "model returns empty content after 7 tool calls agent falls back"
        # All 5 tokens of short appear in long
        assert token_containment_ratio(short, long) == 1.0

    def test_partial_overlap(self):
        a = "search authentication error"
        b = "perplexity search fails with authentication timeout"
        tcr = token_containment_ratio(a, b)
        # "search" and "authentication" overlap, "error" does not → 2/3
        assert tcr == pytest.approx(2 / 3)

    def test_no_overlap(self):
        assert token_containment_ratio("empty content", "search timeout") == 0.0

    def test_empty_returns_zero(self):
        assert token_containment_ratio("", "anything") == 0.0


class TestTokenize:
    def test_lowercases(self):
        assert tokenize_entity_name("Model Empty") == ["model", "empty"]

    def test_hyphens_split(self):
        assert tokenize_entity_name("tool-calls") == ["tool", "calls"]


class TestCosineSimilarity:
    def test_identical(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


class TestSerializeRoundTrip:
    def test_round_trip(self):
        original = [0.1, 0.2, 0.3]
        restored = deserialize_embedding(serialize_embedding(original))
        for a, b in zip(original, restored, strict=True):
            assert a == pytest.approx(b, abs=1e-6)


class TestIsEmbeddingDuplicate:
    def test_tcr_or_embedding_matches_by_tcr(self):
        """TCR match with no embeddings still detects duplicate."""
        items: list[tuple[str, bytes | None]] = [
            ("model returns empty content after tool calls", None),
        ]
        result = is_embedding_duplicate(
            "empty content after tool calls",
            None,
            items,
            DedupStrategy.TCR_OR_EMBEDDING,
            embedding_threshold=0.9,
            tcr_threshold=0.6,
        )
        assert result == 0

    def test_embedding_only_with_high_similarity(self):
        vec = [1.0, 0.0, 0.0]
        items = [("unrelated name", serialize_embedding(vec))]
        result = is_embedding_duplicate(
            "also unrelated", vec, items, DedupStrategy.EMBEDDING_ONLY, 0.9
        )
        assert result == 0

    def test_no_match_returns_none(self):
        items: list[tuple[str, bytes | None]] = [("completely different topic", None)]
        result = is_embedding_duplicate(
            "search auth error",
            None,
            items,
            DedupStrategy.TCR_OR_EMBEDDING,
            embedding_threshold=0.9,
            tcr_threshold=0.6,
        )
        assert result is None


class TestEmbedBatch:
    """Tests for the Ollama embed utility (mocked HTTP)."""

    def test_successful_embed(self, monkeypatch):
        from penny_team.utils.ollama_embed import embed_batch

        def mock_urlopen(req, timeout=None):
            body = json.loads(req.data)
            assert body["model"] == "test-model"
            assert body["input"] == ["hello", "world"]

            class FakeResp:
                def read(self):
                    return json.dumps(
                        {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
                    ).encode()

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    pass

            return FakeResp()

        monkeypatch.setattr("penny_team.utils.ollama_embed.urllib.request.urlopen", mock_urlopen)
        result = embed_batch(["hello", "world"], "http://localhost:11434", "test-model")
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_empty_input_returns_empty(self):
        from penny_team.utils.ollama_embed import embed_batch

        assert embed_batch([], "http://localhost:11434", "model") == []

    def test_network_error_returns_none(self, monkeypatch):
        from penny_team.utils.ollama_embed import embed_batch

        def fail_urlopen(req, timeout=None):
            raise OSError("Connection refused")

        monkeypatch.setattr("penny_team.utils.ollama_embed.urllib.request.urlopen", fail_urlopen)
        result = embed_batch(["hello"], "http://localhost:11434", "model")
        assert result is None

    def test_mismatched_count_returns_none(self, monkeypatch):
        from penny_team.utils.ollama_embed import embed_batch

        def mock_urlopen(req, timeout=None):
            class FakeResp:
                def read(self):
                    return json.dumps({"embeddings": [[0.1]]}).encode()

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    pass

            return FakeResp()

        monkeypatch.setattr("penny_team.utils.ollama_embed.urllib.request.urlopen", mock_urlopen)
        result = embed_batch(["a", "b", "c"], "http://localhost:11434", "model")
        assert result is None

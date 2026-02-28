"""Tests for the shared similarity and dedup module."""

from __future__ import annotations

import math
from unittest.mock import AsyncMock

import pytest

from penny.ollama.embeddings import serialize_embedding
from penny.ollama.similarity import (
    DedupStrategy,
    check_relevance,
    dedup_facts_by_embedding,
    embed_text,
    is_embedding_duplicate,
    normalize_fact,
)

# ── normalize_fact ────────────────────────────────────────────────────────────


class TestNormalizeFact:
    def test_strips_bullet_prefix(self) -> None:
        assert normalize_fact("- foo bar") == "foo bar"

    def test_collapses_whitespace(self) -> None:
        assert normalize_fact("foo  bar   baz") == "foo bar baz"

    def test_lowercases(self) -> None:
        assert normalize_fact("Foo BAR Baz") == "foo bar baz"

    def test_strips_leading_whitespace(self) -> None:
        assert normalize_fact("  - hello world  ") == "hello world"

    def test_combined(self) -> None:
        assert normalize_fact("  -  Foo  BAR ") == "foo bar"


# ── embed_text ────────────────────────────────────────────────────────────────


class TestEmbedText:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_client(self) -> None:
        result = await embed_text(None, "hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_vector_on_success(self) -> None:
        client = AsyncMock()
        client.embed.return_value = [[1.0, 2.0, 3.0]]
        result = await embed_text(client, "hello")
        assert result == [1.0, 2.0, 3.0]

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self) -> None:
        client = AsyncMock()
        client.embed.side_effect = RuntimeError("boom")
        result = await embed_text(client, "hello")
        assert result is None


# ── check_relevance ───────────────────────────────────────────────────────────


class TestCheckRelevance:
    def test_above_threshold_returns_score(self) -> None:
        vec = [1.0, 0.0, 0.0]
        score = check_relevance(vec, vec, threshold=0.5)
        assert score is not None
        assert score == pytest.approx(1.0)

    def test_below_threshold_returns_none(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert check_relevance(a, b, threshold=0.5) is None

    def test_at_threshold_returns_score(self) -> None:
        # cos(45°) ≈ 0.707
        a = [1.0, 1.0, 0.0]
        b = [1.0, 0.0, 0.0]
        score = check_relevance(a, b, threshold=0.7)
        assert score is not None
        assert score == pytest.approx(1.0 / math.sqrt(2), abs=0.01)


# ── is_embedding_duplicate ────────────────────────────────────────────────────


def _make_item(name: str, vec: list[float] | None) -> tuple[str, bytes | None]:
    """Helper to build (name, serialized_embedding) tuple."""
    return (name, serialize_embedding(vec) if vec else None)


class TestIsEmbeddingDuplicate:
    def test_none_candidate_vec_returns_none(self) -> None:
        items = [_make_item("foo", [1.0, 0.0])]
        result = is_embedding_duplicate("foo", None, items, DedupStrategy.EMBEDDING_ONLY, 0.8)
        assert result is None

    def test_embedding_only_match(self) -> None:
        vec = [1.0, 0.0, 0.0]
        items = [_make_item("different name", vec)]
        result = is_embedding_duplicate("candidate", vec, items, DedupStrategy.EMBEDDING_ONLY, 0.9)
        assert result == 0

    def test_embedding_only_no_match(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        items = [_make_item("other", b)]
        result = is_embedding_duplicate("candidate", a, items, DedupStrategy.EMBEDDING_ONLY, 0.5)
        assert result is None

    def test_tcr_and_embedding_both_pass(self) -> None:
        vec = [1.0, 0.0, 0.0]
        items = [_make_item("star trek voyager", vec)]
        result = is_embedding_duplicate(
            "star trek", vec, items, DedupStrategy.TCR_AND_EMBEDDING, 0.9, 0.6
        )
        assert result == 0

    def test_tcr_and_embedding_tcr_fails(self) -> None:
        vec = [1.0, 0.0, 0.0]
        items = [_make_item("completely different", vec)]
        result = is_embedding_duplicate(
            "star trek", vec, items, DedupStrategy.TCR_AND_EMBEDDING, 0.9, 0.6
        )
        assert result is None

    def test_tcr_and_embedding_single_token_bypass(self) -> None:
        """Single-token names skip TCR requirement (e.g. acronyms)."""
        vec = [1.0, 0.0, 0.0]
        items = [_make_item("clps", vec)]
        result = is_embedding_duplicate(
            "foo", vec, items, DedupStrategy.TCR_AND_EMBEDDING, 0.9, 0.6
        )
        assert result == 0

    def test_tcr_or_embedding_tcr_only(self) -> None:
        """TCR passes but no embedding available — still a match in OR mode."""
        items = [("star trek voyager", None)]
        result = is_embedding_duplicate(
            "star trek", [1.0, 0.0], items, DedupStrategy.TCR_OR_EMBEDDING, 0.9, 0.6
        )
        assert result == 0

    def test_tcr_or_embedding_no_candidate_vec_still_matches_tcr(self) -> None:
        """TCR passes with None candidate_vec — still a match in OR mode."""
        items = [("star trek voyager", None)]
        result = is_embedding_duplicate(
            "star trek", None, items, DedupStrategy.TCR_OR_EMBEDDING, 0.9, 0.6
        )
        assert result == 0

    def test_tcr_or_embedding_embedding_only(self) -> None:
        """TCR fails but embedding passes — still a match in OR mode."""
        vec = [1.0, 0.0, 0.0]
        items = [_make_item("completely different", vec)]
        result = is_embedding_duplicate(
            "something else", vec, items, DedupStrategy.TCR_OR_EMBEDDING, 0.9, 0.6
        )
        assert result == 0

    def test_returns_first_match_index(self) -> None:
        vec = [1.0, 0.0, 0.0]
        items = [
            _make_item("no match", [0.0, 1.0, 0.0]),
            _make_item("match", vec),
        ]
        result = is_embedding_duplicate("candidate", vec, items, DedupStrategy.EMBEDDING_ONLY, 0.9)
        assert result == 1


# ── dedup_facts_by_embedding ──────────────────────────────────────────────────


class TestDedupFactsByEmbedding:
    @pytest.mark.asyncio
    async def test_no_client_returns_all(self) -> None:
        survivors, matched = await dedup_facts_by_embedding(
            None, ["fact1", "fact2"], [(0, serialize_embedding([1.0]))], 0.85
        )
        assert survivors == ["fact1", "fact2"]
        assert matched == []

    @pytest.mark.asyncio
    async def test_no_existing_returns_all(self) -> None:
        client = AsyncMock()
        survivors, matched = await dedup_facts_by_embedding(client, ["fact1"], [], 0.85)
        assert survivors == ["fact1"]
        assert matched == []
        client.embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_removes_duplicates(self) -> None:
        vec = [1.0, 0.0, 0.0]
        other = [0.0, 1.0, 0.0]
        client = AsyncMock()
        # First candidate matches existing, second doesn't
        client.embed.return_value = [vec, other]

        existing = [(42, serialize_embedding(vec))]
        survivors, matched = await dedup_facts_by_embedding(
            client, ["duplicate", "unique"], existing, 0.9
        )
        assert survivors == ["unique"]
        assert matched == [42]

    @pytest.mark.asyncio
    async def test_exception_returns_all(self) -> None:
        client = AsyncMock()
        client.embed.side_effect = RuntimeError("boom")
        existing = [(0, serialize_embedding([1.0]))]
        survivors, matched = await dedup_facts_by_embedding(client, ["fact1"], existing, 0.85)
        assert survivors == ["fact1"]
        assert matched == []

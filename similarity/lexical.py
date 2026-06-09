"""Lexical retrieval primitives and rank fusion.

Pure functions — no model, no I/O — shared by the memory store's stage-2
entity ranking and the prompt-validation harness.  The store fuses these
IDF-weighted lexical scores with embedding cosine via reciprocal-rank
fusion so instruction-shaped content (skills, recipes) that scores low on
absolute cosine but shares distinctive vocabulary with the query still
surfaces.
"""

from __future__ import annotations

import math
import re
from typing import TypeVar

_Key = TypeVar("_Key")

# Function words carry no topical signal — they collide lexically with every
# query, so they're stripped before scoring.  Deliberately small: the IDF
# weighting already down-weights common tokens; this only removes the worst
# offenders that would otherwise inflate coverage on any sentence.
STOPWORDS: frozenset[str] = frozenset(
    "a an the of to for and or but in on at is are be can you i me my we it that this with "
    "what how do does some more new when find tell them they about your please get got go "
    "going want need know see if then there here was were has have had will would should "
    "could just back into out up so it's i'm ya".split()
)

_TOKEN_RE = re.compile(r"[^a-z0-9 ]")


def tokens(text: str) -> set[str]:
    """Lowercase content tokens of ``text``, minus stopwords and short noise."""
    cleaned = _TOKEN_RE.sub(" ", text.lower())
    return {t for t in cleaned.split() if t not in STOPWORDS and len(t) > 2}


def idf(token_sets: list[set[str]]) -> dict[str, float]:
    """Inverse document frequency over a corpus of token sets (BM25-style)."""
    n = len(token_sets)
    document_frequency: dict[str, int] = {}
    for token_set in token_sets:
        for token in token_set:
            document_frequency[token] = document_frequency.get(token, 0) + 1
    return {t: math.log((n + 1) / (c + 0.5)) for t, c in document_frequency.items()}


def lexical_coverage(
    query_tokens: set[str], doc_tokens: set[str], idf_map: dict[str, float]
) -> float:
    """IDF-weighted fraction of the query's distinctive tokens present in doc.

    Tokens absent from ``idf_map`` (never seen in the corpus) fall back to a
    neutral weight so an out-of-corpus query term still counts.
    """
    if not query_tokens:
        return 0.0
    denominator = sum(idf_map.get(t, 0.5) for t in query_tokens)
    numerator = sum(idf_map.get(t, 0.5) for t in query_tokens if t in doc_tokens)
    return numerator / denominator if denominator else 0.0


def reciprocal_rank_fusion(rankings: list[list[_Key]], k: int = 60) -> list[_Key]:
    """Fuse several ranked key lists into one, by summed reciprocal rank.

    ``k`` is the standard RRF damping constant — it flattens the contribution
    of deep ranks so the head of each list dominates.  Keys are returned best
    first; a key absent from a ranking simply contributes nothing from it.
    Keys may be any hashable (entry-key strings in the harness, row ids in
    the store).
    """
    score: dict[_Key, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking):
            score[key] = score.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(score, key=lambda key: -score[key])

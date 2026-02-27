"""Tests for embedding utilities and OllamaClient.embed()."""

import math

import ollama
import pytest

from penny.ollama.embeddings import (
    build_entity_embed_text,
    cosine_similarity,
    deserialize_embedding,
    find_similar,
    serialize_embedding,
    token_containment_ratio,
    tokenize_entity_name,
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


class TestTokenizeEntityName:
    """Tests for entity name tokenization with dedup normalization."""

    def test_basic_split(self):
        assert tokenize_entity_name("Stanford University") == ["stanford", "university"]

    def test_underscore_to_space(self):
        assert tokenize_entity_name("agentic_ai_summit") == ["agentic", "ai", "summit"]

    def test_hyphen_to_space(self):
        assert tokenize_entity_name("kef-ls50-meta") == ["kef", "ls50", "meta"]

    def test_year_token_stripped(self):
        assert tokenize_entity_name("aamas 2026") == ["aamas"]

    def test_year_suffix_stripped(self):
        assert tokenize_entity_name("agentica2026") == ["agentica"]

    def test_non_year_number_preserved(self):
        assert tokenize_entity_name("nvidia gb10") == ["nvidia", "gb10"]

    def test_pure_year_only_returns_empty(self):
        assert tokenize_entity_name("2025") == []

    def test_mixed_separators(self):
        assert tokenize_entity_name("etsi_ai-data conference 2026") == [
            "etsi",
            "ai",
            "data",
            "conference",
        ]

    def test_19xx_year_not_stripped(self):
        """Years before 2000 are not stripped (regex is 20xx only)."""
        assert tokenize_entity_name("event 1999") == ["event", "1999"]


class TestTokenContainmentRatio:
    """Tests for TCR with normalized tokenization."""

    def test_underscore_variant_matches(self):
        """agentic_ai_summit vs agentic ai summit → TCR = 1.0"""
        assert token_containment_ratio("agentic_ai_summit", "agentic ai summit") == 1.0

    def test_year_variant_matches(self):
        """etsi ai data conference 2026 vs etsi ai data conference → TCR = 1.0"""
        assert (
            token_containment_ratio("etsi ai data conference 2026", "etsi ai data conference")
            == 1.0
        )

    def test_year_suffix_matches(self):
        """agentica2026 vs agentica → TCR = 1.0"""
        assert token_containment_ratio("agentica2026", "agentica") == 1.0

    def test_abbreviation_partial_overlap(self):
        """applied ai conference vs applied ai conf → TCR = 2/3 ≈ 0.67"""
        tcr = token_containment_ratio("applied ai conference", "applied ai conf")
        assert tcr == pytest.approx(2 / 3)

    def test_no_overlap(self):
        assert token_containment_ratio("kef ls50", "nvidia rtx") == 0.0

    def test_empty_returns_zero(self):
        """Pure year vs pure year → both empty after normalization → 0.0"""
        assert token_containment_ratio("2026", "2025") == 0.0


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
            model="nomic-embed-text",
            max_retries=1,
            retry_delay=0.0,
        )
        result = await client.embed("hello world")

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
            model="nomic-embed-text",
            max_retries=1,
            retry_delay=0.0,
        )
        result = await client.embed(["a", "b", "c"])

        assert len(result) == 3
        assert result == expected

    @pytest.mark.asyncio
    async def test_embed_default_mock(self, mock_ollama):
        """Default mock returns unit vectors."""
        from penny.ollama.client import OllamaClient

        client = OllamaClient(
            api_url="http://localhost:11434",
            model="nomic-embed-text",
            max_retries=1,
            retry_delay=0.0,
        )
        result = await client.embed("test")

        assert len(result) == 1
        assert result[0] == [1.0, 0.0, 0.0, 0.0]

    @pytest.mark.asyncio
    async def test_embed_404_raises_immediately_without_retry(self, mock_ollama):
        """A 404 (model not found) must raise immediately — no retries."""
        from penny.ollama.client import OllamaClient

        error_404 = ollama.ResponseError("model not found", status_code=404)
        call_count = 0

        def raising_handler(model: str, input: str | list[str]) -> list[list[float]]:
            nonlocal call_count
            call_count += 1
            raise error_404

        mock_ollama.set_embed_handler(raising_handler)

        client = OllamaClient(
            api_url="http://localhost:11434",
            model="missing-model",
            max_retries=3,
            retry_delay=0.0,
        )
        with pytest.raises(ollama.ResponseError) as exc_info:
            await client.embed("hello")

        assert exc_info.value.status_code == 404
        # Must have called embed exactly once — no retries on 404
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_embed_transient_error_retries(self, mock_ollama):
        """Non-404 errors should still be retried up to max_retries."""
        from penny.ollama.client import OllamaClient

        call_count = 0

        def flaky_handler(model: str, input: str | list[str]) -> list[list[float]]:
            nonlocal call_count
            call_count += 1
            raise ollama.ResponseError("server error", status_code=500)

        mock_ollama.set_embed_handler(flaky_handler)

        client = OllamaClient(
            api_url="http://localhost:11434",
            model="some-model",
            max_retries=3,
            retry_delay=0.0,
        )
        with pytest.raises(ollama.ResponseError):
            await client.embed("hello")

        # Should have retried all 3 times
        assert call_count == 3


class TestBuildEntityEmbedText:
    """Tests for build_entity_embed_text utility."""

    def test_name_only(self):
        assert build_entity_embed_text("kef ls50 meta", []) == "kef ls50 meta"

    def test_name_with_facts(self):
        result = build_entity_embed_text("kef ls50 meta", ["Costs $1,599", "Award winner"])
        assert result == "kef ls50 meta: Costs $1,599; Award winner"

    def test_single_fact(self):
        result = build_entity_embed_text("python", ["A programming language"])
        assert result == "python: A programming language"

    def test_name_with_tagline(self):
        result = build_entity_embed_text("genesis", [], tagline="british progressive rock band")
        assert result == "genesis (british progressive rock band)"

    def test_name_with_tagline_and_facts(self):
        result = build_entity_embed_text(
            "genesis",
            ["Formed in 1967", "From Surrey, England"],
            tagline="british progressive rock band",
        )
        assert (
            result
            == "genesis (british progressive rock band): Formed in 1967; From Surrey, England"
        )

    def test_tagline_none_unchanged(self):
        result = build_entity_embed_text("kef ls50 meta", ["Costs $1,599"], tagline=None)
        assert result == "kef ls50 meta: Costs $1,599"


class TestDatabaseEmbeddingMethods:
    """Tests for database embedding storage and retrieval."""

    def _setup_db(self, tmp_path):
        """Create a test database with tables and migrations."""
        from penny.database import Database
        from penny.database.migrate import migrate

        db_path = str(tmp_path / "test.db")
        db = Database(db_path)
        db.create_tables()
        migrate(db_path)
        return db

    def test_add_fact_with_embedding(self, tmp_path):
        db = self._setup_db(tmp_path)
        entity = db.entities.get_or_create("+1234", "test entity")
        assert entity is not None and entity.id is not None

        embedding = serialize_embedding([0.1, 0.2, 0.3])
        fact = db.facts.add(entity.id, "test fact", embedding=embedding)

        assert fact is not None
        assert fact.embedding == embedding
        restored = deserialize_embedding(fact.embedding)
        assert len(restored) == 3

    def test_add_fact_without_embedding(self, tmp_path):
        db = self._setup_db(tmp_path)
        entity = db.entities.get_or_create("+1234", "test entity")
        assert entity is not None and entity.id is not None

        fact = db.facts.add(entity.id, "test fact")
        assert fact is not None
        assert fact.embedding is None

    def test_update_entity_embedding(self, tmp_path):
        db = self._setup_db(tmp_path)
        entity = db.entities.get_or_create("+1234", "test entity")
        assert entity is not None and entity.id is not None
        assert entity.embedding is None

        embedding = serialize_embedding([0.1, 0.2, 0.3])
        db.entities.update_embedding(entity.id, embedding)

        # Verify update
        entities = db.entities.get_for_user("+1234")
        assert len(entities) == 1
        assert entities[0].embedding == embedding

    def test_update_fact_embedding(self, tmp_path):
        db = self._setup_db(tmp_path)
        entity = db.entities.get_or_create("+1234", "test entity")
        assert entity is not None and entity.id is not None

        fact = db.facts.add(entity.id, "test fact")
        assert fact is not None and fact.id is not None

        embedding = serialize_embedding([0.7, 0.8])
        db.facts.update_embedding(fact.id, embedding)

        facts = db.facts.get_for_entity(entity.id)
        assert len(facts) == 1
        assert facts[0].embedding == embedding

    def test_get_entities_without_embeddings(self, tmp_path):
        db = self._setup_db(tmp_path)
        # Create two entities
        e1 = db.entities.get_or_create("+1234", "entity a")
        e2 = db.entities.get_or_create("+1234", "entity b")
        assert e1 is not None and e1.id is not None
        assert e2 is not None and e2.id is not None

        # Both should be returned (no embeddings)
        without = db.entities.get_without_embeddings(limit=10)
        assert len(without) == 2

        # Give one an embedding
        db.entities.update_embedding(e1.id, serialize_embedding([0.1]))

        # Only the other should be returned
        without = db.entities.get_without_embeddings(limit=10)
        assert len(without) == 1
        assert without[0].name == "entity b"

    def test_get_facts_without_embeddings(self, tmp_path):
        db = self._setup_db(tmp_path)
        entity = db.entities.get_or_create("+1234", "test")
        assert entity is not None and entity.id is not None

        f1 = db.facts.add(entity.id, "fact one")
        f2 = db.facts.add(entity.id, "fact two", embedding=serialize_embedding([0.1]))
        assert f1 is not None and f2 is not None

        without = db.facts.get_without_embeddings(limit=10)
        assert len(without) == 1
        assert without[0].content == "fact one"


class TestDatabaseEngagementMethods:
    """Tests for database engagement storage and retrieval."""

    def _setup_db(self, tmp_path):
        """Create a test database with tables and migrations."""
        from penny.database import Database
        from penny.database.migrate import migrate

        db_path = str(tmp_path / "test.db")
        db = Database(db_path)
        db.create_tables()
        migrate(db_path)
        return db

    def test_add_and_get_entity_engagements(self, tmp_path):
        from penny.constants import PennyConstants

        db = self._setup_db(tmp_path)
        entity = db.entities.get_or_create("+1234", "kef ls50 meta")
        assert entity is not None and entity.id is not None

        # Add two engagements for this entity
        e1 = db.engagements.add(
            user="+1234",
            engagement_type=PennyConstants.EngagementType.USER_SEARCH,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.6,
            entity_id=entity.id,
        )
        e2 = db.engagements.add(
            user="+1234",
            engagement_type=PennyConstants.EngagementType.MESSAGE_MENTION,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.2,
            entity_id=entity.id,
        )
        assert e1 is not None and e2 is not None

        # Add an engagement for a different entity
        other = db.entities.get_or_create("+1234", "wharfedale linton")
        assert other is not None and other.id is not None
        db.engagements.add(
            user="+1234",
            engagement_type=PennyConstants.EngagementType.MESSAGE_MENTION,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.2,
            entity_id=other.id,
        )

        # get_entity_engagements should return only the two for kef
        engagements = db.engagements.get_for_entity("+1234", entity.id)
        assert len(engagements) == 2
        assert all(e.entity_id == entity.id for e in engagements)

    def test_get_user_engagements(self, tmp_path):
        from penny.constants import PennyConstants

        db = self._setup_db(tmp_path)
        entity = db.entities.get_or_create("+1234", "test entity")
        assert entity is not None and entity.id is not None

        # Add engagements for two different users
        db.engagements.add(
            user="+1234",
            engagement_type=PennyConstants.EngagementType.USER_SEARCH,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.6,
            entity_id=entity.id,
        )
        db.engagements.add(
            user="+5678",
            engagement_type=PennyConstants.EngagementType.USER_SEARCH,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.8,
            entity_id=entity.id,
        )

        # get_user_engagements filters by user
        user1_engagements = db.engagements.get_for_user("+1234")
        assert len(user1_engagements) == 1
        assert user1_engagements[0].user == "+1234"

        user2_engagements = db.engagements.get_for_user("+5678")
        assert len(user2_engagements) == 1
        assert user2_engagements[0].user == "+5678"

    def test_delete_entity_cascades_engagements(self, tmp_path):
        from penny.constants import PennyConstants

        db = self._setup_db(tmp_path)
        entity = db.entities.get_or_create("+1234", "test entity")
        assert entity is not None and entity.id is not None

        db.engagements.add(
            user="+1234",
            engagement_type=PennyConstants.EngagementType.USER_SEARCH,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=1.0,
            entity_id=entity.id,
        )
        db.engagements.add(
            user="+1234",
            engagement_type=PennyConstants.EngagementType.USER_SEARCH,
            valence=PennyConstants.EngagementValence.POSITIVE,
            strength=0.6,
            entity_id=entity.id,
        )

        # Verify engagements exist
        assert len(db.engagements.get_for_entity("+1234", entity.id)) == 2

        # Delete entity
        deleted = db.entities.delete(entity.id)
        assert deleted is True

        # Engagements should be gone
        assert len(db.engagements.get_for_entity("+1234", entity.id)) == 0

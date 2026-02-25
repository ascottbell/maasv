"""
Tests for per-operation LLM provider routing (MAASV-39).

Verifies that get_llm_for() returns per-operation overrides when configured,
and falls back to the default LLM when not.
"""

import pytest


# ============================================================================
# MOCK PROVIDERS
# ============================================================================


class MockEmbedProvider:
    """Deterministic embeddings for testing."""

    def __init__(self, dims=64):
        self.dims = dims

    def embed(self, text: str) -> list[float]:
        import hashlib

        h = hashlib.sha256(text.encode()).digest()
        vec = [b / 255.0 for b in h]
        while len(vec) < self.dims:
            vec.extend(vec)
        return vec[: self.dims]

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)


class MockLLMProvider:
    """Mock LLM with an identifier for routing verification."""

    def __init__(self, name: str = "default"):
        self.name = name
        self.call_count = 0

    def call(self, messages, model, max_tokens, source=""):
        self.call_count += 1
        return "[]"


# ============================================================================
# TESTS
# ============================================================================


class TestGetLLMFor:
    """Test get_llm_for() routing logic."""

    def test_fallback_to_default_llm(self, tmp_path):
        """When no per-operation override, get_llm_for() returns the default."""
        import maasv
        from maasv.config import MaasvConfig

        config = MaasvConfig(
            db_path=tmp_path / "test.db",
            embed_dims=64,
        )
        default_llm = MockLLMProvider("default")
        maasv.init(config=config, llm=default_llm, embed=MockEmbedProvider(dims=64))

        assert maasv.get_llm_for("extraction") is default_llm
        assert maasv.get_llm_for("inference") is default_llm
        assert maasv.get_llm_for("review") is default_llm

    def test_per_operation_override(self, tmp_path):
        """Per-operation LLM overrides take precedence over the default."""
        import maasv
        from maasv.config import MaasvConfig

        extraction_llm = MockLLMProvider("extraction-specific")
        review_llm = MockLLMProvider("review-specific")

        config = MaasvConfig(
            db_path=tmp_path / "test.db",
            embed_dims=64,
            extraction_llm=extraction_llm,
            review_llm=review_llm,
        )
        default_llm = MockLLMProvider("default")
        maasv.init(config=config, llm=default_llm, embed=MockEmbedProvider(dims=64))

        assert maasv.get_llm_for("extraction") is extraction_llm
        assert maasv.get_llm_for("extraction").name == "extraction-specific"

        # inference has no override, falls back to default
        assert maasv.get_llm_for("inference") is default_llm

        assert maasv.get_llm_for("review") is review_llm
        assert maasv.get_llm_for("review").name == "review-specific"

    def test_unknown_operation_falls_back(self, tmp_path):
        """Unknown operation names fall back to the default LLM."""
        import maasv
        from maasv.config import MaasvConfig

        config = MaasvConfig(
            db_path=tmp_path / "test.db",
            embed_dims=64,
        )
        default_llm = MockLLMProvider("default")
        maasv.init(config=config, llm=default_llm, embed=MockEmbedProvider(dims=64))

        assert maasv.get_llm_for("unknown_op") is default_llm

    def test_per_operation_with_different_models(self, tmp_path):
        """Config supports different model names per operation."""
        from maasv.config import MaasvConfig

        config = MaasvConfig(
            db_path=tmp_path / "test.db",
            embed_dims=64,
            extraction_model="claude-haiku-4-5-20251001",
            inference_model="claude-haiku-4-5-20251001",
            review_model="claude-sonnet-4-5-20250514",
        )

        assert config.extraction_model == "claude-haiku-4-5-20251001"
        assert config.review_model == "claude-sonnet-4-5-20250514"

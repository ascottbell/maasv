"""
Tests for embedding model tracking in db_meta.

Verifies that:
1. The model is recorded on first init
2. Re-init with the same model succeeds
3. Re-init with a different model raises RuntimeError
"""

import hashlib
import tempfile
from pathlib import Path

import pytest


class MockEmbedProvider:
    def __init__(self, dims=64):
        self.dims = dims

    def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        vec = [b / 255.0 for b in h]
        while len(vec) < self.dims:
            vec.extend(vec)
        return vec[:self.dims]

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)


class MockLLMProvider:
    def call(self, messages, model="", max_tokens=1024, source=""):
        return "[]"


def _init_maasv(db_path: Path, embed_model: str = "test-model", embed_dims: int = 64):
    """Helper to init maasv with a given model name."""
    import maasv
    from maasv.config import MaasvConfig

    # Reset global state so we can re-init
    maasv._config = None
    maasv._llm = None
    maasv._embed = None
    maasv._initialized = False

    config = MaasvConfig(
        db_path=db_path,
        embed_dims=embed_dims,
        embed_model=embed_model,
        cross_encoder_enabled=False,
    )

    maasv.init(config=config, llm=MockLLMProvider(), embed=MockEmbedProvider(dims=embed_dims))


class TestEmbedModelTracking:
    def test_model_recorded_on_first_init(self):
        """First init records the model in db_meta."""
        tmpdir = tempfile.mkdtemp(prefix="maasv_embed_track_")
        db_path = Path(tmpdir) / "test.db"

        _init_maasv(db_path, embed_model="test-model-v1")

        # Verify it's in the DB
        from maasv.core.db import get_db
        db = get_db()
        row = db.execute("SELECT value FROM db_meta WHERE key = 'embed_model'").fetchone()
        assert row is not None
        assert row["value"] == "test-model-v1"

        dims_row = db.execute("SELECT value FROM db_meta WHERE key = 'embed_dims'").fetchone()
        assert dims_row is not None
        assert dims_row["value"] == "64"
        db.close()

    def test_same_model_reinit_succeeds(self):
        """Re-init with the same model should work fine."""
        tmpdir = tempfile.mkdtemp(prefix="maasv_embed_track_")
        db_path = Path(tmpdir) / "test.db"

        _init_maasv(db_path, embed_model="test-model-v1")
        # Re-init with same model — should not raise
        _init_maasv(db_path, embed_model="test-model-v1")

    def test_different_model_raises(self):
        """Re-init with a different model should raise RuntimeError."""
        tmpdir = tempfile.mkdtemp(prefix="maasv_embed_track_")
        db_path = Path(tmpdir) / "test.db"

        _init_maasv(db_path, embed_model="model-alpha")

        with pytest.raises(RuntimeError, match="Embedding model mismatch"):
            _init_maasv(db_path, embed_model="model-beta")

    def test_different_dims_raises(self):
        """Re-init with different dimensions should raise RuntimeError."""
        tmpdir = tempfile.mkdtemp(prefix="maasv_embed_track_")
        db_path = Path(tmpdir) / "test.db"

        _init_maasv(db_path, embed_model="same-model", embed_dims=64)

        with pytest.raises(RuntimeError, match="dimension mismatch"):
            _init_maasv(db_path, embed_model="same-model", embed_dims=128)

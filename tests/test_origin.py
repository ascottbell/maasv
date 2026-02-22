"""
Tests for origin provenance fields (origin, origin_interface).

Covers: storage, retrieval filtering, supersede inheritance, API endpoints,
and migration compatibility (null origins on existing data).
"""

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient


# ============================================================================
# MOCK PROVIDERS
# ============================================================================

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


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def client():
    """TestClient with mock providers for origin tests."""
    import maasv
    from maasv.config import MaasvConfig

    tmpdir = tempfile.mkdtemp(prefix="maasv_origin_")
    db_path = Path(tmpdir) / "origin_test.db"

    config = MaasvConfig(
        db_path=db_path,
        embed_dims=64,
        extraction_model="test-model",
        inference_model="test-model",
        review_model="test-model",
        cross_encoder_enabled=False,
    )

    maasv.init(config=config, llm=MockLLMProvider(), embed=MockEmbedProvider(dims=64))

    with patch("maasv.server.main._init_maasv"):
        from maasv.server.main import app
        with TestClient(app) as tc:
            yield tc


# ============================================================================
# STORE WITH ORIGIN
# ============================================================================

class TestStoreWithOrigin:
    _claude_code_id: str = ""
    _chatgpt_id: str = ""
    _no_origin_id: str = ""

    def test_store_with_origin(self, client):
        """Store memory with origin and origin_interface."""
        r = client.post("/v1/memory/store", json={
            "content": "Adam prefers dark mode in all editors",
            "category": "preference",
            "subject": "Adam",
            "source": "conversation",
            "origin": "claude",
            "origin_interface": "claude-code",
        })
        assert r.status_code == 200
        TestStoreWithOrigin._claude_code_id = r.json()["memory_id"]

    def test_store_different_origin(self, client):
        """Store from a different origin."""
        r = client.post("/v1/memory/store", json={
            "content": "Adam uses vim keybindings in all editors",
            "category": "preference",
            "subject": "Adam",
            "source": "conversation",
            "origin": "openai",
            "origin_interface": "chatgpt-ios",
        })
        assert r.status_code == 200
        TestStoreWithOrigin._chatgpt_id = r.json()["memory_id"]

    def test_store_without_origin(self, client):
        """Store without origin (backward compat — should be null)."""
        r = client.post("/v1/memory/store", json={
            "content": "Adam lives on the Upper West Side",
            "category": "identity",
            "subject": "Adam",
            "source": "manual",
        })
        assert r.status_code == 200
        TestStoreWithOrigin._no_origin_id = r.json()["memory_id"]

    def test_origin_persisted(self, client):
        """Verify origin fields are stored and returned."""
        r = client.get(f"/v1/memory/{TestStoreWithOrigin._claude_code_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["origin"] == "claude"
        assert data["origin_interface"] == "claude-code"

    def test_no_origin_is_null(self, client):
        """Verify null origin when not provided."""
        r = client.get(f"/v1/memory/{TestStoreWithOrigin._no_origin_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["origin"] is None
        assert data["origin_interface"] is None


# ============================================================================
# SEARCH WITH ORIGIN FILTER
# ============================================================================

class TestSearchByOrigin:
    def test_search_filter_by_origin(self, client):
        """Search with origin filter returns only matching memories."""
        r = client.post("/v1/memory/search", json={
            "query": "Adam editor preferences",
            "limit": 10,
            "origin": "claude",
        })
        assert r.status_code == 200
        results = r.json()["results"]
        for mem in results:
            assert mem.get("origin") == "claude"

    def test_search_filter_by_origin_interface(self, client):
        """Search with origin_interface filter."""
        r = client.post("/v1/memory/search", json={
            "query": "Adam editor preferences",
            "limit": 10,
            "origin_interface": "chatgpt-ios",
        })
        assert r.status_code == 200
        results = r.json()["results"]
        for mem in results:
            assert mem.get("origin_interface") == "chatgpt-ios"

    def test_search_no_origin_filter_returns_all(self, client):
        """Search without origin filter returns from all origins."""
        r = client.post("/v1/memory/search", json={
            "query": "Adam",
            "limit": 10,
        })
        assert r.status_code == 200
        # Should have memories from multiple origins
        results = r.json()["results"]
        origins = {m.get("origin") for m in results}
        # At least null and one named origin
        assert len(origins) >= 2 or len(results) >= 2


# ============================================================================
# SUPERSEDE INHERITS ORIGIN
# ============================================================================

class TestSupersedeOrigin:
    def test_supersede_inherits_origin(self, client):
        """Superseding a memory inherits origin from the old one."""
        # Store with origin
        r = client.post("/v1/memory/store", json={
            "content": "Original fact from Claude Desktop",
            "category": "context",
            "source": "conversation",
            "origin": "claude",
            "origin_interface": "claude-desktop",
        })
        assert r.status_code == 200
        old_id = r.json()["memory_id"]

        # Supersede without specifying origin
        r = client.post("/v1/memory/supersede", json={
            "old_id": old_id,
            "new_content": "Updated fact that supersedes the old one",
        })
        assert r.status_code == 200
        new_id = r.json()["memory_id"]

        # Verify new memory inherited origin
        r = client.get(f"/v1/memory/{new_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["origin"] == "claude"
        assert data["origin_interface"] == "claude-desktop"


# ============================================================================
# GRAPH RELATIONSHIPS WITH ORIGIN
# ============================================================================

class TestGraphOrigin:
    def test_relationship_with_origin(self, client):
        """Create relationship with origin fields."""
        # Create two entities
        r1 = client.post("/v1/graph/entities", json={
            "name": "OriginTestPerson",
            "entity_type": "person",
        })
        assert r1.status_code == 200
        person_id = r1.json()["id"]

        r2 = client.post("/v1/graph/entities", json={
            "name": "OriginTestProject",
            "entity_type": "project",
        })
        assert r2.status_code == 200
        project_id = r2.json()["id"]

        # Create relationship with origin
        r3 = client.post("/v1/graph/relationships", json={
            "subject_id": person_id,
            "predicate": "works_on",
            "object_id": project_id,
            "confidence": 0.9,
            "source": "extracted",
            "origin": "claude",
            "origin_interface": "codex",
        })
        assert r3.status_code == 200
        assert "relationship_id" in r3.json()

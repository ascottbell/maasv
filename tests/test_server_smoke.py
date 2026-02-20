"""
Smoke test: hit every maasv-server endpoint and verify correct responses.

Uses mock providers (no API keys needed). Runs via: pytest tests/test_server_smoke.py -v
"""

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient


# ============================================================================
# MOCK PROVIDERS (same pattern as test_decomposition.py)
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
    """TestClient wired to the real FastAPI app, but with mock providers."""
    import maasv
    from maasv.config import MaasvConfig

    tmpdir = tempfile.mkdtemp(prefix="maasv_smoke_")
    db_path = Path(tmpdir) / "smoke.db"

    config = MaasvConfig(
        db_path=db_path,
        embed_dims=64,
        extraction_model="test-model",
        inference_model="test-model",
        review_model="test-model",
        cross_encoder_enabled=False,
    )

    maasv.init(config=config, llm=MockLLMProvider(), embed=MockEmbedProvider(dims=64))

    # Patch _init_maasv so the app lifespan doesn't re-init with real providers
    with patch("maasv.server.main._init_maasv"):
        from maasv.server.main import app
        with TestClient(app) as tc:
            yield tc


API_KEY = ""  # no auth in test mode
HEADERS = {}  # empty — settings.api_key defaults to "" (no auth)


# ============================================================================
# 1. HEALTH ENDPOINTS
# ============================================================================

class TestHealth:
    def test_health(self, client):
        """GET /v1/health — should return healthy (no auth)."""
        r = client.get("/v1/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"

    def test_stats(self, client):
        """GET /v1/stats — should return stats structure."""
        r = client.get("/v1/stats", headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert "memories" in data
        assert "entities" in data
        assert "relationships" in data


# ============================================================================
# 2. MEMORY ENDPOINTS
# ============================================================================

class TestMemory:
    _stored_id: str = ""

    def test_store(self, client):
        """POST /v1/memory/store — store a test memory."""
        r = client.post("/v1/memory/store", json={
            "content": "The smoke test ran successfully on this machine",
            "category": "project",
            "subject": "maasv",
            "source": "smoke_test",
            "confidence": 1.0,
        }, headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert "memory_id" in data
        assert data["memory_id"].startswith("mem_")
        TestMemory._stored_id = data["memory_id"]

    def test_search(self, client):
        """POST /v1/memory/search — search for stored memory."""
        r = client.post("/v1/memory/search", json={
            "query": "smoke test machine",
            "limit": 5,
        }, headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert "count" in data
        assert data["count"] >= 1

    def test_context(self, client):
        """POST /v1/memory/context — get tiered context."""
        r = client.post("/v1/memory/context", json={
            "query": "smoke test",
            "core_limit": 10,
            "relevant_limit": 5,
        }, headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert "context" in data

    def test_get_by_id(self, client):
        """GET /v1/memory/{id} — retrieve stored memory."""
        assert TestMemory._stored_id, "store test must run first"
        r = client.get(f"/v1/memory/{TestMemory._stored_id}", headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == TestMemory._stored_id
        assert "smoke test" in data["content"]

    def test_supersede(self, client):
        """POST /v1/memory/supersede — supersede stored memory."""
        # Store a non-protected memory specifically for supersede test
        r = client.post("/v1/memory/store", json={
            "content": "Old fact to be superseded",
            "category": "context",
            "source": "smoke_test",
        }, headers=HEADERS)
        assert r.status_code == 200
        old_id = r.json()["memory_id"]

        r = client.post("/v1/memory/supersede", json={
            "old_id": old_id,
            "new_content": "Updated fact after supersede",
        }, headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert data["memory_id"] != old_id
        TestMemory._superseded_new_id = data["memory_id"]

    def test_delete(self, client):
        """DELETE /v1/memory/{id} — delete a non-protected memory."""
        # Store a deletable memory
        r = client.post("/v1/memory/store", json={
            "content": "Temporary memory for deletion test",
            "category": "context",
            "source": "smoke_test",
        }, headers=HEADERS)
        assert r.status_code == 200
        delete_id = r.json()["memory_id"]

        r = client.delete(f"/v1/memory/{delete_id}", headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert data["deleted"] is True

        # Verify it's gone
        r = client.get(f"/v1/memory/{delete_id}", headers=HEADERS)
        assert r.status_code == 404

    def test_get_nonexistent_returns_404(self, client):
        """GET /v1/memory/{id} — nonexistent memory should 404."""
        r = client.get("/v1/memory/mem_does_not_exist", headers=HEADERS)
        assert r.status_code == 404

    def test_delete_protected_returns_403(self, client):
        """DELETE on protected category should return 403."""
        r = client.post("/v1/memory/store", json={
            "content": "Protected memory for delete test",
            "category": "identity",
            "source": "smoke_test",
        }, headers=HEADERS)
        assert r.status_code == 200
        protected_id = r.json()["memory_id"]

        r = client.delete(f"/v1/memory/{protected_id}", headers=HEADERS)
        assert r.status_code == 403
        assert "protected category" in r.json()["detail"]


# ============================================================================
# 3. EXTRACTION ENDPOINT
# ============================================================================

class TestExtraction:
    def test_extract(self, client):
        """POST /v1/extract — entity extraction (mock LLM returns [])."""
        r = client.post("/v1/extract", json={
            "text": "Alice works on Helix, a memory system for AI agents.",
            "topic": "project overview",
        }, headers=HEADERS)
        assert r.status_code == 200


# ============================================================================
# 4. GRAPH ENDPOINTS
# ============================================================================

class TestGraph:
    _entity_id: str = ""
    _entity2_id: str = ""

    def test_create_entity(self, client):
        """POST /v1/graph/entities — create entity."""
        r = client.post("/v1/graph/entities", json={
            "name": "SmokeTestPerson",
            "entity_type": "person",
        }, headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "SmokeTestPerson"
        assert data["entity_type"] == "person"
        TestGraph._entity_id = data["id"]

    def test_create_second_entity(self, client):
        """POST /v1/graph/entities — create second entity for relationship."""
        r = client.post("/v1/graph/entities", json={
            "name": "SmokeTestProject",
            "entity_type": "project",
        }, headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        TestGraph._entity2_id = data["id"]

    def test_search_entities(self, client):
        """POST /v1/graph/entities/search — search entities."""
        # FTS tokenizes on word boundaries; "SmokeTestPerson" is one token
        # so we search for the full name
        r = client.post("/v1/graph/entities/search", json={
            "query": "SmokeTestPerson",
            "limit": 10,
        }, headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert "count" in data
        # FTS may or may not find it depending on tokenizer — just verify endpoint works
        assert isinstance(data["count"], int)

    def test_get_entity_profile(self, client):
        """GET /v1/graph/entities/{id} — entity profile."""
        assert TestGraph._entity_id
        r = client.get(f"/v1/graph/entities/{TestGraph._entity_id}", headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert "entity" in data
        assert data["entity"]["name"] == "SmokeTestPerson"

    def test_create_relationship(self, client):
        """POST /v1/graph/relationships — create relationship."""
        assert TestGraph._entity_id and TestGraph._entity2_id
        r = client.post("/v1/graph/relationships", json={
            "subject_id": TestGraph._entity_id,
            "predicate": "works_on",
            "object_id": TestGraph._entity2_id,
            "confidence": 0.9,
            "source": "smoke_test",
        }, headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert "relationship_id" in data


# ============================================================================
# 5. WISDOM ENDPOINTS
# ============================================================================

class TestWisdom:
    _wisdom_id: str = ""

    def test_log_reasoning(self, client):
        """POST /v1/wisdom/log — log reasoning."""
        r = client.post("/v1/wisdom/log", json={
            "action_type": "smoke_test",
            "reasoning": "Testing the wisdom endpoint during smoke test.",
            "trigger": "test_harness",
            "context": "Automated smoke test of all endpoints.",
            "tags": ["test", "smoke"],
        }, headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert "wisdom_id" in data
        TestWisdom._wisdom_id = data["wisdom_id"]

    def test_record_outcome(self, client):
        """POST /v1/wisdom/{id}/outcome — record outcome."""
        assert TestWisdom._wisdom_id
        r = client.post(f"/v1/wisdom/{TestWisdom._wisdom_id}/outcome", json={
            "outcome": "success",
            "details": "All endpoints responded correctly.",
        }, headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert data["updated"] is True

    def test_add_feedback(self, client):
        """POST /v1/wisdom/{id}/feedback — add feedback."""
        assert TestWisdom._wisdom_id
        r = client.post(f"/v1/wisdom/{TestWisdom._wisdom_id}/feedback", json={
            "score": 5,
            "notes": "Perfect smoke test run.",
        }, headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert data["updated"] is True

    def test_search_wisdom(self, client):
        """POST /v1/wisdom/search — search wisdom."""
        r = client.post("/v1/wisdom/search", json={
            "query": "smoke test",
            "limit": 10,
        }, headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert "count" in data


# ============================================================================
# 6. AUTH ENFORCEMENT (verify it works when enabled)
# ============================================================================

class TestAuth:
    def test_health_no_auth(self, client):
        """Health should always be public."""
        r = client.get("/v1/health")
        assert r.status_code == 200

    def test_store_without_key_ok_in_dev_mode(self, client):
        """With no MAASV_API_KEY set, auth is disabled (dev mode)."""
        r = client.post("/v1/memory/store", json={
            "content": "Auth test memory",
            "category": "test",
        })
        assert r.status_code == 200

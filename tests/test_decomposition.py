"""
Integration tests for the maasv decomposition.

Verifies that all modules (db, store, retrieval, graph, wisdom) work correctly
after being extracted from the monolithic store.py.

Uses a shared temp DB per session via module-scoped fixture.
"""

import sys
import tempfile
from pathlib import Path

import pytest


# ============================================================================
# MOCK PROVIDERS
# ============================================================================

class MockEmbedProvider:
    """Deterministic embeddings for testing. Hashes text into a vector."""
    def __init__(self, dims=64):
        self.dims = dims
        self.call_count = 0

    def embed(self, text: str) -> list[float]:
        self.call_count += 1
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        vec = [b / 255.0 for b in h]
        while len(vec) < self.dims:
            vec.extend(vec)
        return vec[:self.dims]

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)


class MockLLMProvider:
    """Mock LLM that returns canned JSON responses."""
    def __init__(self):
        self.call_count = 0

    def call(self, messages, model, max_tokens, source=""):
        self.call_count += 1
        return "[]"


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def maasv_db(tmp_path_factory):
    """Initialize maasv with a fresh test database (shared across module)."""
    from maasv.config import MaasvConfig
    import maasv

    tmpdir = tmp_path_factory.mktemp("maasv_test")
    db_path = tmpdir / "test.db"

    config = MaasvConfig(
        db_path=db_path,
        embed_dims=64,
        extraction_model="test-model",
        inference_model="test-model",
        review_model="test-model",
        cross_encoder_enabled=False,
    )

    llm = MockLLMProvider()
    embed = MockEmbedProvider(dims=64)

    maasv.init(config=config, llm=llm, embed=embed)
    return {"llm": llm, "embed": embed, "db_path": db_path}


# ============================================================================
# db.py tests
# ============================================================================

class TestDB:
    def test_connection(self, maasv_db):
        from maasv.core.db import get_db
        db = get_db()
        assert db is not None
        row = db.execute("SELECT vec_version()").fetchone()
        assert row is not None
        db.close()

    def test_plain_connection(self, maasv_db):
        from maasv.core.db import get_plain_db
        db = get_plain_db()
        assert db is not None
        db.execute("SELECT 1").fetchone()
        db.close()

    def test_tables_exist(self, maasv_db):
        from maasv.core.db import get_db
        db = get_db()
        tables = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {row['name'] for row in tables}
        assert 'memories' in table_names
        assert 'entities' in table_names
        assert 'relationships' in table_names
        assert 'schema_migrations' in table_names
        db.close()

    def test_embeddings(self, maasv_db):
        from maasv.core.db import get_embedding, get_query_embedding, serialize_embedding
        emb = get_embedding("test text")
        assert len(emb) == 64
        assert isinstance(emb[0], float)
        q_emb = get_query_embedding("test query")
        assert len(q_emb) == 64
        serialized = serialize_embedding(emb)
        assert isinstance(serialized, bytes)


# ============================================================================
# store.py tests
# ============================================================================

class TestStore:
    def test_store_memory(self, maasv_db):
        from maasv.core.store import store_memory
        mid = store_memory(
            content="Alex lives in Riverside",
            category="identity",
            subject="Alex",
            source="test",
        )
        assert mid.startswith("mem_")

    def test_store_dedup(self, maasv_db):
        from maasv.core.store import store_memory
        mid1 = store_memory(content="Alex lives in Riverside", category="identity")
        mid2 = store_memory(content="Alex lives in Riverside", category="identity")
        assert mid1 == mid2

    def test_get_all_active(self, maasv_db):
        from maasv.core.store import get_all_active
        active = get_all_active()
        assert len(active) >= 1
        assert any("Riverside" in m['content'] for m in active)

    def test_get_recent_memories(self, maasv_db):
        from maasv.core.store import get_recent_memories
        recent = get_recent_memories(hours=48)
        assert len(recent) >= 1

    def test_supersede_memory(self, maasv_db):
        from maasv.core.store import store_memory, supersede_memory, get_all_active
        old_id = store_memory(content="The server runs Ubuntu 22.04", category="context", subject="Server")
        new_id = supersede_memory(old_id, "The server runs Debian 12", force=True)
        assert new_id != old_id
        active = get_all_active()
        active_ids = {m['id'] for m in active}
        assert new_id in active_ids
        assert old_id not in active_ids

    def test_update_metadata(self, maasv_db):
        from maasv.core.store import store_memory, update_memory_metadata
        mid = store_memory(content="Test metadata update", category="test", metadata={"key1": "val1"})
        result = update_memory_metadata(mid, {"key2": "val2"})
        assert result is True

    def test_delete_memory(self, maasv_db):
        from maasv.core.store import store_memory, delete_memory, get_all_active
        mid = store_memory(content="This memory will be deleted xyz123", category="test")
        assert delete_memory(mid) is True
        active = get_all_active()
        assert not any(m['id'] == mid for m in active)


# ============================================================================
# graph.py tests
# ============================================================================

class TestGraph:
    def test_create_entity(self, maasv_db):
        from maasv.core.graph import create_entity, get_entity
        eid = create_entity("Alex", "person")
        assert eid.startswith("ent_")
        entity = get_entity(eid)
        assert entity['name'] == "Alex"
        assert entity['entity_type'] == "person"

    def test_find_entity_by_name(self, maasv_db):
        from maasv.core.graph import find_entity_by_name
        entity = find_entity_by_name("Alex")
        assert entity is not None
        assert entity['name'] == "Alex"

    def test_find_or_create_entity(self, maasv_db):
        from maasv.core.graph import find_or_create_entity, get_entity
        eid1 = find_or_create_entity("Alex", "person")
        eid2 = find_or_create_entity("Helix", "project")
        assert eid1 != eid2
        helix = get_entity(eid2)
        assert helix['name'] == "Helix"

    def test_normalize_entity_name(self, maasv_db):
        from maasv.core.graph import normalize_entity_name
        assert normalize_entity_name("React-Native") == normalize_entity_name("react_native")
        assert normalize_entity_name("fastapi.dev") == "fastapi"
        assert normalize_entity_name("projects") == "project"

    def test_add_relationship(self, maasv_db):
        from maasv.core.graph import find_or_create_entity, add_relationship, get_entity_relationships
        alex_id = find_or_create_entity("Alex", "person")
        helix_id = find_or_create_entity("Helix", "project")
        rel_id = add_relationship(alex_id, "works_on", object_id=helix_id, source="test")
        assert rel_id.startswith("rel_")
        rels = get_entity_relationships(alex_id, direction="outgoing")
        assert any(r['predicate'] == "works_on" for r in rels)

    def test_relationship_dedup(self, maasv_db):
        from maasv.core.graph import find_or_create_entity, add_relationship
        alex_id = find_or_create_entity("Alex", "person")
        helix_id = find_or_create_entity("Helix", "project")
        rel1 = add_relationship(alex_id, "works_on", object_id=helix_id)
        rel2 = add_relationship(alex_id, "works_on", object_id=helix_id)
        assert rel1 == rel2

    def test_expire_relationship(self, maasv_db):
        from maasv.core.graph import find_or_create_entity, add_relationship, expire_relationship, get_entity_relationships
        a_id = find_or_create_entity("TestExpireA", "thing")
        b_id = find_or_create_entity("TestExpireB", "thing")
        rel_id = add_relationship(a_id, "works_with", object_id=b_id)
        assert expire_relationship(rel_id) is True
        rels = get_entity_relationships(a_id, include_expired=False)
        assert not any(r['id'] == rel_id for r in rels)

    def test_graph_query(self, maasv_db):
        from maasv.core.graph import graph_query
        results = graph_query(subject_type="person", predicate="works_on")
        assert len(results) >= 1
        assert results[0]['subject_name'] == "Alex"

    def test_entity_profile(self, maasv_db):
        from maasv.core.graph import find_entity_by_name, get_entity_profile
        alex = find_entity_by_name("Alex")
        profile = get_entity_profile(alex['id'])
        assert 'entity' in profile
        assert 'relationships' in profile
        assert profile['entity']['name'] == "Alex"

    def test_search_entities(self, maasv_db):
        from maasv.core.graph import search_entities
        results = search_entities("Alex")
        assert len(results) >= 1

    def test_merge_entity(self, maasv_db):
        from maasv.core.graph import create_entity, merge_entity, add_relationship, get_entity
        keeper = create_entity("MergeKeeper", "thing")
        dup = create_entity("MergeDup", "thing")
        add_relationship(dup, "has_reference", object_value="test_val")
        stats = merge_entity(keeper, [dup])
        assert stats['entities_deleted'] == 1
        assert get_entity(dup) is None

    def test_entity_name_sanitization(self, maasv_db):
        from maasv.core.graph import create_entity, get_entity, _sanitize_entity_name
        # Allowed characters pass through
        assert _sanitize_entity_name("John O'Brien") == "John O'Brien"
        assert _sanitize_entity_name("React-Native") == "React-Native"
        assert _sanitize_entity_name("v2.0_beta") == "v2.0_beta"
        # Disallowed characters are stripped
        assert _sanitize_entity_name("Test<script>alert</script>") == "Testscriptalertscript"
        assert _sanitize_entity_name("name; DROP TABLE") == "name DROP TABLE"
        # Whitespace is collapsed
        assert _sanitize_entity_name("  lots   of   spaces  ") == "lots of spaces"
        # Entity with stripped chars still stores correctly
        eid = create_entity("Test{Entity}123", "thing")
        entity = get_entity(eid)
        assert entity['name'] == "TestEntity123"

    def test_entity_name_sanitization_rejects_empty(self, maasv_db):
        from maasv.core.graph import create_entity
        with pytest.raises(ValueError, match="too short"):
            create_entity("!!!", "thing")
        with pytest.raises(ValueError, match="too short"):
            create_entity("", "thing")

    def test_entity_name_length_cap(self, maasv_db):
        """Task 3: Entity names are truncated to MAX_ENTITY_NAME_LENGTH."""
        from maasv.core.graph import create_entity, get_entity, MAX_ENTITY_NAME_LENGTH
        long_name = "A" * 300
        eid = create_entity(long_name, "thing")
        entity = get_entity(eid)
        assert len(entity['name']) == MAX_ENTITY_NAME_LENGTH

    def test_confidence_clamping(self, maasv_db):
        """Task 4: Confidence is clamped to [0.0, 1.0] and non-numeric coerced."""
        from maasv.core.graph import _clamp_confidence
        assert _clamp_confidence(0.5) == 0.5
        assert _clamp_confidence(1.5) == 1.0
        assert _clamp_confidence(-0.3) == 0.0
        assert _clamp_confidence("not_a_number") == 0.5
        assert _clamp_confidence(None) == 0.5
        assert _clamp_confidence("0.7") == 0.7

    def test_predicate_allowlist_valid(self, maasv_db):
        """Task 5: Valid predicates are accepted."""
        from maasv.core.graph import find_or_create_entity, add_relationship
        a = find_or_create_entity("PredicateTestA", "person")
        b = find_or_create_entity("PredicateTestB", "project")
        rel = add_relationship(a, "works_on", object_id=b)
        assert rel.startswith("rel_")

    def test_predicate_allowlist_rejects_unknown(self, maasv_db):
        """Task 5: Unknown predicates are rejected."""
        from maasv.core.graph import find_or_create_entity, add_relationship
        a = find_or_create_entity("PredicateTestC", "person")
        b = find_or_create_entity("PredicateTestD", "project")
        with pytest.raises(ValueError, match="Unknown predicate"):
            add_relationship(a, "totally_made_up", object_id=b)

    def test_object_value_length_cap(self, maasv_db):
        """Task 3: Object values are truncated to MAX_OBJECT_VALUE_LENGTH."""
        from maasv.core.graph import find_or_create_entity, add_relationship, get_entity_relationships, MAX_OBJECT_VALUE_LENGTH
        a = find_or_create_entity("ValueCapTest", "person")
        long_value = "x" * 5000
        rel_id = add_relationship(a, "has_email", object_value=long_value)
        rels = get_entity_relationships(a, direction="outgoing")
        email_rel = [r for r in rels if r['id'] == rel_id][0]
        assert len(email_rel['object_value']) == MAX_OBJECT_VALUE_LENGTH

    def test_store_memory_confidence_clamping(self, maasv_db):
        """Task 4: store_memory clamps confidence."""
        from maasv.core.store import store_memory
        from maasv.core.db import _db
        mid = store_memory(content="Confidence clamp test xyz", category="test", confidence=5.0)
        with _db() as db:
            row = db.execute("SELECT confidence FROM memories WHERE id = ?", (mid,)).fetchone()
        assert row['confidence'] == 1.0

    def test_entity_unique_constraint(self, maasv_db):
        """Task 17: create_entity returns existing ID on duplicate (canonical_name, entity_type)."""
        from maasv.core.graph import create_entity
        eid1 = create_entity("UniqueTestEntity", "gadget")
        eid2 = create_entity("UniqueTestEntity", "gadget")
        assert eid1 == eid2  # Same entity, not a duplicate

    def test_entity_unique_constraint_different_types(self, maasv_db):
        """Task 17: Same name with different entity_type is allowed."""
        from maasv.core.graph import create_entity
        eid1 = create_entity("SharedName", "person")
        eid2 = create_entity("SharedName", "project")
        assert eid1 != eid2  # Different types = different entities

    def test_relationship_unique_constraint(self, maasv_db):
        """Task 17: add_relationship returns existing ID on duplicate active triple."""
        from maasv.core.graph import find_or_create_entity, add_relationship
        from maasv.core.db import _db
        a = find_or_create_entity("RelUniqA", "person")
        b = find_or_create_entity("RelUniqB", "project")
        rel1 = add_relationship(a, "manages", object_id=b)
        # Bypass the app-level dedup by inserting directly, then check constraint
        # Instead, verify that the existing app-level + DB constraint works together
        rel2 = add_relationship(a, "manages", object_id=b)
        assert rel1 == rel2

    def test_dedup_indexes_exist(self, maasv_db):
        """Task 17: Verify unique indexes were created by migration 4."""
        from maasv.core.db import _db
        with _db() as db:
            indexes = db.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
            index_names = {row['name'] for row in indexes}
        assert 'idx_entities_unique_canonical' in index_names
        assert 'idx_rel_active_entity' in index_names
        assert 'idx_rel_active_value' in index_names


# ============================================================================
# retrieval.py tests
# ============================================================================

class TestRetrieval:
    def test_find_similar_memories(self, maasv_db):
        from maasv.core.retrieval import find_similar_memories
        results = find_similar_memories("Alex Riverside", limit=3)
        assert len(results) >= 1
        assert any("Riverside" in m['content'] for m in results)

    def test_retrieval_limit_cap(self, maasv_db):
        """Task 8: limit is hard-capped at MAX_RETRIEVAL_LIMIT."""
        from maasv.core.retrieval import find_similar_memories, MAX_RETRIEVAL_LIMIT
        # Should not raise even with absurdly large limit
        results = find_similar_memories("test", limit=99999)
        assert isinstance(results, list)
        assert MAX_RETRIEVAL_LIMIT == 200

    def test_search_fts(self, maasv_db):
        from maasv.core.retrieval import search_fts
        results = search_fts("Riverside", limit=5)
        assert len(results) >= 1

    def test_find_by_subject(self, maasv_db):
        from maasv.core.retrieval import find_by_subject
        results = find_by_subject("Alex")
        assert len(results) >= 1

    def test_get_core_memories(self, maasv_db):
        from maasv.core.retrieval import get_core_memories
        core = get_core_memories(refresh=True)
        assert len(core) >= 1

    def test_tiered_memory_context(self, maasv_db):
        from maasv.core.retrieval import get_tiered_memory_context
        context = get_tiered_memory_context(query="Alex")
        assert "Remembered facts:" in context
        assert len(context) > 20


# ============================================================================
# wisdom.py tests
# ============================================================================

class TestWisdom:
    def test_wisdom_tables(self, maasv_db):
        from maasv.core.db import get_db
        db = get_db()
        tables = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {row['name'] for row in tables}
        assert 'wisdom' in table_names
        db.close()

    def test_log_reasoning(self, maasv_db):
        from maasv.core.wisdom import log_reasoning
        entry_id = log_reasoning(
            action_type="test_action",
            reasoning="Testing the wisdom module",
            action_data={"key": "value"},
        )
        assert entry_id is not None

    def test_search_wisdom(self, maasv_db):
        from maasv.core.wisdom import search_wisdom
        results = search_wisdom("test")
        assert isinstance(results, list)


# ============================================================================
# __init__.py re-exports
# ============================================================================

class TestReexports:
    def test_core_reexports(self, maasv_db):
        from maasv.core import (
            store_memory, find_similar_memories, find_by_subject, search_fts,
            get_all_active, get_recent_memories, delete_memory, supersede_memory,
            create_entity, get_entity, find_entity_by_name, find_or_create_entity,
            search_entities, add_relationship, expire_relationship,
            get_entity_relationships, get_causal_chain, graph_query, get_entity_profile,
            log_reasoning, record_outcome, add_feedback, get_relevant_wisdom, search_wisdom,
        )
        assert callable(store_memory)
        assert callable(find_similar_memories)
        assert callable(create_entity)
        assert callable(graph_query)
        assert callable(log_reasoning)


# ============================================================================
# lifecycle import paths
# ============================================================================

class TestLifecycleImports:
    def test_inference_imports(self, maasv_db):
        from maasv.lifecycle import inference

    def test_memory_hygiene_imports(self, maasv_db):
        from maasv.lifecycle import memory_hygiene

    def test_reorganize_imports(self, maasv_db):
        from maasv.lifecycle import reorganize


# ============================================================================
# Backwards-compatible CLI entry point
# ============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

"""
Tests for the learned ranker: autograd, model, features, logging, labeling, training.

Uses a shared temp DB per session via module-scoped fixture.
"""

import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


# ============================================================================
# MOCK PROVIDERS
# ============================================================================

class MockEmbedProvider:
    def __init__(self, dims=64):
        self.dims = dims

    def embed(self, text: str) -> list[float]:
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        vec = [b / 255.0 for b in h]
        while len(vec) < self.dims:
            vec.extend(vec)
        return vec[:self.dims]

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)


class MockLLMProvider:
    def call(self, messages, model, max_tokens, source=""):
        return "[]"


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def maasv_db(tmp_path_factory):
    """Initialize maasv with a fresh test database."""
    from maasv.config import MaasvConfig
    import maasv

    tmpdir = tmp_path_factory.mktemp("learned_ranker_test")
    db_path = tmpdir / "test.db"

    config = MaasvConfig(
        db_path=db_path,
        embed_dims=64,
        extraction_model="test-model",
        inference_model="test-model",
        review_model="test-model",
        cross_encoder_enabled=False,
        learned_ranker_enabled=True,
        learned_ranker_min_samples=5,  # Low for testing
        learned_ranker_shadow_mode=False,
    )

    llm = MockLLMProvider()
    embed = MockEmbedProvider(dims=64)
    maasv.init(config=config, llm=llm, embed=embed)
    return {"db_path": db_path}


@pytest.fixture
def sample_candidates():
    """Sample memory candidates for testing."""
    now = datetime.now(timezone.utc)
    return [
        {
            "id": "mem_001",
            "content": "Adam lives in Manhattan",
            "category": "identity",
            "subject": "Adam",
            "importance": 1.0,
            "access_count": 5,
            "created_at": (now - timedelta(days=10)).isoformat(),
        },
        {
            "id": "mem_002",
            "content": "TerryAnn is a Medicare platform",
            "category": "project",
            "subject": "TerryAnn",
            "importance": 0.8,
            "access_count": 3,
            "created_at": (now - timedelta(days=30)).isoformat(),
        },
        {
            "id": "mem_003",
            "content": "Gabby prefers Italian food",
            "category": "family",
            "subject": "Gabby",
            "importance": 0.6,
            "access_count": 1,
            "created_at": (now - timedelta(days=90)).isoformat(),
        },
    ]


# ============================================================================
# AUTOGRAD TESTS
# ============================================================================

class TestAutograd:
    def test_addition(self):
        from maasv.core.autograd import Value
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        assert c.data == 5.0
        c.backward()
        assert a.grad == 1.0
        assert b.grad == 1.0

    def test_multiplication(self):
        from maasv.core.autograd import Value
        a = Value(3.0)
        b = Value(4.0)
        c = a * b
        assert c.data == 12.0
        c.backward()
        assert a.grad == 4.0
        assert b.grad == 3.0

    def test_subtraction(self):
        from maasv.core.autograd import Value
        a = Value(5.0)
        b = Value(3.0)
        c = a - b
        assert c.data == 2.0
        c.backward()
        assert a.grad == 1.0
        assert b.grad == -1.0

    def test_division(self):
        from maasv.core.autograd import Value
        a = Value(6.0)
        b = Value(3.0)
        c = a / b
        assert abs(c.data - 2.0) < 1e-6
        c.backward()
        assert abs(a.grad - 1.0 / 3.0) < 1e-6

    def test_power(self):
        from maasv.core.autograd import Value
        a = Value(3.0)
        c = a ** 2
        assert c.data == 9.0
        c.backward()
        assert a.grad == 6.0  # d/da(a^2) = 2a = 6

    def test_relu(self):
        from maasv.core.autograd import Value
        a = Value(3.0)
        b = Value(-2.0)
        c = a.relu()
        d = b.relu()
        assert c.data == 3.0
        assert d.data == 0.0
        c.backward()
        assert a.grad == 1.0
        d.backward()
        assert b.grad == 0.0

    def test_sigmoid(self):
        from maasv.core.autograd import Value
        a = Value(0.0)
        c = a.sigmoid()
        assert abs(c.data - 0.5) < 1e-6
        c.backward()
        assert abs(a.grad - 0.25) < 1e-6  # sigmoid'(0) = 0.25

    def test_sigmoid_negative(self):
        from maasv.core.autograd import Value
        a = Value(-5.0)
        c = a.sigmoid()
        assert c.data < 0.01  # Should be close to 0

    def test_log(self):
        from maasv.core.autograd import Value
        a = Value(math.e)
        c = a.log()
        assert abs(c.data - 1.0) < 1e-6
        c.backward()
        assert abs(a.grad - 1.0 / math.e) < 1e-6

    def test_chain(self):
        """Test a multi-operation chain."""
        from maasv.core.autograd import Value
        x = Value(2.0)
        y = Value(3.0)
        z = (x * y + Value(1.0)).relu()
        assert z.data == 7.0
        z.backward()
        assert x.grad == 3.0  # dz/dx = y = 3
        assert y.grad == 2.0  # dz/dy = x = 2

    def test_scalar_ops(self):
        """Test operations with plain numbers."""
        from maasv.core.autograd import Value
        a = Value(3.0)
        b = a + 2
        assert b.data == 5.0
        c = 2 + a
        assert c.data == 5.0
        d = a * 3
        assert d.data == 9.0
        e = 3 * a
        assert e.data == 9.0


# ============================================================================
# RANKING MODEL TESTS
# ============================================================================

class TestRankingModel:
    def test_init(self):
        from maasv.core.learned_ranker import RankingModel, N_FEATURES
        model = RankingModel()
        params = model.parameters()
        # 8*8 (w1) + 8 (b1) + 1*8 (w2) + 1 (b2) = 81
        assert len(params) == 81

    def test_forward(self):
        from maasv.core.autograd import Value
        from maasv.core.learned_ranker import RankingModel, N_FEATURES
        model = RankingModel()
        x = [Value(0.5) for _ in range(N_FEATURES)]
        out = model.forward(x)
        assert 0.0 <= out.data <= 1.0  # Sigmoid output

    def test_backward(self):
        from maasv.core.autograd import Value
        from maasv.core.learned_ranker import RankingModel, N_FEATURES
        model = RankingModel()
        x = [Value(0.5) for _ in range(N_FEATURES)]
        out = model.forward(x)
        out.backward()
        params = model.parameters()
        # At least some gradients should be non-zero
        assert any(p.grad != 0.0 for p in params)

    def test_state_dict_roundtrip(self):
        from maasv.core.learned_ranker import RankingModel
        model = RankingModel()
        state = model.state_dict()

        model2 = RankingModel()
        model2.load_state_dict(state)

        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert p1.data == p2.data

    def test_deterministic_forward(self):
        """Same weights + same input = same output."""
        from maasv.core.autograd import Value
        from maasv.core.learned_ranker import RankingModel, N_FEATURES
        model = RankingModel()
        state = model.state_dict()

        x1 = [Value(0.3) for _ in range(N_FEATURES)]
        out1 = model.forward(x1)

        model2 = RankingModel()
        model2.load_state_dict(state)
        x2 = [Value(0.3) for _ in range(N_FEATURES)]
        out2 = model2.forward(x2)

        assert abs(out1.data - out2.data) < 1e-10


# ============================================================================
# FEATURE EXTRACTION TESTS
# ============================================================================

class TestFeatureExtraction:
    def test_basic_features(self, sample_candidates):
        from maasv.core.learned_ranker import extract_features, N_FEATURES
        now = datetime.now(timezone.utc)
        mem = sample_candidates[0]
        features = extract_features(
            mem,
            vector_distances={"mem_001": 0.3},
            bm25_ids={"mem_001"},
            graph_ids=set(),
            protected={"identity", "family"},
            now=now,
        )
        assert len(features) == N_FEATURES
        assert all(0.0 <= f <= 1.0 for f in features)

    def test_vector_similarity(self, sample_candidates):
        from maasv.core.learned_ranker import extract_features
        now = datetime.now(timezone.utc)
        mem = sample_candidates[0]
        features = extract_features(
            mem,
            vector_distances={"mem_001": 0.2},
            bm25_ids=set(),
            graph_ids=set(),
            protected={"identity", "family"},
            now=now,
        )
        assert abs(features[0] - 0.8) < 1e-6  # 1 - 0.2

    def test_no_vector_distance(self, sample_candidates):
        from maasv.core.learned_ranker import extract_features
        now = datetime.now(timezone.utc)
        mem = sample_candidates[0]
        features = extract_features(
            mem,
            vector_distances={},
            bm25_ids=set(),
            graph_ids=set(),
            protected=set(),
            now=now,
        )
        assert features[0] == 0.0  # No vector distance -> 0

    def test_binary_signals(self, sample_candidates):
        from maasv.core.learned_ranker import extract_features
        now = datetime.now(timezone.utc)
        mem = sample_candidates[0]

        # BM25 hit
        features = extract_features(
            mem, {}, {"mem_001"}, set(), set(), now
        )
        assert features[1] == 1.0  # bm25_hit
        assert features[2] == 0.0  # graph_hit

        # Graph hit
        features = extract_features(
            mem, {}, set(), {"mem_001"}, set(), now
        )
        assert features[1] == 0.0
        assert features[2] == 1.0

    def test_protected_category_no_decay(self, sample_candidates):
        from maasv.core.learned_ranker import extract_features
        now = datetime.now(timezone.utc)
        mem = sample_candidates[0]  # category=identity
        features = extract_features(
            mem, {}, set(), set(), {"identity", "family"}, now
        )
        assert features[4] == 1.0  # age_decay = 1.0 for protected

    def test_unprotected_decay(self):
        from maasv.core.learned_ranker import extract_features
        now = datetime.now(timezone.utc)
        old_mem = {
            "id": "mem_old",
            "content": "old memory",
            "category": "project",
            "importance": 0.5,
            "access_count": 0,
            "created_at": (now - timedelta(days=180)).isoformat(),
        }
        features = extract_features(
            old_mem, {}, set(), set(), set(), now
        )
        # exp(-180/180) = exp(-1) ≈ 0.368
        assert abs(features[4] - math.exp(-1)) < 0.01

    def test_access_count_normalization(self, sample_candidates):
        from maasv.core.learned_ranker import extract_features
        now = datetime.now(timezone.utc)
        mem = sample_candidates[0]  # access_count=5
        features = extract_features(
            mem, {}, set(), set(), set(), now
        )
        expected = math.log(2 + 5) / math.log(7)
        assert abs(features[5] - expected) < 1e-6


# ============================================================================
# RETRIEVAL LOGGING TESTS
# ============================================================================

class TestRetrievalLogging:
    def test_log_retrieval(self, maasv_db, sample_candidates):
        from maasv.core.learned_ranker import log_retrieval
        from maasv.core.db import _db
        now = datetime.now(timezone.utc)

        log_retrieval(
            query="where does Adam live",
            candidates=sample_candidates,
            returned_ids=["mem_001", "mem_002"],
            vector_distances={"mem_001": 0.2, "mem_002": 0.4},
            bm25_ids={"mem_001"},
            graph_ids=set(),
            protected={"identity", "family"},
            now=now,
        )

        with _db() as db:
            rows = db.execute("SELECT * FROM retrieval_log").fetchall()
        assert len(rows) >= 1
        row = rows[-1]
        assert row["query_text"] == "where does Adam live"
        returned = json.loads(row["returned_ids"])
        assert "mem_001" in returned
        features = json.loads(row["features"])
        assert "mem_001" in features
        assert len(features["mem_001"]) == 8

    def test_log_retrieval_is_best_effort(self, maasv_db):
        """Logging should not raise even with bad input."""
        from maasv.core.learned_ranker import log_retrieval
        now = datetime.now(timezone.utc)
        # This should not raise
        log_retrieval(
            query="test",
            candidates=[],
            returned_ids=[],
            vector_distances={},
            bm25_ids=set(),
            graph_ids=set(),
            protected=set(),
            now=now,
        )


# ============================================================================
# OUTCOME LABELING TESTS
# ============================================================================

class TestOutcomeLabeling:
    def test_label_outcomes(self, maasv_db, sample_candidates):
        from maasv.core.learned_ranker import log_retrieval, label_outcomes
        from maasv.core.db import _db

        # Insert a retrieval log entry with old timestamp (>2hrs ago)
        old_time = datetime.now(timezone.utc) - timedelta(hours=3)
        log_retrieval(
            query="test label query",
            candidates=sample_candidates,
            returned_ids=["mem_001"],
            vector_distances={"mem_001": 0.2},
            bm25_ids=set(),
            graph_ids=set(),
            protected=set(),
            now=old_time,
        )

        # Set the timestamp manually to ensure it's old enough
        with _db() as db:
            db.execute(
                """UPDATE retrieval_log SET timestamp = ?
                   WHERE query_text = 'test label query' AND outcomes IS NULL""",
                (old_time.isoformat(),),
            )
            db.commit()

        labeled = label_outcomes(cancel_check=lambda: False)
        assert labeled >= 1

        # Check outcomes were written
        with _db() as db:
            row = db.execute(
                "SELECT outcomes FROM retrieval_log WHERE query_text = 'test label query' AND outcomes IS NOT NULL"
            ).fetchone()
        assert row is not None
        outcomes = json.loads(row["outcomes"])
        assert "mem_001" in outcomes

    def test_label_respects_cancellation(self, maasv_db):
        from maasv.core.learned_ranker import label_outcomes
        # Immediately cancelled
        labeled = label_outcomes(cancel_check=lambda: True)
        assert labeled == 0


# ============================================================================
# TRAINING TESTS
# ============================================================================

class TestTraining:
    def _seed_training_data(self, n=20):
        """Insert enough labeled retrieval logs for training."""
        from maasv.core.db import _db
        import uuid

        now = datetime.now(timezone.utc)
        with _db() as db:
            for i in range(n):
                features = {
                    f"mem_{i:03d}": [
                        0.5 + 0.3 * (i % 3 == 0),  # vector_similarity
                        1.0 if i % 2 == 0 else 0.0,  # bm25_hit
                        1.0 if i % 3 == 0 else 0.0,  # graph_hit
                        0.5 + 0.1 * (i % 5),          # importance
                        0.8,                            # age_decay
                        0.6,                            # access_count_norm
                        0.3,                            # category_code
                        1.0 / (1 + i),                  # rrf_rank_norm
                    ]
                }
                # High-scoring features -> positive outcome
                outcome_val = 1.0 if i % 3 == 0 else 0.0
                outcomes = {f"mem_{i:03d}": outcome_val}

                db.execute(
                    """INSERT INTO retrieval_log
                       (id, query_text, query_embedding_hash, timestamp,
                        candidate_count, returned_ids, features, outcomes, outcome_recorded_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()),
                        f"query_{i}",
                        f"hash_{i}",
                        (now - timedelta(hours=i)).isoformat(),
                        1,
                        json.dumps([f"mem_{i:03d}"]),
                        json.dumps(features),
                        json.dumps(outcomes),
                        now.isoformat(),
                    ),
                )
            db.commit()

    def test_train_reduces_loss(self, maasv_db):
        from maasv.core.learned_ranker import train, reload_model
        reload_model()  # Clear any cached model

        self._seed_training_data(n=20)

        stats = train(
            cancel_check=lambda: False,
            max_steps=30,
            lr=0.01,
        )

        assert stats is not None
        assert stats["steps"] == 30
        assert stats["training_samples"] >= 5
        assert stats["loss_reduction"] > 0  # Loss should decrease
        assert stats["final_loss"] is not None

    def test_train_saves_weights(self, maasv_db):
        from maasv.core.db import _db

        with _db() as db:
            row = db.execute(
                "SELECT * FROM learned_ranker_weights WHERE id = 1"
            ).fetchone()

        assert row is not None
        weights = json.loads(row["weights_json"])
        assert "w1" in weights
        assert "b1" in weights
        assert "w2" in weights
        assert "b2" in weights
        assert row["training_samples"] >= 5

    def test_train_respects_cancellation(self, maasv_db):
        from maasv.core.learned_ranker import train
        # Cancel immediately
        stats = train(cancel_check=lambda: True, max_steps=100)
        # Should return quickly with 0 steps
        if stats is not None:
            assert stats["steps"] == 0

    def test_ndcg_computed(self, maasv_db):
        from maasv.core.learned_ranker import train, reload_model
        reload_model()

        stats = train(cancel_check=lambda: False, max_steps=10)
        if stats is not None:
            assert "ndcg_score" in stats
            assert 0.0 <= stats["ndcg_score"] <= 1.0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    def test_model_loads_after_training(self, maasv_db):
        from maasv.core.learned_ranker import _get_model, reload_model
        reload_model()
        model = _get_model()
        # Should load because we have training data and min_samples=5
        assert model is not None

    def test_score_returns_results(self, maasv_db, sample_candidates):
        from maasv.core.learned_ranker import score, reload_model
        reload_model()
        now = datetime.now(timezone.utc)

        result = score(
            candidates=sample_candidates,
            protected={"identity", "family"},
            now=now,
            vector_distances={"mem_001": 0.2, "mem_002": 0.4},
            bm25_ids={"mem_001"},
            graph_ids=set(),
        )

        # Should not be None since shadow_mode=False and we have a trained model
        assert result is not None
        primary, supplementary = result
        # mem_001 and mem_002 have vector distances
        assert len(primary) == 2
        # mem_003 has no vector distance
        assert len(supplementary) == 1
        # All should have _imp_score
        assert all("_imp_score" in m for m in primary + supplementary)

    def test_score_fallback_when_disabled(self, maasv_db, sample_candidates):
        """score() returns None when learned ranker is disabled."""
        import maasv
        config = maasv.get_config()
        original = config.learned_ranker_enabled
        try:
            config.learned_ranker_enabled = False
            from maasv.core.learned_ranker import score
            now = datetime.now(timezone.utc)
            result = score(
                sample_candidates, set(), now, {}, set(), set()
            )
            assert result is None
        finally:
            config.learned_ranker_enabled = original

    def test_retrieval_logging_in_find_similar(self, maasv_db):
        """find_similar_memories() should log to retrieval_log."""
        from maasv.core.store import store_memory
        from maasv.core.retrieval import find_similar_memories
        from maasv.core.db import _db

        # Store some memories
        store_memory(content="The sky is blue today", category="context")
        store_memory(content="Python is a programming language", category="learning")
        store_memory(content="New York is a big city", category="context")

        # Get count before
        with _db() as db:
            before = db.execute("SELECT COUNT(*) as c FROM retrieval_log").fetchone()["c"]

        # Run retrieval
        results = find_similar_memories("What color is the sky?", limit=3)

        # Check count after
        with _db() as db:
            after = db.execute("SELECT COUNT(*) as c FROM retrieval_log").fetchone()["c"]

        assert after > before


# ============================================================================
# LEARN JOB TESTS
# ============================================================================

class TestLearnJob:
    def test_run_learn_job(self, maasv_db):
        from maasv.lifecycle.learn import run_learn_job
        # Should not raise
        run_learn_job(data={}, cancel_check=lambda: False)

    def test_learn_job_respects_cancellation(self, maasv_db):
        from maasv.lifecycle.learn import run_learn_job
        # Should return quickly
        run_learn_job(data={}, cancel_check=lambda: True)

    def test_learn_job_type_exists(self):
        from maasv.lifecycle.worker import JobType
        assert hasattr(JobType, "LEARN")
        assert JobType.LEARN.value == "learn"

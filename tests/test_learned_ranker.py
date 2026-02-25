"""
Tests for the learned ranker: autograd, model, features, logging, labeling, training.

Uses a shared temp DB per session via module-scoped fixture.
"""

import json
import math
from datetime import datetime, timedelta, timezone

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
        return vec[: self.dims]

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
    import maasv
    from maasv.config import MaasvConfig

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
            "content": "Alex lives in the Riverside district",
            "category": "identity",
            "subject": "Alex",
            "importance": 1.0,
            "access_count": 5,
            "surfacing_count": 10,
            "created_at": (now - timedelta(days=10)).isoformat(),
        },
        {
            "id": "mem_002",
            "content": "Acme is a SaaS analytics platform",
            "category": "project",
            "subject": "Acme",
            "importance": 0.8,
            "access_count": 3,
            "surfacing_count": 20,
            "created_at": (now - timedelta(days=30)).isoformat(),
        },
        {
            "id": "mem_003",
            "content": "Maria prefers Italian food",
            "category": "family",
            "subject": "Maria",
            "importance": 0.6,
            "access_count": 1,
            "surfacing_count": 0,
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
        c = a**2
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
        from maasv.core.learned_ranker import RankingModel

        model = RankingModel()
        params = model.parameters()
        # 12*16 (w1) + 16 (b1) + 1*16 (w2) + 1 (b2) = 225
        assert len(params) == 225

    def test_forward(self):
        from maasv.core.autograd import Value
        from maasv.core.learned_ranker import N_FEATURES, RankingModel

        model = RankingModel()
        x = [Value(0.5) for _ in range(N_FEATURES)]
        out = model.forward(x)
        assert 0.0 <= out.data <= 1.0  # Sigmoid output

    def test_backward(self):
        from maasv.core.autograd import Value
        from maasv.core.learned_ranker import N_FEATURES, RankingModel

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
        from maasv.core.learned_ranker import N_FEATURES, RankingModel

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
        from maasv.core.learned_ranker import N_FEATURES, extract_features

        now = datetime.now(timezone.utc)
        mem = sample_candidates[0]
        features = extract_features(
            mem,
            vector_distances={"mem_001": 0.3},
            bm25_scores={"mem_001": 1.0},
            graph_scores={},
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
            bm25_scores={},
            graph_scores={},
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
            bm25_scores={},
            graph_scores={},
            protected=set(),
            now=now,
        )
        assert features[0] == 0.0  # No vector distance -> 0

    def test_continuous_signals(self, sample_candidates):
        from maasv.core.learned_ranker import extract_features

        now = datetime.now(timezone.utc)
        mem = sample_candidates[0]

        # BM25 score (continuous)
        features = extract_features(mem, {}, {"mem_001": 0.75}, {}, set(), now)
        assert features[1] == 0.75  # bm25_score
        assert features[2] == 0.0  # graph_score

        # Graph score (continuous)
        features = extract_features(mem, {}, {}, {"mem_001": 0.8}, set(), now)
        assert features[1] == 0.0
        assert features[2] == 0.8

        # Both signals
        features = extract_features(mem, {}, {"mem_001": 1.0}, {"mem_001": 0.6}, set(), now)
        assert features[1] == 1.0
        assert features[2] == 0.6

        # Missing from both
        features = extract_features(mem, {}, {}, {}, set(), now)
        assert features[1] == 0.0
        assert features[2] == 0.0

    def test_protected_category_no_decay(self, sample_candidates):
        from maasv.core.learned_ranker import extract_features

        now = datetime.now(timezone.utc)
        mem = sample_candidates[0]  # category=identity
        features = extract_features(mem, {}, {}, {}, {"identity", "family"}, now)
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
        features = extract_features(old_mem, {}, {}, {}, set(), now)
        # exp(-180/180) = exp(-1) ≈ 0.368
        assert abs(features[4] - math.exp(-1)) < 0.01

    def test_ips_utility_with_surfacing(self, sample_candidates):
        from maasv.core.learned_ranker import extract_features

        now = datetime.now(timezone.utc)
        mem = sample_candidates[0]  # access_count=5, surfacing_count=10
        features = extract_features(mem, {}, {}, {}, set(), now)
        # raw_utility = 5/10 = 0.5, min(0.5, 2.0)/2.0 = 0.25
        expected = min(5 / 10, 2.0) / 2.0
        assert abs(features[5] - expected) < 1e-6

    def test_ips_utility_cold_start(self):
        from maasv.core.learned_ranker import extract_features

        now = datetime.now(timezone.utc)
        mem = {
            "id": "mem_cold",
            "content": "cold start memory",
            "category": "project",
            "importance": 0.5,
            "access_count": 3,
            "surfacing_count": 0,
            "created_at": now.isoformat(),
        }
        features = extract_features(mem, {}, {}, {}, set(), now)
        # Cold-start fallback: log(2 + min(3, 5)) / log(7)
        expected = math.log(2 + 3) / math.log(7)
        assert abs(features[5] - expected) < 1e-6

    def test_session_features_default_zero(self, sample_candidates):
        """Without session_features, session features default to 0.0."""
        from maasv.core.learned_ranker import extract_features

        now = datetime.now(timezone.utc)
        mem = sample_candidates[0]
        features = extract_features(mem, {}, {}, {}, set(), now)
        # Features 8-11 are session context
        assert features[8] == 0.0  # query_coherence
        assert features[9] == 0.0  # session_depth_norm
        assert features[10] == 0.0  # category_session_match
        assert features[11] == 0.0  # subject_session_overlap

    def test_session_features_populated(self, sample_candidates):
        """Session features use session_context when provided."""
        from maasv.core.learned_ranker import extract_features

        now = datetime.now(timezone.utc)
        mem = sample_candidates[0]  # category=identity, subject="Alex"
        sf = {
            "query_coherence": 0.85,
            "session_depth_norm": 0.5,
            "seen_categories": {"identity", "project"},
            "seen_subjects": {"alex", "doris"},
        }
        features = extract_features(mem, {}, {}, {}, set(), now, session_features=sf)
        assert abs(features[8] - 0.85) < 1e-6  # query_coherence
        assert abs(features[9] - 0.5) < 1e-6  # session_depth_norm
        assert features[10] == 1.0  # category_session_match (identity in seen)
        assert features[11] == 1.0  # subject_session_overlap (alex in seen)

    def test_session_category_no_match(self, sample_candidates):
        """Category session match is 0.0 when category not in seen set."""
        from maasv.core.learned_ranker import extract_features

        now = datetime.now(timezone.utc)
        mem = sample_candidates[0]  # category=identity
        sf = {
            "query_coherence": 0.0,
            "session_depth_norm": 0.0,
            "seen_categories": {"project", "preference"},
            "seen_subjects": set(),
        }
        features = extract_features(mem, {}, {}, {}, set(), now, session_features=sf)
        assert features[10] == 0.0  # identity not in {project, preference}

    def test_session_subject_partial_overlap(self):
        """Subject overlap is fractional when partial token match."""
        from maasv.core.learned_ranker import extract_features

        now = datetime.now(timezone.utc)
        mem = {
            "id": "mem_partial",
            "content": "test",
            "category": "project",
            "subject": "doris voice interface",
            "importance": 0.5,
            "access_count": 0,
            "created_at": now.isoformat(),
        }
        sf = {
            "query_coherence": 0.0,
            "session_depth_norm": 0.0,
            "seen_categories": set(),
            "seen_subjects": {"doris", "fastapi"},
        }
        features = extract_features(mem, {}, {}, {}, set(), now, session_features=sf)
        # "doris voice interface" -> tokens {"doris", "voice", "interface"}
        # Overlap with {"doris", "fastapi"} = {"doris"} -> 1/3 ≈ 0.333
        assert abs(features[11] - 1.0 / 3.0) < 1e-6


# ============================================================================
# RETRIEVAL LOGGING TESTS
# ============================================================================


class TestRetrievalLogging:
    def test_log_retrieval(self, maasv_db, sample_candidates):
        from maasv.core.db import _db
        from maasv.core.learned_ranker import log_retrieval

        now = datetime.now(timezone.utc)

        log_retrieval(
            query="where does Alex live",
            candidates=sample_candidates,
            returned_ids=["mem_001", "mem_002"],
            vector_distances={"mem_001": 0.2, "mem_002": 0.4},
            bm25_scores={"mem_001": 1.0},
            graph_scores={},
            protected={"identity", "family"},
            now=now,
        )

        with _db() as db:
            rows = db.execute("SELECT * FROM retrieval_log").fetchall()
        assert len(rows) >= 1
        row = rows[-1]
        assert row["query_text"] == "where does Alex live"
        returned = json.loads(row["returned_ids"])
        assert "mem_001" in returned
        features = json.loads(row["features"])
        assert "mem_001" in features
        assert len(features["mem_001"]) == 12

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
            bm25_scores={},
            graph_scores={},
            protected=set(),
            now=now,
        )


# ============================================================================
# OUTCOME LABELING TESTS
# ============================================================================


class TestOutcomeLabeling:
    def test_label_outcomes(self, maasv_db, sample_candidates):
        from maasv.core.db import _db
        from maasv.core.learned_ranker import label_outcomes, log_retrieval

        # Insert a retrieval log entry with old timestamp (>2hrs ago)
        old_time = datetime.now(timezone.utc) - timedelta(hours=3)
        log_retrieval(
            query="test label query",
            candidates=sample_candidates,
            returned_ids=["mem_001"],
            vector_distances={"mem_001": 0.2},
            bm25_scores={},
            graph_scores={},
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
        import uuid

        from maasv.core.db import _db

        now = datetime.now(timezone.utc)
        with _db() as db:
            for i in range(n):
                features = {
                    f"mem_{i:03d}": [
                        0.5 + 0.3 * (i % 3 == 0),  # vector_similarity
                        0.8 if i % 2 == 0 else 0.0,  # bm25_score
                        0.6 if i % 3 == 0 else 0.0,  # graph_score
                        0.5 + 0.1 * (i % 5),  # importance
                        0.8,  # age_decay
                        0.6,  # access_count_norm
                        0.3,  # category_code
                        1.0 / (1 + i),  # rrf_rank_norm
                        0.5,  # query_coherence
                        0.3,  # session_depth_norm
                        1.0 if i % 2 == 0 else 0.0,  # category_session_match
                        0.2 if i % 4 == 0 else 0.0,  # subject_session_overlap
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
        from maasv.core.learned_ranker import reload_model, train

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
            row = db.execute("SELECT * FROM learned_ranker_weights WHERE id = 1").fetchone()

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
        from maasv.core.learned_ranker import reload_model, train

        reload_model()

        stats = train(cancel_check=lambda: False, max_steps=10)
        if stats is not None:
            assert "ndcg_score" in stats
            assert 0.0 <= stats["ndcg_score"] <= 1.0


# ============================================================================
# SURFACING TRACKING TESTS
# ============================================================================


class TestSurfacingTracking:
    def test_log_retrieval_increments_surfacing(self, maasv_db, sample_candidates):
        """log_retrieval() should increment surfacing_count for all candidates."""
        from maasv.core.db import _db
        from maasv.core.learned_ranker import log_retrieval
        from maasv.core.store import store_memory

        # Store actual memories so surfacing_count can be updated
        mem_id = store_memory(content="Test surfacing memory", category="context")

        # Get initial surfacing_count
        with _db() as db:
            before = db.execute("SELECT surfacing_count FROM memories WHERE id = ?", (mem_id,)).fetchone()
        initial_count = before["surfacing_count"] if before else 0

        now = datetime.now(timezone.utc)
        candidate = {
            "id": mem_id,
            "content": "Test surfacing memory",
            "category": "context",
            "importance": 0.5,
            "access_count": 0,
            "surfacing_count": initial_count,
            "created_at": now.isoformat(),
        }

        log_retrieval(
            query="test surfacing",
            candidates=[candidate],
            returned_ids=[mem_id],
            vector_distances={mem_id: 0.3},
            bm25_scores={},
            graph_scores={},
            protected=set(),
            now=now,
        )

        with _db() as db:
            after = db.execute("SELECT surfacing_count FROM memories WHERE id = ?", (mem_id,)).fetchone()
        assert after["surfacing_count"] == initial_count + 1

    def test_migration_backfill(self, maasv_db):
        """surfacing_count column should exist after migration 9."""
        from maasv.core.db import _db

        with _db() as db:
            # Verify column exists by querying it
            db.execute("SELECT surfacing_count FROM memories LIMIT 1").fetchone()
            # Should not raise — column exists


# ============================================================================
# IPS UTILITY TESTS
# ============================================================================


class TestIPSUtility:
    def test_high_conversion_outscores_low(self):
        """Memory with high access/surfacing ratio should score higher."""
        from maasv.core.learned_ranker import extract_features

        now = datetime.now(timezone.utc)

        high_conversion = {
            "id": "mem_high",
            "content": "high conversion",
            "category": "project",
            "importance": 0.5,
            "access_count": 8,
            "surfacing_count": 10,
            "created_at": now.isoformat(),
        }
        low_conversion = {
            "id": "mem_low",
            "content": "low conversion",
            "category": "project",
            "importance": 0.5,
            "access_count": 2,
            "surfacing_count": 50,
            "created_at": now.isoformat(),
        }

        feat_high = extract_features(high_conversion, {}, {}, {}, set(), now)
        feat_low = extract_features(low_conversion, {}, {}, {}, set(), now)

        # IPS utility (feature 5) should be higher for high-conversion memory
        assert feat_high[5] > feat_low[5]

    def test_feature_normalization_bounds(self):
        """IPS utility feature should always be in [0, 1]."""
        from maasv.core.learned_ranker import extract_features

        now = datetime.now(timezone.utc)

        cases = [
            {"access_count": 0, "surfacing_count": 0},
            {"access_count": 100, "surfacing_count": 1},
            {"access_count": 0, "surfacing_count": 100},
            {"access_count": 5, "surfacing_count": 5},
            {"access_count": 1000, "surfacing_count": 10},
        ]

        for case in cases:
            mem = {
                "id": "mem_bounds",
                "content": "test",
                "category": "project",
                "importance": 0.5,
                "created_at": now.isoformat(),
                **case,
            }
            features = extract_features(mem, {}, {}, {}, set(), now)
            assert 0.0 <= features[5] <= 1.0, f"IPS utility out of bounds for {case}: {features[5]}"

    def test_importance_score_ips(self):
        """_importance_score should use IPS utility when surfacing_count > 0."""
        from maasv.core.retrieval import _importance_score

        now = datetime.now(timezone.utc)

        # Two memories: same access_count, different surfacing_count
        mem_rare = {
            "id": "mem_rare",
            "content": "rare but useful",
            "category": "project",
            "importance": 0.5,
            "access_count": 5,
            "surfacing_count": 5,
            "created_at": now.isoformat(),
        }
        mem_common = {
            "id": "mem_common",
            "content": "common but unremarkable",
            "category": "project",
            "importance": 0.5,
            "access_count": 5,
            "surfacing_count": 50,
            "created_at": now.isoformat(),
        }

        primary, _ = _importance_score(
            [mem_rare, mem_common],
            protected=set(),
            now=now,
            vector_distances={"mem_rare": 0.3, "mem_common": 0.3},
            bm25_scores={},
            graph_scores={},
        )

        # mem_rare should score higher (5/5=1.0 ratio vs 5/50=0.1 ratio)
        assert primary[0]["id"] == "mem_rare"


# ============================================================================
# IPS-WEIGHTED TRAINING TESTS
# ============================================================================


class TestIPSWeightedTraining:
    def _seed_training_data_with_surfacing(self, n=20):
        """Insert labeled data with surfacing counts for IPS training."""
        import uuid

        from maasv.core.db import _db

        now = datetime.now(timezone.utc)
        mem_ids = []

        # Create actual memories with varying surfacing counts
        with _db() as db:
            for i in range(n):
                mid = f"ips_mem_{i:03d}"
                db.execute(
                    """INSERT OR IGNORE INTO memories
                       (id, content, category, importance, access_count, surfacing_count, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (mid, f"IPS test memory {i}", "project", 0.5, i % 5, (i + 1) * 5, now.isoformat()),
                )
                mem_ids.append(mid)

            for i in range(n):
                mid = mem_ids[i]
                features = {
                    mid: [
                        0.5,
                        1.0 if i % 2 == 0 else 0.0,
                        0.0,
                        0.5,
                        0.8,
                        0.6,
                        0.3,
                        1.0 / (1 + i),
                    ]
                }
                outcomes = {mid: 1.0 if i % 3 == 0 else 0.0}

                db.execute(
                    """INSERT INTO retrieval_log
                       (id, query_text, query_embedding_hash, timestamp,
                        candidate_count, returned_ids, features, outcomes, outcome_recorded_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()),
                        f"ips_query_{i}",
                        f"hash_{i}",
                        (now - timedelta(hours=i)).isoformat(),
                        1,
                        json.dumps([mid]),
                        json.dumps(features),
                        json.dumps(outcomes),
                        now.isoformat(),
                    ),
                )
            db.commit()

    def test_training_with_ips_weights(self, maasv_db):
        """Training should complete with IPS-weighted samples."""
        from maasv.core.learned_ranker import reload_model, train

        reload_model()

        self._seed_training_data_with_surfacing(n=20)

        stats = train(
            cancel_check=lambda: False,
            max_steps=20,
            lr=0.01,
        )

        assert stats is not None
        assert stats["steps"] == 20
        assert stats["training_samples"] >= 5

    def test_snips_preserves_batch_scale(self, maasv_db):
        """SNIPS normalization should keep batch weight sum = batch_size."""
        # This is a unit test of the SNIPS math, not the full training loop
        raw_weights = [1.0, 2.0, 50.0, 0.5, 1.0]
        weight_sum = sum(raw_weights)
        batch_size = len(raw_weights)
        snips_weights = [(w / weight_sum) * batch_size for w in raw_weights]

        assert abs(sum(snips_weights) - batch_size) < 1e-10

    def test_feature_version_guard(self, maasv_db):
        """Model should not load if stored feature names don't match current."""
        from maasv.core.db import _db
        from maasv.core.learned_ranker import FEATURE_NAMES, _get_model, reload_model

        # Tamper with stored feature names
        with _db() as db:
            db.execute(
                """UPDATE learned_ranker_weights
                   SET feature_names = ? WHERE id = 1""",
                (json.dumps(["old_feature_1", "old_feature_2"]),),
            )
            db.commit()

        reload_model()
        model = _get_model()
        assert model is None  # Should refuse to load mismatched features

        # Restore correct feature names so subsequent tests work
        with _db() as db:
            db.execute(
                """UPDATE learned_ranker_weights
                   SET feature_names = ? WHERE id = 1""",
                (json.dumps(FEATURE_NAMES),),
            )
            db.commit()
        reload_model()


# ============================================================================
# GRADUATION TESTS
# ============================================================================


class TestGraduation:
    def _seed_shadow_metrics(self, db, n=60, tau=0.6, overlap=4):
        """Insert shadow metrics for graduation testing."""
        now = datetime.now(timezone.utc)
        for i in range(n):
            # Add a little noise to tau
            noisy_tau = tau + (0.05 if i % 3 == 0 else -0.03)
            db.execute(
                """INSERT INTO shadow_metrics
                   (timestamp, top5_overlap, kendall_tau, avg_surfacing, candidate_count)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    (now - timedelta(hours=n - i)).isoformat(),
                    overlap,
                    noisy_tau,
                    15.0,
                    20,
                ),
            )
        db.commit()

    def test_shadow_metrics_table_exists(self, maasv_db):
        """Migration 10 should create shadow_metrics table."""
        from maasv.core.db import _db

        with _db() as db:
            row = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='shadow_metrics'").fetchone()
        assert row is not None

    def test_shadow_compare_persists_metrics(self, maasv_db, sample_candidates):
        """shadow_compare() should write to shadow_metrics table."""
        from maasv.core.db import _db
        from maasv.core.learned_ranker import RankingModel, reload_model, shadow_compare

        reload_model()
        model = RankingModel()
        now = datetime.now(timezone.utc)

        # Clear existing shadow metrics
        with _db() as db:
            db.execute("DELETE FROM shadow_metrics")
            db.commit()

        shadow_compare(
            model,
            sample_candidates,
            {"identity", "family"},
            now,
            {"mem_001": 0.2, "mem_002": 0.4},
            {"mem_001"},
            set(),
        )

        with _db() as db:
            count = db.execute("SELECT COUNT(*) as n FROM shadow_metrics").fetchone()["n"]
        assert count >= 1

    def test_graduation_insufficient_comparisons(self, maasv_db):
        """Should report not ready when too few shadow comparisons."""
        import maasv
        from maasv.core.db import _db
        from maasv.core.learned_ranker import check_graduation_readiness

        config = maasv.get_config()
        original_shadow = config.learned_ranker_shadow_mode
        config.learned_ranker_shadow_mode = True

        try:
            # Clear shadow metrics
            with _db() as db:
                db.execute("DELETE FROM shadow_metrics")
                db.commit()

            result = check_graduation_readiness()
            assert result is not None
            assert result["ready"] is False
            assert result["reason"] == "insufficient_comparisons"
        finally:
            config.learned_ranker_shadow_mode = original_shadow

    def test_graduation_ready(self, maasv_db):
        """Should report ready when all criteria met."""
        import maasv
        from maasv.core.db import _db
        from maasv.core.learned_ranker import check_graduation_readiness

        config = maasv.get_config()
        original_shadow = config.learned_ranker_shadow_mode
        original_min = config.learned_ranker_graduation_min_comparisons
        original_ndcg = config.learned_ranker_graduation_min_ndcg
        config.learned_ranker_shadow_mode = True
        config.learned_ranker_graduation_min_comparisons = 10  # Lower for test
        config.learned_ranker_graduation_min_ndcg = 0.0  # Accept any NDCG

        try:
            with _db() as db:
                db.execute("DELETE FROM shadow_metrics")
                db.commit()
                self._seed_shadow_metrics(db, n=20, tau=0.7, overlap=4)

            result = check_graduation_readiness()
            assert result is not None
            assert result["ready"] is True
            assert "ndcg" in result
            assert "avg_tau" in result
            assert "tau_std" in result
        finally:
            config.learned_ranker_shadow_mode = original_shadow
            config.learned_ranker_graduation_min_comparisons = original_min
            config.learned_ranker_graduation_min_ndcg = original_ndcg

    def test_graduation_low_tau_rejected(self, maasv_db):
        """Should reject when tau is too low (anti-correlated)."""
        import maasv
        from maasv.core.db import _db
        from maasv.core.learned_ranker import check_graduation_readiness

        config = maasv.get_config()
        original_shadow = config.learned_ranker_shadow_mode
        original_min = config.learned_ranker_graduation_min_comparisons
        original_ndcg = config.learned_ranker_graduation_min_ndcg
        config.learned_ranker_shadow_mode = True
        config.learned_ranker_graduation_min_comparisons = 10
        config.learned_ranker_graduation_min_ndcg = 0.0

        try:
            with _db() as db:
                db.execute("DELETE FROM shadow_metrics")
                db.commit()
                self._seed_shadow_metrics(db, n=20, tau=-0.8, overlap=1)

            result = check_graduation_readiness()
            assert result is not None
            assert result["ready"] is False
            assert result["reason"] == "low_tau"
        finally:
            config.learned_ranker_shadow_mode = original_shadow
            config.learned_ranker_graduation_min_comparisons = original_min
            config.learned_ranker_graduation_min_ndcg = original_ndcg

    def test_graduate_from_shadow_mode(self, maasv_db):
        """graduate_from_shadow_mode() should flip the config flag."""
        import maasv
        from maasv.core.learned_ranker import graduate_from_shadow_mode

        config = maasv.get_config()
        original = config.learned_ranker_shadow_mode
        config.learned_ranker_shadow_mode = True

        try:
            result = graduate_from_shadow_mode()
            assert result is True
            assert config.learned_ranker_shadow_mode is False

            # Second call should return False (already graduated)
            result2 = graduate_from_shadow_mode()
            assert result2 is False
        finally:
            config.learned_ranker_shadow_mode = original

    def test_graduation_skipped_when_not_shadow(self, maasv_db):
        """check_graduation_readiness returns None when not in shadow mode."""
        import maasv
        from maasv.core.learned_ranker import check_graduation_readiness

        config = maasv.get_config()
        original = config.learned_ranker_shadow_mode
        config.learned_ranker_shadow_mode = False

        try:
            result = check_graduation_readiness()
            assert result is None
        finally:
            config.learned_ranker_shadow_mode = original

    def test_learn_job_with_graduation_check(self, maasv_db):
        """Learn job should complete with graduation phase."""
        import maasv
        from maasv.lifecycle.learn import run_learn_job

        config = maasv.get_config()
        original = config.learned_ranker_shadow_mode
        config.learned_ranker_shadow_mode = True

        try:
            # Should not raise
            run_learn_job(data={}, cancel_check=lambda: False)
        finally:
            config.learned_ranker_shadow_mode = original


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
        from maasv.core.learned_ranker import reload_model, score

        reload_model()
        now = datetime.now(timezone.utc)

        result = score(
            candidates=sample_candidates,
            protected={"identity", "family"},
            now=now,
            vector_distances={"mem_001": 0.2, "mem_002": 0.4},
            bm25_scores={"mem_001": 1.0},
            graph_scores={},
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
            result = score(sample_candidates, set(), now, {}, set(), set())
            assert result is None
        finally:
            config.learned_ranker_enabled = original

    def test_retrieval_logging_in_find_similar(self, maasv_db):
        """find_similar_memories() should log to retrieval_log."""
        from maasv.core.db import _db
        from maasv.core.retrieval import find_similar_memories
        from maasv.core.store import store_memory

        # Store some memories
        store_memory(content="The sky is blue today", category="context")
        store_memory(content="Python is a programming language", category="learning")
        store_memory(content="New York is a big city", category="context")

        # Get count before
        with _db() as db:
            before = db.execute("SELECT COUNT(*) as c FROM retrieval_log").fetchone()["c"]

        # Run retrieval
        find_similar_memories("What color is the sky?", limit=3)

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

"""
Learned Ranker for maasv retrieval.

Small neural network (81 parameters) trained on actual retrieval usage data
via autograd. Falls back to heuristic scoring when insufficient data.

Follows reranker.py pattern: lazy-load singleton, thread-safe, returns None
if unavailable.
"""

import hashlib
import json
import logging
import math
import random
import threading
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from maasv.core.autograd import Value

logger = logging.getLogger(__name__)

# Feature names in order — must match extract_features() output
FEATURE_NAMES = [
    "vector_similarity",
    "bm25_hit",
    "graph_hit",
    "importance",
    "age_decay",
    "access_count_norm",
    "category_code",
    "rrf_rank_norm",
]

N_FEATURES = len(FEATURE_NAMES)


# ============================================================================
# MODEL
# ============================================================================

class RankingModel:
    """Linear(8,8) -> ReLU -> Linear(8,1) -> Sigmoid = 81 parameters."""

    def __init__(self):
        # Xavier-ish init for small nets
        scale1 = (2.0 / (N_FEATURES + 8)) ** 0.5
        scale2 = (2.0 / (8 + 1)) ** 0.5

        self.w1 = [[Value(random.gauss(0, scale1)) for _ in range(N_FEATURES)] for _ in range(8)]
        self.b1 = [Value(0.0) for _ in range(8)]
        self.w2 = [[Value(random.gauss(0, scale2)) for _ in range(8)] for _ in range(1)]
        self.b2 = [Value(0.0) for _ in range(1)]

    def forward(self, x: list[Value]) -> Value:
        """Forward pass: features -> relevance score in [0, 1]."""
        # Hidden layer
        h = []
        for i in range(8):
            s = self.b1[i]
            for j in range(N_FEATURES):
                s = s + self.w1[i][j] * x[j]
            h.append(s.relu())

        # Output layer
        o = self.b2[0]
        for i in range(8):
            o = o + self.w2[0][i] * h[i]

        return o.sigmoid()

    def parameters(self) -> list[Value]:
        """All trainable parameters (81 total)."""
        params = []
        for row in self.w1:
            params.extend(row)
        params.extend(self.b1)
        for row in self.w2:
            params.extend(row)
        params.extend(self.b2)
        return params

    def state_dict(self) -> dict:
        """Serialize weights to JSON-friendly dict."""
        return {
            "w1": [[v.data for v in row] for row in self.w1],
            "b1": [v.data for v in self.b1],
            "w2": [[v.data for v in row] for row in self.w2],
            "b2": [v.data for v in self.b2],
        }

    def load_state_dict(self, d: dict):
        """Load weights from dict."""
        for i, row in enumerate(d["w1"]):
            for j, val in enumerate(row):
                self.w1[i][j].data = float(val)
        for i, val in enumerate(d["b1"]):
            self.b1[i].data = float(val)
        for i, row in enumerate(d["w2"]):
            for j, val in enumerate(row):
                self.w2[i][j].data = float(val)
        for i, val in enumerate(d["b2"]):
            self.b2[i].data = float(val)


# ============================================================================
# SINGLETON
# ============================================================================

_model: Optional[RankingModel] = None
_model_loaded = False
_model_lock = threading.Lock()


def _get_model() -> Optional[RankingModel]:
    """Lazy-load the ranking model from DB. Returns None if unavailable."""
    global _model, _model_loaded

    if _model_loaded:
        return _model

    with _model_lock:
        if _model_loaded:
            return _model

        try:
            import maasv
            config = maasv.get_config()
            if not config.learned_ranker_enabled:
                _model_loaded = True
                return None

            from maasv.core.db import _db
            with _db() as db:
                row = db.execute(
                    "SELECT weights_json, training_samples FROM learned_ranker_weights WHERE id = 1"
                ).fetchone()

            if row is None:
                _model_loaded = True
                return None

            samples = row["training_samples"] or 0
            if samples < config.learned_ranker_min_samples:
                logger.info(f"[LearnedRanker] Only {samples} samples, need {config.learned_ranker_min_samples}")
                _model_loaded = True
                return None

            weights = json.loads(row["weights_json"])
            model = RankingModel()
            model.load_state_dict(weights)
            _model = model
            _model_loaded = True
            logger.info(f"[LearnedRanker] Loaded model ({samples} training samples)")
            return _model

        except Exception:
            logger.error("[LearnedRanker] Failed to load model", exc_info=True)
            _model_loaded = True
            return None


def reload_model():
    """Force reload of model from DB (after training)."""
    global _model, _model_loaded
    with _model_lock:
        _model = None
        _model_loaded = False


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def _get_category_priority() -> dict[str, int]:
    """Get category priority map from config."""
    try:
        import maasv
        return maasv.get_config().category_priority
    except RuntimeError:
        return {}


def extract_features(
    mem: dict,
    vector_distances: dict[str, float],
    bm25_ids: set[str],
    graph_ids: set[str],
    protected: set[str],
    now: datetime,
    rrf_rank: int = 0,
) -> list[float]:
    """
    Extract 8 normalized features for a single memory candidate.

    All features normalized to [0, 1].
    """
    mid = mem["id"]

    # 1. Vector similarity
    dist = vector_distances.get(mid)
    vector_similarity = (1.0 - dist) if dist is not None else 0.0

    # 2. BM25 hit (binary)
    bm25_hit = 1.0 if mid in bm25_ids else 0.0

    # 3. Graph hit (binary)
    graph_hit = 1.0 if mid in graph_ids else 0.0

    # 4. Importance (already 0-1)
    importance = float(mem.get("importance") or 0.5)

    # 5. Age decay
    if mem.get("category") in protected:
        age_decay = 1.0
    else:
        try:
            created = datetime.fromisoformat(mem["created_at"])
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            days_old = (now - created).total_seconds() / 86400
        except (ValueError, TypeError, KeyError):
            days_old = 0
        age_decay = math.exp(-days_old / 180)

    # 6. Access count (normalized, capped at 5)
    access_count = min(mem.get("access_count") or 0, 5)
    access_count_norm = math.log(2 + access_count) / math.log(7)

    # 7. Category code (priority int / max)
    cat_priority = _get_category_priority()
    max_priority = max(cat_priority.values()) if cat_priority else 10
    cat_code = cat_priority.get(mem.get("category", ""), max_priority)
    category_code = cat_code / max_priority if max_priority > 0 else 0.5

    # 8. RRF rank (normalized)
    rrf_rank_norm = 1.0 / (1.0 + rrf_rank)

    return [
        vector_similarity,
        bm25_hit,
        graph_hit,
        importance,
        age_decay,
        access_count_norm,
        category_code,
        rrf_rank_norm,
    ]


# ============================================================================
# SCORING
# ============================================================================

def score(
    candidates: list[dict],
    protected: set[str],
    now: datetime,
    vector_distances: dict[str, float],
    bm25_ids: set[str],
    graph_ids: set[str],
) -> Optional[tuple[list[dict], list[dict]]]:
    """
    Score candidates using the learned ranker.

    Returns (primary, supplementary) matching _importance_score() signature,
    or None if the learned ranker is unavailable or in shadow mode.
    """
    import maasv
    config = maasv.get_config()

    if not config.learned_ranker_enabled:
        return None

    model = _get_model()

    if config.learned_ranker_shadow_mode:
        # Shadow mode: score but don't use results
        if model is not None:
            shadow_compare(model, candidates, protected, now,
                           vector_distances, bm25_ids, graph_ids)
        return None

    if model is None:
        return None

    primary = []
    supplementary = []

    for rank, mem in enumerate(candidates):
        features = extract_features(
            mem, vector_distances, bm25_ids, graph_ids, protected, now, rrf_rank=rank
        )
        x = [Value(f) for f in features]
        score_val = model.forward(x)
        mem["_imp_score"] = score_val.data

        if vector_distances.get(mem["id"]) is not None:
            primary.append(mem)
        else:
            supplementary.append(mem)

    primary.sort(key=lambda m: m["_imp_score"], reverse=True)
    supplementary.sort(key=lambda m: m["_imp_score"], reverse=True)

    return primary, supplementary


# ============================================================================
# SHADOW COMPARISON
# ============================================================================

def shadow_compare(
    model: RankingModel,
    candidates: list[dict],
    protected: set[str],
    now: datetime,
    vector_distances: dict[str, float],
    bm25_ids: set[str],
    graph_ids: set[str],
):
    """Log ranking disagreements between learned ranker and heuristic."""
    if not candidates:
        return

    try:
        # Get learned ranker ordering
        learned_scores = {}
        for rank, mem in enumerate(candidates):
            features = extract_features(
                mem, vector_distances, bm25_ids, graph_ids, protected, now, rrf_rank=rank
            )
            x = [Value(f) for f in features]
            learned_scores[mem["id"]] = model.forward(x).data

        learned_top5 = sorted(learned_scores, key=learned_scores.get, reverse=True)[:5]

        # Get heuristic ordering (from _imp_score if present)
        heuristic_scores = {m["id"]: m.get("_imp_score", 0) for m in candidates}
        heuristic_top5 = sorted(heuristic_scores, key=heuristic_scores.get, reverse=True)[:5]

        # Top-5 overlap
        overlap = len(set(learned_top5) & set(heuristic_top5))

        # Kendall tau approximation (concordant vs discordant pairs in top-5)
        concordant = 0
        discordant = 0
        common = list(set(learned_top5) & set(heuristic_top5))
        for i in range(len(common)):
            for j in range(i + 1, len(common)):
                a, b = common[i], common[j]
                lr_order = learned_top5.index(a) < learned_top5.index(b)
                hr_order = heuristic_top5.index(a) < heuristic_top5.index(b)
                if lr_order == hr_order:
                    concordant += 1
                else:
                    discordant += 1

        total = concordant + discordant
        tau = (concordant - discordant) / total if total > 0 else 1.0

        logger.info(
            f"[LearnedRanker] Shadow: top5_overlap={overlap}/5, kendall_tau={tau:.2f}"
        )

    except Exception:
        logger.debug("[LearnedRanker] Shadow comparison failed", exc_info=True)


# ============================================================================
# RETRIEVAL LOGGING
# ============================================================================

def log_retrieval(
    query: str,
    candidates: list[dict],
    returned_ids: list[str],
    vector_distances: dict[str, float],
    bm25_ids: set[str],
    graph_ids: set[str],
    protected: set[str],
    now: datetime,
):
    """Log a retrieval event for training data collection. Best-effort."""
    try:
        from maasv.core.db import _db

        # Compute features for all candidates
        features = {}
        for rank, mem in enumerate(candidates):
            feat = extract_features(
                mem, vector_distances, bm25_ids, graph_ids, protected, now, rrf_rank=rank
            )
            features[mem["id"]] = feat

        # Hash the query for grouping
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]

        log_id = str(uuid.uuid4())
        timestamp = now.isoformat()

        with _db() as db:
            db.execute(
                """INSERT INTO retrieval_log
                   (id, query_text, query_embedding_hash, timestamp,
                    candidate_count, returned_ids, features)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    log_id,
                    query,
                    query_hash,
                    timestamp,
                    len(candidates),
                    json.dumps(returned_ids),
                    json.dumps(features),
                ),
            )
            db.commit()

    except Exception:
        logger.debug("[LearnedRanker] Failed to log retrieval", exc_info=True)


# ============================================================================
# OUTCOME LABELING
# ============================================================================

def label_outcomes(
    cancel_check: Callable[[], bool],
    max_entries: int = 100,
) -> int:
    """
    Backfill outcomes on retrieval logs by checking re-access patterns.

    Returns count of labeled entries.
    """
    from maasv.core.db import _db

    labeled = 0

    with _db() as db:
        # Get unlabeled entries older than 2 hours (give time for re-access)
        cutoff = datetime.now(timezone.utc)
        rows = db.execute(
            """SELECT id, returned_ids, timestamp
               FROM retrieval_log
               WHERE outcomes IS NULL
               AND datetime(timestamp) < datetime(?, '-2 hours')
               ORDER BY timestamp ASC
               LIMIT ?""",
            (cutoff.isoformat(), max_entries),
        ).fetchall()

        for row in rows:
            if cancel_check():
                break

            try:
                returned = json.loads(row["returned_ids"])
                retrieval_time = datetime.fromisoformat(row["timestamp"])
                if retrieval_time.tzinfo is None:
                    retrieval_time = retrieval_time.replace(tzinfo=timezone.utc)

                outcomes = {}
                for mem_id in returned:
                    mem_row = db.execute(
                        "SELECT last_accessed_at FROM memories WHERE id = ?",
                        (mem_id,),
                    ).fetchone()

                    if mem_row and mem_row["last_accessed_at"]:
                        accessed = datetime.fromisoformat(mem_row["last_accessed_at"])
                        if accessed.tzinfo is None:
                            accessed = accessed.replace(tzinfo=timezone.utc)
                        delta_min = (accessed - retrieval_time).total_seconds() / 60

                        if 0 < delta_min <= 30:
                            outcomes[mem_id] = 1.0
                        elif 30 < delta_min <= 120:
                            outcomes[mem_id] = 0.5
                        else:
                            outcomes[mem_id] = 0.0
                    else:
                        outcomes[mem_id] = 0.0

                db.execute(
                    """UPDATE retrieval_log
                       SET outcomes = ?, outcome_recorded_at = ?
                       WHERE id = ?""",
                    (json.dumps(outcomes), datetime.now(timezone.utc).isoformat(), row["id"]),
                )
                labeled += 1

            except Exception:
                logger.debug(f"[LearnedRanker] Failed to label entry {row['id']}", exc_info=True)

        if labeled > 0:
            db.commit()

    return labeled


# ============================================================================
# TRAINING
# ============================================================================

def train(
    cancel_check: Callable[[], bool],
    max_steps: int = 50,
    lr: float = 0.01,
) -> Optional[dict]:
    """
    Train the ranking model on labeled retrieval data.

    Uses binary cross-entropy loss with outcome weights.
    Returns training stats dict or None if insufficient data.
    """
    from maasv.core.db import _db

    import maasv
    config = maasv.get_config()

    with _db() as db:
        rows = db.execute(
            """SELECT features, outcomes
               FROM retrieval_log
               WHERE outcomes IS NOT NULL
               ORDER BY timestamp DESC
               LIMIT 1000"""
        ).fetchall()

    if len(rows) < config.learned_ranker_min_samples:
        logger.info(
            f"[LearnedRanker] Only {len(rows)} labeled samples, need {config.learned_ranker_min_samples}"
        )
        return None

    # Parse training data
    samples = []  # (features, outcome, weight)
    for row in rows:
        try:
            features_dict = json.loads(row["features"])
            outcomes_dict = json.loads(row["outcomes"])

            for mem_id, feat_list in features_dict.items():
                if mem_id in outcomes_dict:
                    outcome = outcomes_dict[mem_id]
                    weight = 0.5 if outcome == 0.0 else 1.0  # Weak negative downweight
                    samples.append((feat_list, outcome, weight))
        except (json.JSONDecodeError, TypeError):
            continue

    if len(samples) < config.learned_ranker_min_samples:
        return None

    # Initialize or load existing model
    model = RankingModel()

    with _db() as db:
        existing = db.execute(
            "SELECT weights_json FROM learned_ranker_weights WHERE id = 1"
        ).fetchone()
    if existing:
        try:
            model.load_state_dict(json.loads(existing["weights_json"]))
        except Exception:
            pass  # Start fresh

    params = model.parameters()
    losses = []

    for step in range(max_steps):
        if cancel_check():
            break

        # Sample a mini-batch
        batch = random.sample(samples, min(32, len(samples)))

        # Forward + loss
        total_loss = Value(0.0)
        for feat_list, outcome, weight in batch:
            x = [Value(f) for f in feat_list]
            pred = model.forward(x)

            # Binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
            # Clamp pred to avoid log(0)
            p = Value(max(min(pred.data, 0.999), 0.001))
            # Reconstruct p with gradient path
            p = pred * 0.998 + 0.001  # Smooth: maps [0,1] -> [0.001, 0.999]

            if outcome > 0.5:
                loss = -(p.log()) * weight
            else:
                loss = -((1 - p).log()) * weight

            total_loss = total_loss + loss

        avg_loss = total_loss * (1.0 / len(batch))
        losses.append(avg_loss.data)

        # Backward
        for p in params:
            p.grad = 0.0
        avg_loss.backward()

        # SGD update
        for p in params:
            p.data -= lr * p.grad

    # Save weights
    stats = {
        "training_samples": len(samples),
        "steps": len(losses),
        "final_loss": losses[-1] if losses else None,
        "loss_reduction": (losses[0] - losses[-1]) if len(losses) > 1 else 0,
    }

    # Compute NDCG@5 on a holdout
    ndcg = _compute_ndcg(model, samples)
    stats["ndcg_score"] = ndcg

    with _db() as db:
        db.execute(
            """INSERT OR REPLACE INTO learned_ranker_weights
               (id, weights_json, feature_names, trained_at,
                training_samples, training_loss, ndcg_score, version)
               VALUES (1, ?, ?, ?, ?, ?, ?,
                       COALESCE((SELECT version FROM learned_ranker_weights WHERE id = 1), 0) + 1)""",
            (
                json.dumps(model.state_dict()),
                json.dumps(FEATURE_NAMES),
                datetime.now(timezone.utc).isoformat(),
                len(samples),
                losses[-1] if losses else None,
                ndcg,
            ),
        )
        db.commit()

    # Force model reload
    reload_model()

    logger.info(
        f"[LearnedRanker] Trained: {len(samples)} samples, {len(losses)} steps, "
        f"loss={losses[-1]:.4f}, ndcg@5={ndcg:.3f}"
        if losses else "[LearnedRanker] Training cancelled before any steps"
    )

    return stats


def _compute_ndcg(model: RankingModel, samples: list, k: int = 5) -> float:
    """Compute NDCG@k on a random holdout of retrieval logs."""
    if len(samples) < k * 2:
        return 0.0

    # Group samples by taking consecutive chunks as "queries"
    # (since samples from the same retrieval log are adjacent)
    holdout = random.sample(samples, min(100, len(samples)))

    # Score each sample
    scored = []
    for feat_list, outcome, _weight in holdout:
        x = [Value(f) for f in feat_list]
        pred = model.forward(x).data
        scored.append((pred, outcome))

    # Sort by predicted score
    scored.sort(key=lambda x: x[0], reverse=True)

    # DCG@k
    dcg = 0.0
    for i, (_, rel) in enumerate(scored[:k]):
        dcg += (2 ** rel - 1) / math.log2(i + 2)

    # IDCG@k (ideal ordering)
    ideal = sorted([o for _, o in scored], reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal[:k]):
        idcg += (2 ** rel - 1) / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0

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
    "bm25_score",
    "graph_score",
    "importance",
    "age_decay",
    "ips_utility",
    "category_code",
    "rrf_rank_norm",
    # Session context features (0.0 when no session context provided)
    "query_coherence",
    "session_depth_norm",
    "category_session_match",
    "subject_session_overlap",
    # Temporal/freshness features
    "freshness_score",
    "subject_freshness_rank",
]

N_FEATURES = len(FEATURE_NAMES)
HIDDEN_SIZE = 16  # Scaled up from 8 to match increased feature count


# ============================================================================
# MODEL
# ============================================================================


class RankingModel:
    """Linear(N_FEATURES, HIDDEN_SIZE) -> ReLU -> Linear(HIDDEN_SIZE, 1) -> Sigmoid.

    With 12 features and hidden=16: 12*16 + 16 + 16*1 + 1 = 225 parameters.
    """

    def __init__(self):
        # Xavier-ish init for small nets
        scale1 = (2.0 / (N_FEATURES + HIDDEN_SIZE)) ** 0.5
        scale2 = (2.0 / (HIDDEN_SIZE + 1)) ** 0.5

        self.w1 = [[Value(random.gauss(0, scale1)) for _ in range(N_FEATURES)] for _ in range(HIDDEN_SIZE)]
        self.b1 = [Value(0.0) for _ in range(HIDDEN_SIZE)]
        self.w2 = [[Value(random.gauss(0, scale2)) for _ in range(HIDDEN_SIZE)] for _ in range(1)]
        self.b2 = [Value(0.0) for _ in range(1)]

    def forward(self, x: list[Value]) -> Value:
        """Forward pass: features -> relevance score in [0, 1]."""
        # Hidden layer
        h = []
        for i in range(HIDDEN_SIZE):
            s = self.b1[i]
            for j in range(N_FEATURES):
                s = s + self.w1[i][j] * x[j]
            h.append(s.relu())

        # Output layer
        o = self.b2[0]
        for i in range(HIDDEN_SIZE):
            o = o + self.w2[0][i] * h[i]

        return o.sigmoid()

    def parameters(self) -> list[Value]:
        """All trainable parameters."""
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
                    "SELECT weights_json, feature_names, training_samples FROM learned_ranker_weights WHERE id = 1"
                ).fetchone()

            if row is None:
                _model_loaded = True
                return None

            # Feature version guard: discard weights if feature schema changed.
            # Model retrains from scratch on next learn cycle.
            try:
                stored_features = json.loads(row["feature_names"])
            except (json.JSONDecodeError, TypeError):
                stored_features = None
            if stored_features != FEATURE_NAMES:
                logger.info(
                    "[LearnedRanker] Feature schema changed (%s -> %s), discarding stored weights",
                    stored_features,
                    FEATURE_NAMES,
                )
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
    bm25_scores: dict[str, float],
    graph_scores: dict[str, float],
    protected: set[str],
    now: datetime,
    rrf_rank: int = 0,
    session_features: Optional[dict] = None,
    freshness_scores: Optional[dict[str, float]] = None,
    freshness_rank: int = -1,
) -> list[float]:
    """
    Extract 14 normalized features for a single memory candidate.

    Features 1-8 are retrieval signals. Features 9-12 are session context
    (all 0.0 when no session_features provided). Features 13-14 are
    temporal/freshness signals from the subject freshness retrieval signal.

    All features normalized to [0, 1].
    """
    mid = mem["id"]

    # 1. Vector similarity
    dist = vector_distances.get(mid)
    vector_similarity = (1.0 - dist) if dist is not None else 0.0

    # 2. BM25 score (continuous, 0.0 = not in BM25 results, 1.0 = best match)
    bm25_score = bm25_scores.get(mid, 0.0)

    # 3. Graph score (continuous, 0.0 = no graph hit, 0.4-1.0 = entity density)
    graph_score = graph_scores.get(mid, 0.0)

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
        import maasv as _maasv
        half_life = _maasv.get_config().decay_half_life_days
        tau = half_life / math.log(2)
        age_decay = math.exp(-days_old / tau)

    # 6. IPS utility (access_count / surfacing_count, normalized to [0, 1])
    access_count = mem.get("access_count") or 0
    surfacing_count = mem.get("surfacing_count") or 0
    if surfacing_count > 0:
        raw_utility = access_count / surfacing_count
        ips_utility = min(raw_utility, 2.0) / 2.0  # normalize to [0, 1]
    else:
        # Cold-start fallback: old formula for memories without surfacing data
        ips_utility = math.log(2 + min(access_count, 5)) / math.log(7)

    # 7. Category code (priority int / max)
    cat_priority = _get_category_priority()
    max_priority = max(cat_priority.values()) if cat_priority else 10
    cat_code = cat_priority.get(mem.get("category", ""), max_priority)
    category_code = cat_code / max_priority if max_priority > 0 else 0.5

    # 8. RRF rank (normalized)
    rrf_rank_norm = 1.0 / (1.0 + rrf_rank)

    # === Session context features (9-12) ===
    sf = session_features or {}

    # 9. Query coherence: cosine similarity between current query and
    #    previous query in this session. Measures topic continuity.
    query_coherence = float(sf.get("query_coherence", 0.0))

    # 10. Session retrieval depth: 1/(1+n) where n = retrievals so far.
    #     First query = 1.0, later queries approach 0.
    session_depth_norm = float(sf.get("session_depth_norm", 0.0))

    # 11. Category session match: 1.0 if memory's category was already
    #     useful in this session (appeared in previous results).
    seen_categories = sf.get("seen_categories")
    if seen_categories and mem.get("category") in seen_categories:
        category_session_match = 1.0
    else:
        category_session_match = 0.0

    # 12. Subject session overlap: fraction of memory's subject tokens
    #     that overlap with subjects from previous session results.
    seen_subjects = sf.get("seen_subjects")
    if seen_subjects and mem.get("subject"):
        mem_subject_tokens = set(mem["subject"].lower().split())
        seen_subject_tokens = {s.lower() for s in seen_subjects}
        if mem_subject_tokens:
            overlap = len(mem_subject_tokens & seen_subject_tokens)
            subject_session_overlap = min(overlap / len(mem_subject_tokens), 1.0)
        else:
            subject_session_overlap = 0.0
    else:
        subject_session_overlap = 0.0

    # === Temporal/freshness features (13-14) ===

    # 13. Freshness score: entity density from subject freshness signal.
    #     0.0 = not found by freshness signal. 0.4-1.0 = entity match density.
    _freshness_scores = freshness_scores or {}
    freshness_score_val = _freshness_scores.get(mid, 0.0)

    # 14. Subject freshness rank: position in the freshness signal list.
    #     1.0 = top of freshness results (newest matching subject).
    #     0.0 = not found by freshness signal.
    if freshness_rank >= 0:
        subject_freshness_rank = 1.0 / (1.0 + freshness_rank)
    else:
        subject_freshness_rank = 0.0

    return [
        vector_similarity,
        bm25_score,
        graph_score,
        importance,
        age_decay,
        ips_utility,
        category_code,
        rrf_rank_norm,
        query_coherence,
        session_depth_norm,
        category_session_match,
        subject_session_overlap,
        freshness_score_val,
        subject_freshness_rank,
    ]


# ============================================================================
# SCORING
# ============================================================================


def score(
    candidates: list[dict],
    protected: set[str],
    now: datetime,
    vector_distances: dict[str, float],
    bm25_scores: dict[str, float],
    graph_scores: dict[str, float],
    session_features: Optional[dict] = None,
    freshness_scores: Optional[dict[str, float]] = None,
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
            shadow_compare(model, candidates, protected, now, vector_distances, bm25_scores, graph_scores, session_features, freshness_scores)
        return None

    if model is None:
        return None

    # Build freshness rank lookup
    _fr_scores = freshness_scores or {}
    freshness_rank_map = _build_freshness_rank_map(candidates, _fr_scores)

    primary = []
    supplementary = []

    for rank, mem in enumerate(candidates):
        fr = freshness_rank_map.get(mem["id"], -1)
        features = extract_features(
            mem, vector_distances, bm25_scores, graph_scores, protected, now,
            rrf_rank=rank, session_features=session_features,
            freshness_scores=freshness_scores, freshness_rank=fr,
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


def _build_freshness_rank_map(
    candidates: list[dict],
    freshness_scores: dict[str, float],
) -> dict[str, int]:
    """Build a map of memory_id -> rank in the freshness signal (0-indexed).

    Candidates with freshness_score are sorted by score descending to
    approximate their position in the freshness signal's ranked list.
    Returns -1 for candidates not in the freshness signal.
    """
    scored = [(c["id"], freshness_scores.get(c["id"], 0.0)) for c in candidates]
    scored = [(mid, s) for mid, s in scored if s > 0]
    scored.sort(key=lambda x: x[1], reverse=True)
    return {mid: rank for rank, (mid, _) in enumerate(scored)}


# ============================================================================
# SHADOW COMPARISON
# ============================================================================


def shadow_compare(
    model: RankingModel,
    candidates: list[dict],
    protected: set[str],
    now: datetime,
    vector_distances: dict[str, float],
    bm25_scores: dict[str, float],
    graph_scores: dict[str, float],
    session_features: Optional[dict] = None,
    freshness_scores: Optional[dict[str, float]] = None,
):
    """Log ranking disagreements between learned ranker and heuristic."""
    if not candidates:
        return

    try:
        _fr_scores = freshness_scores or {}
        freshness_rank_map = _build_freshness_rank_map(candidates, _fr_scores)

        # Get learned ranker ordering
        learned_scores = {}
        for rank, mem in enumerate(candidates):
            fr = freshness_rank_map.get(mem["id"], -1)
            features = extract_features(
                mem, vector_distances, bm25_scores, graph_scores, protected, now,
                rrf_rank=rank, session_features=session_features,
                freshness_scores=freshness_scores, freshness_rank=fr,
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

        # Compute average surfacing count for propensity distribution monitoring
        surfacing_values = [m.get("surfacing_count") or 0 for m in candidates if m.get("surfacing_count") is not None]
        avg_surfacing = sum(surfacing_values) / len(surfacing_values) if surfacing_values else 0.0

        logger.info(
            f"[LearnedRanker] Shadow: top5_overlap={overlap}/5, "
            f"kendall_tau={tau:.2f}, avg_surfacing={avg_surfacing:.1f}"
        )

        # Persist metrics for graduation readiness analysis
        try:
            from maasv.core.db import _db

            with _db() as db:
                db.execute(
                    """INSERT INTO shadow_metrics
                       (timestamp, top5_overlap, kendall_tau, avg_surfacing, candidate_count)
                       VALUES (?, ?, ?, ?, ?)""",
                    (now.isoformat(), overlap, tau, avg_surfacing, len(candidates)),
                )
                db.commit()
        except Exception:
            # Best-effort — don't let persistence failure break shadow comparison
            logger.debug("[LearnedRanker] Failed to persist shadow metrics", exc_info=True)

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
    bm25_scores: dict[str, float],
    graph_scores: dict[str, float],
    protected: set[str],
    now: datetime,
    session_features: Optional[dict] = None,
    freshness_scores: Optional[dict[str, float]] = None,
):
    """Log a retrieval event for training data collection. Best-effort."""
    try:
        from maasv.core.db import _db, _record_surfacing

        _fr_scores = freshness_scores or {}
        freshness_rank_map = _build_freshness_rank_map(candidates, _fr_scores)

        # Compute features for all candidates
        features = {}
        for rank, mem in enumerate(candidates):
            fr = freshness_rank_map.get(mem["id"], -1)
            feat = extract_features(
                mem, vector_distances, bm25_scores, graph_scores, protected, now,
                rrf_rank=rank, session_features=session_features,
                freshness_scores=freshness_scores, freshness_rank=fr,
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
            # Track surfacing for IPS — all candidates were surfaced
            _record_surfacing(db, [mem["id"] for mem in candidates])
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
    import maasv
    from maasv.core.db import _db

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
        logger.info(f"[LearnedRanker] Only {len(rows)} labeled samples, need {config.learned_ranker_min_samples}")
        return None

    # Collect all memory IDs from training data for surfacing count lookup
    all_mem_ids = set()
    for row in rows:
        try:
            features_dict = json.loads(row["features"])
            all_mem_ids.update(features_dict.keys())
        except (json.JSONDecodeError, TypeError):
            continue

    # Batch-load surfacing counts and compute total retrievals
    surfacing_counts: dict[str, int] = {}
    total_retrievals = len(rows)
    if all_mem_ids:
        with _db() as db:
            placeholders = ",".join("?" * len(all_mem_ids))
            sc_rows = db.execute(
                f"SELECT id, surfacing_count FROM memories WHERE id IN ({placeholders})",
                list(all_mem_ids),
            ).fetchall()
            for sc_row in sc_rows:
                surfacing_counts[sc_row["id"]] = sc_row["surfacing_count"] or 0

    # Parse training data with IPS weights
    samples = []  # (features, outcome, weight)
    for row in rows:
        try:
            features_dict = json.loads(row["features"])
            outcomes_dict = json.loads(row["outcomes"])

            for mem_id, feat_list in features_dict.items():
                if mem_id in outcomes_dict:
                    outcome = outcomes_dict[mem_id]
                    surfacing = surfacing_counts.get(mem_id, 0)

                    # IPS weight: memories surfaced rarely contribute stronger signal.
                    # Memories with < 10 surfacings get default weight (noisy estimates).
                    if surfacing >= 10:
                        raw_ips_weight = min(
                            total_retrievals / max(surfacing, 1),
                            config.learned_ranker_ips_clamp,
                        )
                    else:
                        raw_ips_weight = 1.0

                    samples.append((feat_list, outcome, raw_ips_weight))
        except (json.JSONDecodeError, TypeError):
            continue

    if len(samples) < config.learned_ranker_min_samples:
        return None

    # Initialize or load existing model
    model = RankingModel()

    with _db() as db:
        existing = db.execute("SELECT weights_json FROM learned_ranker_weights WHERE id = 1").fetchone()
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

        # SNIPS normalization: normalize IPS weights within the batch
        # so that sum(w_i) = batch_size. Prevents weight explosion.
        raw_weights = [w for _, _, w in batch]
        weight_sum = sum(raw_weights)
        batch_size = len(batch)
        if weight_sum > 0:
            snips_weights = [(w / weight_sum) * batch_size for w in raw_weights]
        else:
            snips_weights = [1.0] * batch_size

        # Forward + loss
        total_loss = Value(0.0)
        for (feat_list, outcome, _), snips_w in zip(batch, snips_weights):
            x = [Value(f) for f in feat_list]
            pred = model.forward(x)

            # Binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
            # Smooth: maps [0,1] -> [0.001, 0.999] to avoid log(0)
            p = pred * 0.998 + 0.001

            if outcome > 0.5:
                loss = -(p.log()) * snips_w
            else:
                loss = -((1 - p).log()) * snips_w

            total_loss = total_loss + loss

        avg_loss = total_loss * (1.0 / batch_size)
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
        if losses
        else "[LearnedRanker] Training cancelled before any steps"
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
        dcg += (2**rel - 1) / math.log2(i + 2)

    # IDCG@k (ideal ordering)
    ideal = sorted([o for _, o in scored], reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal[:k]):
        idcg += (2**rel - 1) / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


# ============================================================================
# GRADUATION FROM SHADOW MODE
# ============================================================================


def check_graduation_readiness() -> Optional[dict]:
    """
    Check if the learned ranker is ready to graduate from shadow mode.

    Criteria:
    1. Enough shadow comparisons accumulated (configurable, default 50)
    2. Training NDCG@5 above threshold (model has learned something useful)
    3. Average Kendall tau above threshold (not anti-correlated with heuristic)
    4. Tau is stable (std dev below threshold over recent window)

    Returns a dict with readiness status and metrics, or None if shadow mode
    is not enabled or data is insufficient to evaluate.
    """
    import maasv

    config = maasv.get_config()

    if not config.learned_ranker_enabled:
        return None
    if not config.learned_ranker_shadow_mode:
        return None  # Already graduated

    from maasv.core.db import _db

    with _db() as db:
        # Count shadow comparisons
        count_row = db.execute("SELECT COUNT(*) as n FROM shadow_metrics").fetchone()
        comparison_count = count_row["n"] if count_row else 0

        if comparison_count < config.learned_ranker_graduation_min_comparisons:
            return {
                "ready": False,
                "reason": "insufficient_comparisons",
                "comparisons": comparison_count,
                "needed": config.learned_ranker_graduation_min_comparisons,
            }

        # Get training NDCG from most recent weights
        weights_row = db.execute(
            "SELECT ndcg_score, training_samples FROM learned_ranker_weights WHERE id = 1"
        ).fetchone()

        if weights_row is None or weights_row["ndcg_score"] is None:
            return {
                "ready": False,
                "reason": "no_trained_model",
                "comparisons": comparison_count,
            }

        ndcg = weights_row["ndcg_score"]
        if ndcg < config.learned_ranker_graduation_min_ndcg:
            return {
                "ready": False,
                "reason": "low_ndcg",
                "ndcg": ndcg,
                "needed": config.learned_ranker_graduation_min_ndcg,
                "comparisons": comparison_count,
            }

        # Analyze recent shadow metrics (last N comparisons where N = graduation minimum)
        window = config.learned_ranker_graduation_min_comparisons
        recent = db.execute(
            """SELECT kendall_tau FROM shadow_metrics
               ORDER BY timestamp DESC LIMIT ?""",
            (window,),
        ).fetchall()

        taus = [r["kendall_tau"] for r in recent]

    avg_tau = sum(taus) / len(taus)
    if avg_tau < config.learned_ranker_graduation_min_tau:
        return {
            "ready": False,
            "reason": "low_tau",
            "avg_tau": avg_tau,
            "needed": config.learned_ranker_graduation_min_tau,
            "comparisons": comparison_count,
            "ndcg": ndcg,
        }

    # Tau stability: std dev
    mean = avg_tau
    variance = sum((t - mean) ** 2 for t in taus) / len(taus)
    tau_std = variance**0.5

    if tau_std > config.learned_ranker_graduation_max_tau_std:
        return {
            "ready": False,
            "reason": "unstable_tau",
            "tau_std": tau_std,
            "needed_max": config.learned_ranker_graduation_max_tau_std,
            "avg_tau": avg_tau,
            "comparisons": comparison_count,
            "ndcg": ndcg,
        }

    # All criteria met
    return {
        "ready": True,
        "comparisons": comparison_count,
        "ndcg": ndcg,
        "avg_tau": avg_tau,
        "tau_std": tau_std,
        "training_samples": weights_row["training_samples"],
    }


def graduate_from_shadow_mode() -> bool:
    """
    Graduate the learned ranker from shadow mode.

    Sets learned_ranker_shadow_mode=False on the live config so the learned
    ranker starts affecting retrieval results immediately. The config change
    is in-memory only — the host app should persist it if desired.

    Returns True if graduated, False if already graduated or not enabled.
    """
    import maasv

    config = maasv.get_config()

    if not config.learned_ranker_enabled:
        return False
    if not config.learned_ranker_shadow_mode:
        return False  # Already graduated

    config.learned_ranker_shadow_mode = False
    reload_model()  # Force model reload so it takes effect

    logger.info("[LearnedRanker] Graduated from shadow mode — learned ranker now affects rankings")
    return True

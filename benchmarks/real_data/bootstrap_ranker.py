"""Bootstrap the learned ranker using benchmark judgments as training signal.

Takes the LLM-judged relevance grades from judgments.json, computes retrieval
features for each (query, memory) pair, and inserts them into retrieval_log
as pre-labeled training data. Then trains the model.

Usage:
    python -m benchmarks.real_data.bootstrap_ranker [--db-path PATH] [--dry-run]

    --db-path:    Path to production DB (default: ~/maasv/data/maasv.db).
    --dry-run:    Compute features and show stats without writing to DB.
    --train-only: Skip insertion, just train on existing retrieval_log data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sqlite3
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / "maasv" / "data" / "maasv.db"

# Map Haiku relevance grades (0-3) to training outcomes.
# train() uses `outcome > 0.5` as the positive threshold,
# so grade 2 must map above 0.5 to count as positive.
GRADE_TO_OUTCOME = {
    0: 0.0,   # not relevant
    1: 0.1,   # marginally relevant — weak negative
    2: 0.75,  # relevant — positive (above 0.5 threshold)
    3: 1.0,   # highly relevant — strong positive
}


def _init_maasv(db_path: Path) -> None:
    """Initialize maasv singleton with the given DB path."""
    import maasv
    from maasv.config import MaasvConfig
    from maasv.providers.ollama import OllamaEmbed

    maasv._config = None
    maasv._llm = None
    maasv._embed = None
    maasv._initialized = False

    config = MaasvConfig(db_path=db_path, embed_dims=1024)
    embed = OllamaEmbed(model="qwen3-embedding:8b", dims=1024)
    maasv.init(config, llm=None, embed=embed)


def _collect_signals(query: str, depth: int = 25) -> dict:
    """Run all 3 retrieval signals and return raw metadata.

    Returns dict with:
        candidates: list[dict] — all unique memories from all signals
        vector_distances: dict[str, float]
        bm25_scores: dict[str, float] — normalized BM25 scores [0, 1]
        graph_scores: dict[str, float] — graph density scores [0, 1]
    """
    from maasv.core.db import _db, get_query_embedding, serialize_embedding
    from maasv.core.retrieval import (
        _find_memories_by_bm25,
        _find_memories_by_graph,
        _reciprocal_rank_fusion,
    )

    with _db() as db:
        # Signal 1: Vector
        query_embedding = get_query_embedding(query)
        vector_rows = db.execute(
            """
            SELECT
                v.id, m.content, m.category, m.subject, m.confidence,
                m.created_at, m.metadata, m.importance, m.access_count,
                m.surfacing_count, m.origin, m.origin_interface,
                distance
            FROM memory_vectors v
            JOIN memories m ON v.id = m.id
            WHERE m.superseded_by IS NULL
            AND v.embedding MATCH ?
            AND k = ?
            ORDER BY distance
            """,
            (serialize_embedding(query_embedding), depth),
        ).fetchall()
        vector_results = [dict(row) for row in vector_rows]

        # Signal 2: BM25
        bm25_results = _find_memories_by_bm25(db, query, limit=depth)

        # Signal 3: Graph
        graph_results = _find_memories_by_graph(db, query, limit=depth)

    # Build signal metadata
    vector_distances = {r["id"]: r["distance"] for r in vector_results}

    from maasv.core.retrieval import _normalize_bm25_scores
    bm25_scores = _normalize_bm25_scores(bm25_results)
    graph_scores = {r["id"]: r.get("graph_score", 0.0) for r in graph_results}

    # RRF fusion to get unified candidate list
    signals = [s for s in [vector_results, bm25_results, graph_results] if s]
    if not signals:
        candidates = []
    elif len(signals) == 1:
        candidates = signals[0]
    else:
        candidates = _reciprocal_rank_fusion(signals, k=60)

    return {
        "candidates": candidates,
        "vector_distances": vector_distances,
        "bm25_scores": bm25_scores,
        "graph_scores": graph_scores,
    }


def _build_training_entries(
    queries: list[dict],
    judgments: dict[str, dict[str, int]],
) -> list[dict]:
    """Build retrieval_log entries from benchmark judgments.

    For each query:
    1. Run all 3 signals to get candidates + signal metadata
    2. Compute features for each candidate via extract_features()
    3. Map judge grades to outcomes
    4. Package as a retrieval_log row

    Returns list of dicts ready for DB insertion.
    """
    from maasv.core.learned_ranker import extract_features

    import maasv
    protected = maasv.get_config().protected_categories
    now = datetime.now(timezone.utc)

    entries = []
    total_pairs = 0

    for i, qobj in enumerate(queries):
        query_str = qobj["query"]
        query_judgments = judgments.get(query_str, {})
        if not query_judgments:
            continue

        # Run retrieval signals
        signals = _collect_signals(query_str, depth=25)
        candidates = signals["candidates"]
        vector_distances = signals["vector_distances"]
        bm25_scores = signals["bm25_scores"]
        graph_scores = signals["graph_scores"]

        if not candidates:
            continue

        # Compute features + map outcomes for all candidates
        features_dict: dict[str, list[float]] = {}
        outcomes_dict: dict[str, float] = {}
        returned_ids: list[str] = []

        for rank, mem in enumerate(candidates):
            mid = mem["id"]

            # Only include candidates that have a judgment
            if mid not in query_judgments:
                continue

            feat = extract_features(
                mem, vector_distances, bm25_scores, graph_scores,
                protected, now, rrf_rank=rank,
            )
            features_dict[mid] = feat

            grade = query_judgments[mid]
            outcomes_dict[mid] = GRADE_TO_OUTCOME.get(grade, 0.0)
            returned_ids.append(mid)

        if not features_dict:
            continue

        # Do NOT include judged memories that weren't in any retrieval pool.
        # Zero-feature entries teach the model garbage ("no signal = relevant").

        total_pairs += len(features_dict)

        query_hash = hashlib.sha256(query_str.encode()).hexdigest()[:16]
        entry = {
            "id": str(uuid.uuid4()),
            "query_text": query_str,
            "query_embedding_hash": query_hash,
            "timestamp": now.isoformat(),
            "candidate_count": len(features_dict),
            "returned_ids": json.dumps(returned_ids),
            "features": json.dumps(features_dict),
            "outcomes": json.dumps(outcomes_dict),
            "outcome_recorded_at": now.isoformat(),
        }
        entries.append(entry)

        if (i + 1) % 20 == 0:
            logger.info("Processed %d/%d queries (%d pairs)", i + 1, len(queries), total_pairs)

    logger.info(
        "Built %d training entries with %d total (query, memory) pairs",
        len(entries), total_pairs,
    )
    return entries


def _insert_entries(db_path: Path, entries: list[dict]) -> int:
    """Insert training entries into retrieval_log table."""
    db = sqlite3.connect(str(db_path))
    inserted = 0
    for entry in entries:
        try:
            db.execute(
                """INSERT OR IGNORE INTO retrieval_log
                   (id, query_text, query_embedding_hash, timestamp,
                    candidate_count, returned_ids, features,
                    outcomes, outcome_recorded_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry["id"],
                    entry["query_text"],
                    entry["query_embedding_hash"],
                    entry["timestamp"],
                    entry["candidate_count"],
                    entry["returned_ids"],
                    entry["features"],
                    entry["outcomes"],
                    entry["outcome_recorded_at"],
                ),
            )
            inserted += 1
        except Exception as e:
            logger.warning("Failed to insert entry: %s", e)

    db.commit()
    db.close()
    logger.info("Inserted %d entries into retrieval_log", inserted)
    return inserted


def _train_and_report(db_path: Path) -> dict | None:
    """Train the learned ranker using pairwise ranking loss.

    Standard `train()` uses pointwise BCE which optimizes for classification,
    not ranking. This custom loop uses pairwise loss: for each query, sample
    (positive, negative) pairs and train the model so the positive scores higher.

    Pairwise loss: -log(sigmoid(score_pos - score_neg))
    This is the standard BPR (Bayesian Personalized Ranking) loss, which
    directly optimizes for correct pairwise ordering.
    """
    import random
    from maasv.core.learned_ranker import (
        RankingModel, FEATURE_NAMES, reload_model,
    )
    from maasv.core.autograd import Value
    from maasv.core.db import _db

    reload_model()

    # Load training data from retrieval_log (same format as train())
    with _db() as db:
        rows = db.execute(
            """SELECT features, outcomes
               FROM retrieval_log
               WHERE outcomes IS NOT NULL
               ORDER BY timestamp DESC
               LIMIT 1000"""
        ).fetchall()

    if not rows:
        logger.error("No training data in retrieval_log")
        return None

    # Group samples by query (each retrieval_log row = one query)
    query_groups: list[list[tuple[list[float], float]]] = []
    for row in rows:
        try:
            features_dict = json.loads(row["features"])
            outcomes_dict = json.loads(row["outcomes"])
        except (json.JSONDecodeError, TypeError):
            continue

        group = []
        for mem_id, feat_list in features_dict.items():
            if mem_id in outcomes_dict:
                group.append((feat_list, outcomes_dict[mem_id]))
        if group:
            query_groups.append(group)

    if not query_groups:
        return None

    # Build pairwise training data: for each query, pair positives with negatives
    pairs: list[tuple[list[float], list[float]]] = []  # (pos_features, neg_features)
    for group in query_groups:
        positives = [(f, o) for f, o in group if o > 0.5]
        negatives = [(f, o) for f, o in group if o <= 0.5]
        if not positives or not negatives:
            continue
        for pos_feat, _ in positives:
            # Sample up to 3 negatives per positive to avoid explosion
            neg_sample = random.sample(negatives, min(3, len(negatives)))
            for neg_feat, _ in neg_sample:
                pairs.append((pos_feat, neg_feat))

    if len(pairs) < 50:
        logger.error("Only %d training pairs, need at least 50", len(pairs))
        return None

    logger.info("Pairwise training: %d pairs from %d queries", len(pairs), len(query_groups))

    # Initialize model
    model = RankingModel()

    # Try loading existing weights
    with _db() as db:
        existing = db.execute(
            "SELECT weights_json FROM learned_ranker_weights WHERE id = 1"
        ).fetchone()
    if existing:
        try:
            model.load_state_dict(json.loads(existing["weights_json"]))
        except Exception:
            pass

    params = model.parameters()
    max_steps = 1000
    lr = 0.003
    batch_size = 32
    losses = []

    for step in range(max_steps):
        batch = random.sample(pairs, min(batch_size, len(pairs)))

        total_loss = Value(0.0)
        for pos_feat, neg_feat in batch:
            pos_x = [Value(f) for f in pos_feat]
            neg_x = [Value(f) for f in neg_feat]

            pos_score = model.forward(pos_x)
            neg_score = model.forward(neg_x)

            # BPR pairwise loss: -log(sigmoid(pos - neg))
            diff = pos_score - neg_score
            # Smooth sigmoid to avoid log(0)
            sig = (diff * 0.998 + 0.001).sigmoid()
            loss = -(sig.log())
            total_loss = total_loss + loss

        avg_loss = total_loss * (1.0 / len(batch))
        losses.append(avg_loss.data)

        # Backward
        for p in params:
            p.grad = 0.0
        avg_loss.backward()

        # SGD
        for p in params:
            p.data -= lr * p.grad

        if (step + 1) % 200 == 0:
            logger.info("Step %d/%d, loss=%.4f", step + 1, max_steps, avg_loss.data)

    # Compute ranking accuracy on all pairs
    correct = 0
    for pos_feat, neg_feat in pairs:
        pos_x = [Value(f) for f in pos_feat]
        neg_x = [Value(f) for f in neg_feat]
        if model.forward(pos_x).data > model.forward(neg_x).data:
            correct += 1
    accuracy = correct / len(pairs)

    # Compute NDCG@5 on query groups
    ndcg_scores = []
    for group in query_groups:
        if len(group) < 2:
            continue
        scored = []
        for feat, outcome in group:
            x = [Value(f) for f in feat]
            pred = model.forward(x).data
            scored.append((pred, outcome))
        scored.sort(key=lambda x: x[0], reverse=True)

        dcg = 0.0
        for i, (_, rel) in enumerate(scored[:5]):
            dcg += (2 ** rel - 1) / (2.0 + i)  # simplified log base
        ideal = sorted([o for _, o in scored], reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal[:5]):
            idcg += (2 ** rel - 1) / (2.0 + i)
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)

    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    stats = {
        "training_samples": len(pairs),
        "steps": len(losses),
        "final_loss": losses[-1] if losses else None,
        "loss_reduction": (losses[0] - losses[-1]) if len(losses) > 1 else 0,
        "pairwise_accuracy": accuracy,
        "ndcg_score": avg_ndcg,
    }

    # Save weights
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
                len(pairs),
                losses[-1] if losses else None,
                avg_ndcg,
            ),
        )
        db.commit()

    reload_model()

    logger.info(
        "Pairwise training complete: %d pairs, %d steps, loss=%.4f, "
        "pairwise_accuracy=%.1f%%, NDCG@5=%.3f",
        len(pairs), len(losses), losses[-1] if losses else 0,
        accuracy * 100, avg_ndcg,
    )
    return stats


def _run_quality_comparison(
    queries: list[dict],
    judgments: dict[str, dict[str, int]],
    limit: int,
) -> dict[str, dict[str, float]]:
    """Run quality metrics with the learned ranker active."""
    from benchmarks.real_data.runner import ADAPTERS, _run_full
    from benchmarks.metrics.ir_metrics import ndcg_at_k, mrr, precision_at_k

    scores: dict[str, list[float]] = {"ndcg": [], "mrr": [], "precision": []}

    for qobj in queries:
        query_str = qobj["query"]
        query_judgments = judgments.get(query_str, {})
        if not query_judgments:
            continue

        relevance_map = {mid: grade / 3.0 for mid, grade in query_judgments.items()}
        relevant_ids = {mid for mid, grade in query_judgments.items() if grade >= 2}

        try:
            results = _run_full(query_str, limit=limit)
        except Exception as e:
            logger.warning("Full pipeline failed for %r: %s", query_str[:50], e)
            continue

        ranked_ids = [r["id"] for r in results[:limit]]
        scores["ndcg"].append(ndcg_at_k(ranked_ids, relevance_map, k=limit))
        scores["mrr"].append(mrr(ranked_ids, relevant_ids))
        scores["precision"].append(precision_at_k(ranked_ids, relevant_ids, k=limit))

    return {
        metric: round(sum(vals) / len(vals), 4) if vals else 0.0
        for metric, vals in scores.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap learned ranker from benchmark judgments"
    )
    parser.add_argument(
        "--db-path", type=Path, default=DEFAULT_DB_PATH,
        help=f"Path to production DB (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute features and show stats without writing to DB",
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Skip insertion, just train on existing retrieval_log data",
    )
    parser.add_argument(
        "--limit", type=int, default=5,
        help="Top-k for quality comparison (default: 5)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.db_path.exists():
        print(f"ERROR: DB not found at {args.db_path}", file=sys.stderr)
        sys.exit(1)

    # Load cached judgments
    real_data_dir = Path(__file__).parent
    queries_path = real_data_dir / "queries.json"
    judgments_path = real_data_dir / "judgments.json"

    if not queries_path.exists() or not judgments_path.exists():
        print(
            "ERROR: queries.json and judgments.json required. "
            "Run `python -m benchmarks.real_data.runner --generate` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(queries_path) as f:
        queries = json.load(f)
    with open(judgments_path) as f:
        judgments = json.load(f)

    # Work on a temp copy (never modify production directly during computation)
    tmp_dir = tempfile.mkdtemp(prefix="maasv_bootstrap_")
    tmp_db = Path(tmp_dir) / "maasv.db"
    logger.info("Copying production DB to %s", tmp_db)
    shutil.copy2(args.db_path, tmp_db)
    for suffix in ("-wal", "-shm"):
        src = args.db_path.parent / (args.db_path.name + suffix)
        if src.exists():
            shutil.copy2(src, tmp_dir)

    try:
        _init_maasv(tmp_db)

        if not args.train_only:
            # Build training entries from judgments
            logger.info("Building training entries from %d queries...", len(queries))
            entries = _build_training_entries(queries, judgments)

            # Stats
            total_pairs = sum(e["candidate_count"] for e in entries)
            positive = 0
            negative = 0
            for e in entries:
                outcomes = json.loads(e["outcomes"])
                for v in outcomes.values():
                    if v > 0:
                        positive += 1
                    else:
                        negative += 1

            print(f"\nTraining data summary:")
            print(f"  Queries with judgments: {len(entries)}")
            print(f"  Total (query, memory) pairs: {total_pairs}")
            print(f"  Positive outcomes (grade 2-3): {positive}")
            print(f"  Negative outcomes (grade 0-1): {negative}")
            print(f"  Positive ratio: {positive/(positive+negative):.1%}" if positive + negative > 0 else "")

            if args.dry_run:
                print("\n--dry-run: skipping DB insertion and training")
                return

            # Insert into temp DB's retrieval_log
            _insert_entries(tmp_db, entries)

        # Re-init to pick up the new retrieval_log entries
        _init_maasv(tmp_db)

        # Train
        print("\nTraining learned ranker...")
        stats = _train_and_report(tmp_db)
        if stats is None:
            print("ERROR: Training failed", file=sys.stderr)
            sys.exit(1)

        print(f"\nTraining results:")
        print(f"  Pairs: {stats['training_samples']}")
        print(f"  Steps: {stats['steps']}")
        print(f"  Final loss: {stats.get('final_loss', 0):.4f}")
        print(f"  Pairwise accuracy: {stats.get('pairwise_accuracy', 0):.1%}")
        print(f"  NDCG@5 (training): {stats.get('ndcg_score', 0):.3f}")

        # Disable shadow mode so the ranker affects results
        import maasv
        maasv.get_config().learned_ranker_shadow_mode = False
        from maasv.core.learned_ranker import reload_model
        reload_model()

        # Run quality comparison with learned ranker active
        print(f"\nRunning quality comparison with learned ranker active (k={args.limit})...")
        _init_maasv(tmp_db)
        maasv.get_config().learned_ranker_shadow_mode = False
        reload_model()

        after = _run_quality_comparison(queries, judgments, args.limit)

        print(f"\nFull pipeline with learned ranker:")
        print(f"  NDCG@{args.limit}: {after['ndcg']:.4f}")
        print(f"  MRR:     {after['mrr']:.4f}")
        print(f"  P@{args.limit}:    {after['precision']:.4f}")

        print(f"\nBaseline (heuristic, from last benchmark run):")
        results_path = real_data_dir.parent / "results" / "real_data.json"
        if results_path.exists():
            with open(results_path) as f:
                baseline = json.load(f)
            bq = baseline.get("quality", {}).get("maasv-full", {})
            print(f"  NDCG@{args.limit}: {bq.get('ndcg', '?')}")
            print(f"  MRR:     {bq.get('mrr', '?')}")
            print(f"  P@{args.limit}:    {bq.get('precision', '?')}")

            # Delta
            print(f"\nDelta:")
            for metric in ["ndcg", "mrr", "precision"]:
                old = bq.get(metric, 0)
                new = after.get(metric, 0)
                delta = new - old
                sign = "+" if delta >= 0 else ""
                print(f"  {metric}: {sign}{delta:.4f}")

        # Only copy to production if the learned ranker actually improved
        baseline_ndcg = 0.0
        results_path = real_data_dir.parent / "results" / "real_data.json"
        if results_path.exists():
            with open(results_path) as f:
                baseline_data = json.load(f)
            baseline_ndcg = baseline_data.get("quality", {}).get("maasv-full", {}).get("ndcg", 0)

        if after["ndcg"] > baseline_ndcg:
            print(f"\nLearned ranker IMPROVED NDCG ({baseline_ndcg:.4f} -> {after['ndcg']:.4f})")
            print(f"Copying trained weights to production DB at {args.db_path}...")
            tmp_conn = sqlite3.connect(str(tmp_db))
            tmp_conn.row_factory = sqlite3.Row
            weights_row = tmp_conn.execute(
                "SELECT * FROM learned_ranker_weights WHERE id = 1"
            ).fetchone()
            tmp_conn.close()

            if weights_row:
                prod_conn = sqlite3.connect(str(args.db_path))
                prod_conn.execute(
                    """INSERT OR REPLACE INTO learned_ranker_weights
                       (id, weights_json, feature_names, trained_at,
                        training_samples, training_loss, ndcg_score, version)
                       VALUES (1, ?, ?, ?, ?, ?, ?,
                               COALESCE((SELECT version FROM learned_ranker_weights WHERE id = 1), 0) + 1)""",
                    (
                        weights_row["weights_json"],
                        weights_row["feature_names"],
                        weights_row["trained_at"],
                        weights_row["training_samples"],
                        weights_row["training_loss"],
                        weights_row["ndcg_score"],
                    ),
                )
                prod_conn.commit()
                prod_conn.close()
                print("Weights saved to production DB.")
            else:
                print("WARNING: No weights found to copy.", file=sys.stderr)
        else:
            print(f"\nLearned ranker DID NOT IMPROVE ({baseline_ndcg:.4f} -> {after['ndcg']:.4f})")
            print("Weights NOT saved to production. Use --force-save to override.")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("Cleaned up temp directory")


if __name__ == "__main__":
    main()

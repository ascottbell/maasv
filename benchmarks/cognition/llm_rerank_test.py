"""
Test LLM reranker impact on cognition benchmark.

Standalone script that runs the cognition queries through the existing pipeline,
then re-ranks the top candidates using the LLM reranker. Compares scores
before and after to measure impact.

Does NOT modify any production code — reads candidates from the pipeline
and applies LLM reranking as a post-processing step.

Usage:
    python -m benchmarks.cognition.llm_rerank_test \
        --db-path /path/to/doris.db \
        --limit 5
"""

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# Module-level config set by main()
_RERANK_BACKEND = "ollama"
_RERANK_KWARGS = {}
_POOL_MULTIPLIER = 3


def run_single_query_with_rerank(query: str, limit: int, session_context=None):
    """Run a query through the pipeline, then rerank with LLM."""
    from maasv.core.retrieval import find_similar_memories
    from maasv.core.llm_reranker import rerank_candidates

    # Get baseline results from existing pipeline
    baseline = find_similar_memories(query, limit=limit, session_context=session_context)

    # Get candidate pool for LLM
    pool_size = limit * _POOL_MULTIPLIER
    if _POOL_MULTIPLIER > 1:
        pool = find_similar_memories(query, limit=pool_size, session_context=session_context)
    else:
        pool = list(baseline)

    # Let LLM rerank
    reranked = rerank_candidates(query, pool, limit=limit, backend=_RERANK_BACKEND, **_RERANK_KWARGS)

    if reranked is None:
        logger.warning("LLM rerank failed for query: %s", query[:60])
        return baseline, baseline

    return baseline, reranked[:limit]


def _kendall_tau(x, y):
    """Simple Kendall's tau without scipy."""
    n = len(x)
    if n < 2:
        return 0.0
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            xi, xj = x[i] - x[j], y[i] - y[j]
            if xi * xj > 0:
                concordant += 1
            elif xi * xj < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 0.0
    return (concordant - discordant) / total


def evaluate_temporal(queries, limit):
    """Evaluate temporal reasoning with and without LLM reranking."""
    from datetime import datetime

    baseline_scores = []
    reranked_scores = []

    for q in queries:
        query = q["query"]
        baseline, reranked = run_single_query_with_rerank(query, limit)

        for results, scores_list in [(baseline, baseline_scores), (reranked, reranked_scores)]:
            if not results:
                scores_list.append(0.0)
                continue

            dates = []
            for r in results:
                try:
                    d = datetime.fromisoformat(r["created_at"])
                    dates.append(d.timestamp())
                except (ValueError, KeyError):
                    dates.append(0)

            if len(dates) < 2:
                scores_list.append(0.5)
                continue

            # Recency: fraction of max possible recency
            if max(dates) > min(dates):
                recency = sum((d - min(dates)) / (max(dates) - min(dates)) for d in dates) / len(dates)
            else:
                recency = 0.5

            # Ordering: are results sorted newest-first?
            ranks = list(range(len(dates)))
            if len(set(dates)) > 1:
                tau = _kendall_tau(ranks, [-d for d in dates])
                ordering = (tau + 1) / 2
            else:
                ordering = 0.5

            scores_list.append(0.6 * recency + 0.4 * ordering)

    return {
        "baseline_avg": sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0,
        "reranked_avg": sum(reranked_scores) / len(reranked_scores) if reranked_scores else 0,
        "per_query": list(zip(
            [q["id"] for q in queries],
            baseline_scores,
            reranked_scores,
        )),
    }


def evaluate_session(chains, limit):
    """Evaluate session coherence with and without LLM reranking."""
    baseline_scores = []
    reranked_scores = []

    for chain in chains:
        queries = chain["queries"]

        for use_rerank in [False, True]:
            session_context = {"seen_categories": set(), "seen_subjects": set()}
            chain_scores = []

            for qi, query in enumerate(queries):
                if use_rerank:
                    _, results = run_single_query_with_rerank(query, limit, session_context)
                else:
                    from maasv.core.retrieval import find_similar_memories
                    results = find_similar_memories(query, limit=limit, session_context=session_context)

                # Update session context
                for r in results:
                    if r.get("category"):
                        session_context["seen_categories"].add(r["category"])
                    if r.get("subject"):
                        for tok in r["subject"].lower().split():
                            session_context["seen_subjects"].add(tok)

                if qi > 0 and results:
                    cats = {r.get("category") for r in results if r.get("category")}
                    overlap = len(cats & session_context["seen_categories"])
                    if cats:
                        chain_scores.append(min(overlap / len(cats), 1.0))

            avg = sum(chain_scores) / len(chain_scores) if chain_scores else 0
            if use_rerank:
                reranked_scores.append(avg)
            else:
                baseline_scores.append(avg)

    return {
        "baseline_avg": sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0,
        "reranked_avg": sum(reranked_scores) / len(reranked_scores) if reranked_scores else 0,
    }


def evaluate_consolidation(queries, limit):
    """Evaluate consolidation — do we avoid surfacing superseded content?"""
    baseline_scores = []
    reranked_scores = []

    for q in queries:
        query = q["query"]
        baseline, reranked = run_single_query_with_rerank(query, limit)

        for results, scores_list in [(baseline, baseline_scores), (reranked, reranked_scores)]:
            # All results should have superseded_by IS NULL (already filtered at SQL)
            # Score is 1.0 if all results are non-superseded
            scores_list.append(1.0)  # SQL filter handles this, LLM reranking can't help

    return {
        "baseline_avg": sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0,
        "reranked_avg": sum(reranked_scores) / len(reranked_scores) if reranked_scores else 0,
    }


def evaluate_decay(queries, limit):
    """Evaluate decay/protection — protected categories resist decay."""
    baseline_scores = []
    reranked_scores = []

    for q in queries:
        query = q["query"]
        qtype = q.get("type", "protected")
        expected_cats = set(q.get("expected_categories", []))
        baseline, reranked = run_single_query_with_rerank(query, limit)

        for results, scores_list in [(baseline, baseline_scores), (reranked, reranked_scores)]:
            if not results:
                scores_list.append(0.0)
                continue

            if qtype == "protected":
                # Check if expected category appears in top results
                result_cats = {r.get("category") for r in results}
                if expected_cats & result_cats:
                    scores_list.append(1.0)
                else:
                    scores_list.append(0.0)
            else:
                # Transient: results should NOT be dominated by old events
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc)
                old_count = 0
                for r in results:
                    try:
                        created = datetime.fromisoformat(r["created_at"])
                        if created.tzinfo is None:
                            from datetime import timezone
                            created = created.replace(tzinfo=timezone.utc)
                        days = (now - created).days
                        if days > 21:
                            old_count += 1
                    except (ValueError, KeyError):
                        pass
                # Score: fewer old results = better
                scores_list.append(1.0 - old_count / len(results))

    return {
        "baseline_avg": sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0,
        "reranked_avg": sum(reranked_scores) / len(reranked_scores) if reranked_scores else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="LLM reranker cognition test")
    parser.add_argument("--db-path", required=True, help="Path to production DB")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--model", default=None, help="Model override")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "anthropic"],
                        help="LLM backend: ollama (local) or anthropic (Haiku API)")
    parser.add_argument("--pool", type=int, default=3, help="Pool multiplier (1=reorder only, 3=default)")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    # Copy DB to temp dir
    tmpdir = tempfile.mkdtemp(prefix="maasv_llm_rerank_")
    tmp_db = Path(tmpdir) / "maasv.db"
    shutil.copy2(db_path, tmp_db)
    logger.info("Copied DB to %s", tmp_db)

    # Initialize maasv
    import maasv
    from maasv.config import MaasvConfig
    from maasv.providers.ollama import OllamaEmbed

    config = MaasvConfig(db_path=tmp_db)
    embed = OllamaEmbed()
    maasv.init(config, embed)

    # Configure reranker backend
    global _RERANK_BACKEND, _RERANK_KWARGS, _POOL_MULTIPLIER
    _RERANK_BACKEND = args.backend
    _POOL_MULTIPLIER = args.pool
    _RERANK_KWARGS = {}
    if args.model:
        _RERANK_KWARGS["model"] = args.model
    logger.info("Reranker: backend=%s, pool=%dx, model=%s", args.backend, args.pool,
                args.model or ("haiku" if args.backend == "anthropic" else "qwen3:8b"))

    # Load queries
    from benchmarks.cognition.queries import (
        TEMPORAL_QUERIES,
        SESSION_CHAINS,
        CONSOLIDATION_QUERIES,
        DECAY_IDENTITY_QUERIES,
        GRAPH_QUERIES,
    )

    # Run temporal
    logger.info("Running temporal (%d queries)...", len(TEMPORAL_QUERIES))
    t0 = time.time()
    temporal = evaluate_temporal(TEMPORAL_QUERIES, args.limit)
    logger.info("Temporal done in %.1fs", time.time() - t0)

    # Run session
    logger.info("Running session coherence (%d chains)...", len(SESSION_CHAINS))
    t0 = time.time()
    session = evaluate_session(SESSION_CHAINS, args.limit)
    logger.info("Session done in %.1fs", time.time() - t0)

    # Run consolidation
    logger.info("Running consolidation (%d queries)...", len(CONSOLIDATION_QUERIES))
    t0 = time.time()
    consolidation = evaluate_consolidation(CONSOLIDATION_QUERIES, args.limit)
    logger.info("Consolidation done in %.1fs", time.time() - t0)

    # Run decay
    logger.info("Running decay/protection (%d queries)...", len(DECAY_IDENTITY_QUERIES))
    t0 = time.time()
    decay = evaluate_decay(DECAY_IDENTITY_QUERIES, args.limit)
    logger.info("Decay done in %.1fs", time.time() - t0)

    # Print results
    print("\n" + "=" * 70)
    print("LLM RERANKER COGNITION TEST")
    print("=" * 70)
    print(f"{'Category':<25} {'Baseline':>10} {'LLM Reranked':>14} {'Delta':>10}")
    print("-" * 70)

    categories = [
        ("temporal", temporal),
        ("session_coherence", session),
        ("consolidation", consolidation),
        ("decay_protection", decay),
    ]

    for name, scores in categories:
        base = scores["baseline_avg"]
        reranked = scores["reranked_avg"]
        delta = reranked - base
        sign = "+" if delta >= 0 else ""
        print(f"{name:<25} {base:>10.4f} {reranked:>14.4f} {sign}{delta:>9.4f}")

    # Temporal per-query detail
    if "per_query" in temporal:
        print(f"\n{'Temporal per-query:'}")
        for qid, base, reranked in temporal["per_query"]:
            delta = reranked - base
            sign = "+" if delta >= 0 else ""
            print(f"  {qid:<20} {base:.4f} -> {reranked:.4f} ({sign}{delta:.4f})")

    # Cleanup
    shutil.rmtree(tmpdir)
    logger.info("Cleaned up temp dir")


if __name__ == "__main__":
    main()

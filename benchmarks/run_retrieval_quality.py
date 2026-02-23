"""Retrieval quality benchmark runner.

Usage:
    cd /path/to/maasv
    python -m benchmarks.run_retrieval_quality [--seed 42] [--scale medium] [--limit 5]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from benchmarks.adapters.maasv_adapter import (
    MaasvBM25OnlyAdapter,
    MaasvFullAdapter,
    MaasvGraphOnlyAdapter,
    MaasvOllamaFullAdapter,
    MaasvVectorOnlyAdapter,
)
from benchmarks.dataset.generator import generate_dataset
from benchmarks.metrics.ir_metrics import mrr, ndcg_at_k, precision_at_k

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _evaluate_adapter(adapter, dataset, limit: int) -> dict:
    """Run all queries through an adapter and compute aggregate metrics."""
    per_query: list[dict] = []

    for judgment in dataset.judgments:
        # Build relevance map: memory_id -> grade
        relevance_map: dict[str, float] = {}
        relevant_ids: set[str] = set()
        for idx, grade in zip(
            judgment.relevant_memory_indices, judgment.relevance_grades
        ):
            mid = adapter.get_memory_id_for_index(idx)
            if mid:
                relevance_map[mid] = grade
                relevant_ids.add(mid)

        # Run search
        try:
            results = adapter.search(judgment.query, limit=limit)
        except Exception as e:
            logger.warning(
                "Adapter %s failed on query %r: %s",
                adapter.name,
                judgment.query,
                e,
            )
            per_query.append(
                {
                    "query": judgment.query,
                    "ndcg": 0.0,
                    "mrr": 0.0,
                    "precision": 0.0,
                    "num_results": 0,
                    "error": str(e),
                }
            )
            continue

        ranked_ids = [r["id"] for r in results]

        q_ndcg = ndcg_at_k(ranked_ids, relevance_map, k=limit)
        q_mrr = mrr(ranked_ids, relevant_ids)
        q_prec = precision_at_k(ranked_ids, relevant_ids, k=limit)

        per_query.append(
            {
                "query": judgment.query,
                "ndcg": round(q_ndcg, 4),
                "mrr": round(q_mrr, 4),
                "precision": round(q_prec, 4),
                "num_results": len(results),
            }
        )

    # Aggregate
    n = len(per_query)
    if n == 0:
        return {"per_query": [], "aggregate": {}}

    agg = {
        "ndcg_mean": round(sum(q["ndcg"] for q in per_query) / n, 4),
        "mrr_mean": round(sum(q["mrr"] for q in per_query) / n, 4),
        "precision_mean": round(sum(q["precision"] for q in per_query) / n, 4),
        "num_queries": n,
    }

    return {"per_query": per_query, "aggregate": agg}


def main():
    parser = argparse.ArgumentParser(description="maasv retrieval quality benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--scale",
        choices=["small", "medium", "large"],
        default="small",
        help="Dataset scale",
    )
    parser.add_argument("--limit", type=int, default=5, help="Search result limit (k)")
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Include Ollama adapter (requires local Ollama with qwen3-embedding:8b)",
    )
    args = parser.parse_args()

    print(f"Generating dataset (seed={args.seed}, scale={args.scale})...")
    dataset = generate_dataset(seed=args.seed, scale=args.scale)
    print(
        f"  {len(dataset.memories)} memories, "
        f"{len(dataset.entities)} entities, "
        f"{len(dataset.relationships)} relationships, "
        f"{len(dataset.judgments)} queries"
    )

    adapters = [
        MaasvFullAdapter(),
        MaasvVectorOnlyAdapter(),
        MaasvBM25OnlyAdapter(),
        MaasvGraphOnlyAdapter(),
    ]

    if args.ollama:
        available, msg = MaasvOllamaFullAdapter.is_available()
        if available:
            adapters.append(MaasvOllamaFullAdapter())
            print("Ollama adapter enabled (this will be slower — embedding via local model)")
        else:
            print(f"Ollama adapter skipped: {msg}")

    results: dict = {
        "meta": {
            "seed": args.seed,
            "scale": args.scale,
            "limit": args.limit,
            "num_memories": len(dataset.memories),
            "num_queries": len(dataset.judgments),
        },
        "adapters": {},
    }

    for adapter in adapters:
        print(f"\n{'='*60}")
        print(f"Running: {adapter.name}")
        print(f"{'='*60}")

        # Setup
        t0 = time.monotonic()
        try:
            adapter.setup(dataset)
        except Exception as e:
            print(f"  SETUP FAILED: {e}")
            results["adapters"][adapter.name] = {"error": f"setup failed: {e}"}
            continue
        setup_time = time.monotonic() - t0
        print(f"  Setup: {setup_time:.2f}s")

        # Evaluate
        t0 = time.monotonic()
        adapter_results = _evaluate_adapter(adapter, dataset, limit=args.limit)
        eval_time = time.monotonic() - t0
        print(f"  Eval:  {eval_time:.2f}s")

        adapter_results["timing"] = {
            "setup_s": round(setup_time, 3),
            "eval_s": round(eval_time, 3),
        }

        agg = adapter_results.get("aggregate", {})
        print(
            f"  NDCG@{args.limit}: {agg.get('ndcg_mean', 0):.4f}  "
            f"MRR: {agg.get('mrr_mean', 0):.4f}  "
            f"P@{args.limit}: {agg.get('precision_mean', 0):.4f}"
        )

        results["adapters"][adapter.name] = adapter_results

        # Teardown
        adapter.teardown()

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    header = f"{'Adapter':<25} {'NDCG@'+str(args.limit):<10} {'MRR':<10} {'P@'+str(args.limit):<10}"
    print(header)
    print("-" * len(header))
    for name, data in results["adapters"].items():
        if "error" in data:
            print(f"{name:<25} ERROR: {data['error']}")
            continue
        agg = data.get("aggregate", {})
        print(
            f"{name:<25} "
            f"{agg.get('ndcg_mean', 0):<10.4f} "
            f"{agg.get('mrr_mean', 0):<10.4f} "
            f"{agg.get('precision_mean', 0):<10.4f}"
        )

    # Write JSON
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"retrieval_quality_{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

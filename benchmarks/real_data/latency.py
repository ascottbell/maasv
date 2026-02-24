"""Per-signal latency profiling for retrieval pipeline.

Runs queries multiple times and reports P50, P95, and mean latency
for each signal (vector, BM25, graph) and the full pipeline.
"""

from __future__ import annotations

import logging
import statistics
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _percentile(data: list[float], p: float) -> float:
    """Compute p-th percentile (0-100) from sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def profile_latency(
    queries: list[str],
    db_path: Path,
    num_runs: int = 3,
) -> dict[str, dict[str, float]]:
    """Profile retrieval latency per signal.

    Runs all queries (1 warmup + num_runs-1 timed), measures each signal
    independently plus the full pipeline.

    Args:
        queries: List of query strings.
        db_path: Path to the (copied) production database.
        num_runs: Total runs including warmup (min 2).

    Returns:
        Dict with signal names -> {p50, p95, mean} in milliseconds.
    """
    import maasv
    from maasv.core.db import _db, get_query_embedding, serialize_embedding
    from maasv.core.retrieval import (
        find_similar_memories,
        _find_memories_by_bm25,
        _find_memories_by_graph,
    )

    num_runs = max(2, num_runs)
    timings: dict[str, list[float]] = {
        "vector": [],
        "bm25": [],
        "graph": [],
        "full_pipeline": [],
    }

    for run_idx in range(num_runs):
        is_warmup = run_idx == 0
        label = "warmup" if is_warmup else f"run {run_idx}"
        logger.info("Latency profiling %s (%d queries)", label, len(queries))

        for query in queries:
            # Vector: embed + KNN search
            t0 = time.perf_counter()
            embedding = get_query_embedding(query)
            blob = serialize_embedding(embedding)
            with _db() as db:
                db.execute(
                    """
                    SELECT id, distance
                    FROM memory_vectors
                    WHERE embedding MATCH ?
                    AND k = 20
                    ORDER BY distance
                    """,
                    (blob,),
                ).fetchall()
            t_vector = (time.perf_counter() - t0) * 1000

            # BM25
            t0 = time.perf_counter()
            with _db() as db:
                _find_memories_by_bm25(db, query, limit=20)
            t_bm25 = (time.perf_counter() - t0) * 1000

            # Graph
            t0 = time.perf_counter()
            with _db() as db:
                _find_memories_by_graph(db, query, limit=20)
            t_graph = (time.perf_counter() - t0) * 1000

            # Full pipeline
            t0 = time.perf_counter()
            find_similar_memories(query, limit=20)
            t_full = (time.perf_counter() - t0) * 1000

            if not is_warmup:
                timings["vector"].append(t_vector)
                timings["bm25"].append(t_bm25)
                timings["graph"].append(t_graph)
                timings["full_pipeline"].append(t_full)

    # Compute stats
    results: dict[str, dict[str, float]] = {}
    for signal, values in timings.items():
        if not values:
            results[signal] = {"p50": 0.0, "p95": 0.0, "mean": 0.0}
            continue
        results[signal] = {
            "p50": round(_percentile(values, 50), 2),
            "p95": round(_percentile(values, 95), 2),
            "mean": round(statistics.mean(values), 2),
        }

    # Compute fusion overhead: full - sum(individual)
    if timings["full_pipeline"] and timings["vector"]:
        overhead_values = []
        n = min(
            len(timings["full_pipeline"]),
            len(timings["vector"]),
            len(timings["bm25"]),
            len(timings["graph"]),
        )
        for i in range(n):
            individual_sum = (
                timings["vector"][i]
                + timings["bm25"][i]
                + timings["graph"][i]
            )
            overhead_values.append(timings["full_pipeline"][i] - individual_sum)

        results["fusion_overhead"] = {
            "p50": round(_percentile(overhead_values, 50), 2),
            "p95": round(_percentile(overhead_values, 95), 2),
            "mean": round(statistics.mean(overhead_values), 2),
        }

    return results

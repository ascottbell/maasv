"""Main entry point for real-data benchmarks.

Usage:
    python -m benchmarks.real_data.runner [--db-path PATH] [--generate] [--limit 5]

    --generate: Generate queries + run pooled retrieval + LLM judging.
                Requires ANTHROPIC_API_KEY.
    --db-path:  Path to production DB (default: ~/maasv/data/maasv.db).
    --limit:    Top-k for metric computation (default: 5).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

from benchmarks.metrics.ir_metrics import ndcg_at_k, mrr, precision_at_k

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / "maasv" / "data" / "maasv.db"
QUERIES_FILE = "queries.json"
JUDGMENTS_FILE = "judgments.json"


# ============================================================================
# ADAPTER FUNCTIONS
# ============================================================================

def _init_maasv(db_path: Path) -> None:
    """Initialize maasv singleton with the given DB path."""
    import maasv
    from maasv.config import MaasvConfig
    from maasv.providers.ollama import OllamaEmbed

    # Reset singleton state for clean adapter runs
    maasv._config = None
    maasv._llm = None
    maasv._embed = None
    maasv._initialized = False

    config = MaasvConfig(db_path=db_path, embed_dims=1024)
    embed = OllamaEmbed(model="qwen3-embedding:8b", dims=1024)
    maasv.init(config, llm=None, embed=embed)


def _run_full(query: str, limit: int = 20) -> list[dict]:
    """Full 3-signal + RRF + reranking + diversity pipeline."""
    from maasv.core.retrieval import find_similar_memories
    return find_similar_memories(query, limit=limit)


def _run_vector_only(query: str, limit: int = 20) -> list[dict]:
    """Vector-only: embed query + sqlite-vec KNN."""
    from maasv.core.db import _db, get_query_embedding, serialize_embedding

    embedding = get_query_embedding(query)
    blob = serialize_embedding(embedding)
    with _db() as db:
        rows = db.execute(
            """
            SELECT
                v.id, m.content, m.category, m.subject,
                m.importance, m.access_count, v.distance
            FROM memory_vectors v
            JOIN memories m ON v.id = m.id
            WHERE m.superseded_by IS NULL
            AND v.embedding MATCH ?
            AND k = ?
            ORDER BY distance
            """,
            (blob, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def _run_bm25_only(query: str, limit: int = 20) -> list[dict]:
    """BM25-only: FTS5 keyword matching."""
    from maasv.core.db import _db
    from maasv.core.retrieval import _find_memories_by_bm25

    with _db() as db:
        return _find_memories_by_bm25(db, query, limit=limit)


def _run_graph_only(query: str, limit: int = 20) -> list[dict]:
    """Graph-only: entity traversal."""
    from maasv.core.db import _db
    from maasv.core.retrieval import _find_memories_by_graph

    with _db() as db:
        return _find_memories_by_graph(db, query, limit=limit)


ADAPTERS = {
    "maasv-full": _run_full,
    "maasv-vector-only": _run_vector_only,
    "maasv-bm25-only": _run_bm25_only,
    "maasv-graph-only": _run_graph_only,
}


def _run_all_adapters(query: str, db_path: Path) -> dict[str, list[dict]]:
    """Run a query through all 4 adapters. Used for pooled judging."""
    results = {}
    for name, fn in ADAPTERS.items():
        try:
            results[name] = fn(query, limit=20)
        except Exception as e:
            logger.warning("Adapter %s failed for query %r: %s", name, query[:50], e)
            results[name] = []
    return results


# ============================================================================
# METRIC COMPUTATION
# ============================================================================

def _compute_metrics(
    queries: list[dict],
    judgments: dict[str, dict[str, int]],
    limit: int,
    db_path: Path,
) -> dict[str, dict[str, float]]:
    """Run all queries through each adapter and compute IR metrics.

    Returns:
        Dict mapping adapter_name -> {ndcg, mrr, precision} (averages).
    """
    adapter_scores: dict[str, dict[str, list[float]]] = {
        name: {"ndcg": [], "mrr": [], "precision": []}
        for name in ADAPTERS
    }

    for qobj in queries:
        query_str = qobj["query"]
        query_judgments = judgments.get(query_str, {})
        if not query_judgments:
            continue

        # Build relevance map (normalized to 0-1 for NDCG) and relevant set
        relevance_map: dict[str, float] = {
            mid: grade / 3.0 for mid, grade in query_judgments.items()
        }
        relevant_ids: set[str] = {
            mid for mid, grade in query_judgments.items() if grade >= 2
        }

        for adapter_name, adapter_fn in ADAPTERS.items():
            try:
                results = adapter_fn(query_str, limit=limit)
            except Exception as e:
                logger.warning(
                    "Adapter %s failed for %r: %s",
                    adapter_name,
                    query_str[:50],
                    e,
                )
                results = []

            ranked_ids = [r["id"] for r in results[:limit]]
            adapter_scores[adapter_name]["ndcg"].append(
                ndcg_at_k(ranked_ids, relevance_map, k=limit)
            )
            adapter_scores[adapter_name]["mrr"].append(
                mrr(ranked_ids, relevant_ids)
            )
            adapter_scores[adapter_name]["precision"].append(
                precision_at_k(ranked_ids, relevant_ids, k=limit)
            )

    # Average
    results = {}
    for name, scores in adapter_scores.items():
        results[name] = {}
        for metric, values in scores.items():
            results[name][metric] = round(
                sum(values) / len(values), 4
            ) if values else 0.0
    return results


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def _get_db_stats(db_path: Path) -> dict:
    """Get key stats from the production DB."""
    db = sqlite3.connect(str(db_path))
    stats = {}
    try:
        stats["active_memories"] = db.execute(
            "SELECT COUNT(*) FROM memories WHERE superseded_by IS NULL"
        ).fetchone()[0]
        stats["entities"] = db.execute(
            "SELECT COUNT(*) FROM entities"
        ).fetchone()[0]
        stats["relationships"] = db.execute(
            "SELECT COUNT(*) FROM relationships WHERE valid_to IS NULL"
        ).fetchone()[0]
    except sqlite3.OperationalError as e:
        logger.warning("Could not get DB stats: %s", e)
    finally:
        db.close()
    return stats


def _print_results(
    quality: dict[str, dict[str, float]],
    latency: dict[str, dict[str, float]] | None,
    db_stats: dict,
    num_queries: int,
    limit: int,
) -> None:
    """Print formatted summary tables to stdout."""
    mem_count = db_stats.get("active_memories", "?")
    print(f"\nRETRIEVAL QUALITY ({mem_count} memories, {num_queries} queries, k={limit})")
    print(f"{'Adapter':<25} {'NDCG@'+str(limit):<12} {'MRR':<12} {'P@'+str(limit):<12}")
    print(f"{'-------':<25} {'------':<12} {'---':<12} {'---':<12}")
    for name in ADAPTERS:
        q = quality.get(name, {})
        print(
            f"{name:<25} {q.get('ndcg', 0):<12.4f} "
            f"{q.get('mrr', 0):<12.4f} {q.get('precision', 0):<12.4f}"
        )

    if latency:
        print(f"\nLATENCY (P50 / P95 / mean, ms)")
        print(f"{'Signal':<25} {'P50':<12} {'P95':<12} {'Mean':<12}")
        print(f"{'------':<25} {'---':<12} {'---':<12} {'----':<12}")
        signal_display = {
            "vector": "vector (embed+knn)",
            "bm25": "bm25",
            "graph": "graph",
            "full_pipeline": "full pipeline",
            "fusion_overhead": "fusion overhead",
        }
        for signal in ["vector", "bm25", "graph", "full_pipeline", "fusion_overhead"]:
            if signal not in latency:
                continue
            s = latency[signal]
            label = signal_display.get(signal, signal)
            print(
                f"{label:<25} {s['p50']:<12.1f} "
                f"{s['p95']:<12.1f} {s['mean']:<12.1f}"
            )

    print()


def _save_results(
    quality: dict,
    latency: dict | None,
    db_stats: dict,
    num_queries: int,
    limit: int,
    output_path: Path,
) -> None:
    """Save aggregate metrics (no PII) to JSON."""
    results = {
        "db_stats": db_stats,
        "num_queries": num_queries,
        "limit": limit,
        "quality": quality,
    }
    if latency:
        results["latency"] = latency

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)


# ============================================================================
# PREFLIGHT CHECKS
# ============================================================================

def _check_ollama() -> None:
    """Verify Ollama is running and has the required model."""
    import urllib.request
    import urllib.error

    try:
        resp = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
        data = json.loads(resp.read())
    except (urllib.error.URLError, ConnectionError, OSError):
        print(
            "ERROR: Ollama not reachable at localhost:11434. "
            "Start it with `ollama serve`",
            file=sys.stderr,
        )
        sys.exit(1)

    model_names = [m.get("name", "") for m in data.get("models", [])]
    if not any("qwen3-embedding" in n for n in model_names):
        print(
            "ERROR: qwen3-embedding:8b not found. "
            "Run `ollama pull qwen3-embedding:8b`",
            file=sys.stderr,
        )
        sys.exit(1)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-data retrieval benchmarks using production maasv DB"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to production DB (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate queries + run pooled retrieval + LLM judging (requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Top-k for metric computation (default: 5)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Validate DB exists
    if not args.db_path.exists():
        print(
            f"ERROR: DB not found at {args.db_path}. "
            "Specify with --db-path",
            file=sys.stderr,
        )
        sys.exit(1)

    # Preflight: check Ollama
    _check_ollama()

    # Copy production DB to temp directory (never modify original)
    tmp_dir = tempfile.mkdtemp(prefix="maasv_bench_")
    tmp_db = Path(tmp_dir) / "maasv.db"
    logger.info("Copying production DB to %s", tmp_db)
    shutil.copy2(args.db_path, tmp_db)
    # Also copy WAL/SHM if they exist
    for suffix in ("-wal", "-shm"):
        src = args.db_path.parent / (args.db_path.name + suffix)
        if src.exists():
            shutil.copy2(src, tmp_dir)

    try:
        # Initialize maasv with the temp copy
        _init_maasv(tmp_db)
        db_stats = _get_db_stats(tmp_db)
        logger.info("DB stats: %s", db_stats)

        # Paths for cached data
        real_data_dir = Path(__file__).parent
        queries_path = real_data_dir / QUERIES_FILE
        judgments_path = real_data_dir / JUDGMENTS_FILE
        results_path = Path(__file__).parent.parent / "results" / "real_data.json"

        if args.generate:
            # Step 1: Generate queries
            from benchmarks.real_data.queries import generate_all_queries

            queries = generate_all_queries(
                db_path=tmp_db,
                output_path=queries_path,
            )
            logger.info("Generated %d queries", len(queries))

            # Step 2: Pooled judging
            from benchmarks.real_data.judge import pool_and_judge

            judgments = pool_and_judge(
                db_path=tmp_db,
                queries=queries,
                run_adapters_fn=_run_all_adapters,
                judgments_path=judgments_path,
            )
            logger.info("Judged %d queries", len(judgments))

        # Load cached queries + judgments
        if not queries_path.exists():
            print(
                "ERROR: No queries found. Run with --generate first.",
                file=sys.stderr,
            )
            sys.exit(1)

        with open(queries_path) as f:
            queries = json.load(f)

        if not judgments_path.exists():
            print(
                "ERROR: No judgments found. Run with --generate first.",
                file=sys.stderr,
            )
            sys.exit(1)

        with open(judgments_path) as f:
            judgments = json.load(f)

        # Re-init maasv (clean singleton state for metric computation)
        _init_maasv(tmp_db)

        # Step 3: Compute quality metrics
        logger.info("Computing quality metrics (k=%d, %d queries)", args.limit, len(queries))
        quality = _compute_metrics(queries, judgments, args.limit, tmp_db)

        # Step 4: Latency profiling
        from benchmarks.real_data.latency import profile_latency

        # Re-init for clean latency measurement
        _init_maasv(tmp_db)
        logger.info("Running latency profiling")
        query_strings = [q["query"] for q in queries]
        latency = profile_latency(query_strings, tmp_db)

        # Output
        _print_results(quality, latency, db_stats, len(queries), args.limit)
        _save_results(quality, latency, db_stats, len(queries), args.limit, results_path)

    finally:
        # Clean up temp directory
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("Cleaned up temp directory")


if __name__ == "__main__":
    main()

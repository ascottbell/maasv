"""Cognition benchmark runner.

Usage:
    python -m benchmarks.cognition.runner [--db-path PATH] [--limit 5] [--with-llm]

    --db-path:  Path to production DB (default: ~/maasv/data/maasv.db).
    --limit:    Top-k for retrieval (default: 5).
    --with-llm: Enable LLM-as-judge for proactive relevance (requires ANTHROPIC_API_KEY).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

from benchmarks.cognition.queries import (
    CONSOLIDATION_QUERIES,
    DECAY_IDENTITY_QUERIES,
    GRAPH_QUERIES,
    PROACTIVE_QUERIES,
    SESSION_CHAINS,
    TEMPORAL_QUERIES,
)
from benchmarks.cognition.scoring import (
    score_consolidation,
    score_decay_protection,
    score_graph_traversal,
    score_proactive_llm,
    score_session_coherence,
    score_temporal,
)

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / "maasv" / "data" / "maasv.db"

# Category weights for composite cognition score
CATEGORY_WEIGHTS = {
    "temporal": 0.20,
    "session_coherence": 0.15,
    "graph_traversal": 0.20,
    "consolidation": 0.10,
    "decay_protection": 0.15,
    "proactive": 0.20,
}


# ============================================================================
# MAASV INIT + ADAPTERS
# ============================================================================

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


def _run_full(query: str, limit: int = 10, session_context: dict | None = None) -> list[dict]:
    from maasv.core.retrieval import find_similar_memories
    return find_similar_memories(query, limit=limit, session_context=session_context)


def _run_vector_only(query: str, limit: int = 10) -> list[dict]:
    from maasv.core.db import _db, get_query_embedding, serialize_embedding
    embedding = get_query_embedding(query)
    blob = serialize_embedding(embedding)
    with _db() as db:
        rows = db.execute(
            """
            SELECT v.id, m.content, m.category, m.subject,
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


def _run_graph_only(query: str, limit: int = 10) -> list[dict]:
    from maasv.core.db import _db
    from maasv.core.retrieval import _find_memories_by_graph
    with _db() as db:
        return _find_memories_by_graph(db, query, limit=limit)


# ============================================================================
# BENCHMARK RUNNERS
# ============================================================================

def _run_temporal(db_conn: sqlite3.Connection, limit: int) -> dict:
    """Run temporal reasoning benchmark."""
    logger.info("Running temporal reasoning (%d queries)", len(TEMPORAL_QUERIES))
    results = []
    for q in TEMPORAL_QUERIES:
        full = _run_full(q["query"], limit=limit)
        detail = score_temporal(q, full, db_conn)
        detail["query_id"] = q["id"]
        detail["query"] = q["query"]
        results.append(detail)

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    return {
        "category": "temporal",
        "avg_score": round(avg_score, 4),
        "num_queries": len(results),
        "per_query": results,
    }


def _run_session_coherence(db_conn: sqlite3.Connection, limit: int) -> dict:
    """Run session coherence benchmark."""
    logger.info("Running session coherence (%d chains)", len(SESSION_CHAINS))
    results = []

    for chain in SESSION_CHAINS:
        queries = chain["queries"]

        # Run WITH session context
        session_ctx = {}
        results_with = []
        for q_str in queries:
            r = _run_full(q_str, limit=limit, session_context=session_ctx)
            results_with.append(r)

        # Run WITHOUT session context (each query independent)
        results_without = []
        for q_str in queries:
            r = _run_full(q_str, limit=limit, session_context=None)
            results_without.append(r)

        detail = score_session_coherence(chain, results_with, results_without, db_conn)
        detail["chain_id"] = chain["id"]
        detail["description"] = chain["description"]
        results.append(detail)

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    return {
        "category": "session_coherence",
        "avg_score": round(avg_score, 4),
        "num_chains": len(results),
        "per_chain": results,
    }


def _run_graph_traversal(db_conn: sqlite3.Connection, limit: int) -> dict:
    """Run cross-domain graph traversal benchmark."""
    logger.info("Running graph traversal (%d queries)", len(GRAPH_QUERIES))
    results = []

    for q in GRAPH_QUERIES:
        full = _run_full(q["query"], limit=limit)
        vector = _run_vector_only(q["query"], limit=limit)
        graph = _run_graph_only(q["query"], limit=limit)

        detail = score_graph_traversal(q, full, vector, graph, db_conn)
        detail["query_id"] = q["id"]
        detail["query"] = q["query"]
        results.append(detail)

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    return {
        "category": "graph_traversal",
        "avg_score": round(avg_score, 4),
        "num_queries": len(results),
        "per_query": results,
    }


def _run_consolidation(db_conn: sqlite3.Connection, limit: int) -> dict:
    """Run consolidation resistance benchmark."""
    logger.info("Running consolidation (%d queries)", len(CONSOLIDATION_QUERIES))
    results = []

    for q in CONSOLIDATION_QUERIES:
        full = _run_full(q["query"], limit=limit)
        detail = score_consolidation(q, full, db_conn)
        detail["query_id"] = q["id"]
        detail["query"] = q["query"]
        results.append(detail)

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    return {
        "category": "consolidation",
        "avg_score": round(avg_score, 4),
        "num_queries": len(results),
        "per_query": results,
    }


def _run_decay_protection(db_conn: sqlite3.Connection, limit: int) -> dict:
    """Run decay + identity protection benchmark."""
    logger.info("Running decay/protection (%d queries)", len(DECAY_IDENTITY_QUERIES))
    results = []

    for q in DECAY_IDENTITY_QUERIES:
        full = _run_full(q["query"], limit=limit)
        detail = score_decay_protection(q, full, db_conn)
        detail["query_id"] = q["id"]
        detail["query"] = q["query"]
        results.append(detail)

    # Separate protected vs transient scores
    protected = [r for r in results if "protect" in r["query_id"]]
    transient = [r for r in results if "event" in r["query_id"]]

    avg_protected = sum(r["score"] for r in protected) / len(protected) if protected else 0.0
    avg_transient = sum(r["score"] for r in transient) / len(transient) if transient else 0.0
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0

    return {
        "category": "decay_protection",
        "avg_score": round(avg_score, 4),
        "identity_protection_score": round(avg_protected, 4),
        "event_decay_score": round(avg_transient, 4),
        "num_queries": len(results),
        "per_query": results,
    }


def _run_proactive(
    db_conn: sqlite3.Connection, limit: int, use_llm: bool = False
) -> dict:
    """Run proactive relevance benchmark."""
    logger.info(
        "Running proactive relevance (%d queries, llm=%s)",
        len(PROACTIVE_QUERIES),
        use_llm,
    )

    client = None
    if use_llm:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
        else:
            logger.warning("ANTHROPIC_API_KEY not set, skipping LLM judge")

    results = []
    for q in PROACTIVE_QUERIES:
        full = _run_full(q["query"], limit=limit)
        vector = _run_vector_only(q["query"], limit=limit)

        detail = score_proactive_llm(q, full, vector, db_conn, client=client)
        detail["query_id"] = q["id"]
        detail["query"] = q["query"]
        results.append(detail)

    # Filter out skipped (-1) scores
    scored = [r for r in results if r["score"] >= 0]
    avg_score = sum(r["score"] for r in scored) / len(scored) if scored else -1.0

    return {
        "category": "proactive",
        "avg_score": round(avg_score, 4),
        "num_queries": len(results),
        "num_scored": len(scored),
        "per_query": results,
    }


# ============================================================================
# OUTPUT
# ============================================================================

def _compute_composite(category_scores: dict[str, float]) -> float:
    """Compute weighted composite cognition score."""
    total = 0.0
    weight_sum = 0.0
    for cat, weight in CATEGORY_WEIGHTS.items():
        score = category_scores.get(cat, -1.0)
        if score >= 0:  # Skip categories that weren't run
            total += score * weight
            weight_sum += weight
    if weight_sum == 0:
        return 0.0
    return total / weight_sum


def _print_results(all_results: dict, db_stats: dict) -> None:
    """Print formatted cognition benchmark results."""
    mem_count = db_stats.get("active_memories", "?")
    print(f"\nCOGNITION BENCHMARK ({mem_count} memories)")
    print(f"{'Category':<25} {'Score':<12} {'Details'}")
    print(f"{'--------':<25} {'-----':<12} {'-------'}")

    category_scores = {}
    for cat_key, cat_data in all_results.items():
        score = cat_data.get("avg_score", -1.0)
        category_scores[cat_key] = score

        if score < 0:
            score_str = "SKIPPED"
        else:
            score_str = f"{score:.4f}"

        # Build detail string
        details = []
        if "num_queries" in cat_data:
            details.append(f"{cat_data['num_queries']} queries")
        if "num_chains" in cat_data:
            details.append(f"{cat_data['num_chains']} chains")
        if "identity_protection_score" in cat_data:
            details.append(
                f"protect={cat_data['identity_protection_score']:.2f} "
                f"decay={cat_data['event_decay_score']:.2f}"
            )
        if "num_scored" in cat_data and cat_data.get("num_scored", 0) != cat_data.get("num_queries", 0):
            details.append(f"{cat_data['num_scored']} scored (LLM)")

        detail_str = ", ".join(details)
        print(f"{cat_key:<25} {score_str:<12} {detail_str}")

    composite = _compute_composite(category_scores)
    print(f"\n{'COMPOSITE':<25} {composite:<12.4f}")
    print()


def _save_results(
    all_results: dict,
    db_stats: dict,
    limit: int,
    output_path: Path,
) -> None:
    """Save cognition benchmark results to JSON (no PII)."""
    # Strip memory content from per-query results before saving
    clean_results = {}
    for cat_key, cat_data in all_results.items():
        clean = dict(cat_data)
        # Remove any content fields from per-query detail
        for key in ("per_query", "per_chain"):
            if key in clean:
                for entry in clean[key]:
                    entry.pop("content", None)
                    # Strip grades that reference memory content
                    for grade_key in ("full_grades", "vector_grades"):
                        if grade_key in entry:
                            # Keep only {mem_id: grade}, no content
                            pass
        clean_results[cat_key] = clean

    category_scores = {k: v.get("avg_score", -1.0) for k, v in all_results.items()}

    output = {
        "db_stats": db_stats,
        "limit": limit,
        "composite_score": round(_compute_composite(category_scores), 4),
        "category_scores": {k: round(v, 4) for k, v in category_scores.items()},
        "category_weights": CATEGORY_WEIGHTS,
        "details": clean_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Cognition results saved to %s", output_path)


# ============================================================================
# PREFLIGHT
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
            "ERROR: Ollama not reachable at localhost:11434.",
            file=sys.stderr,
        )
        sys.exit(1)

    model_names = [m.get("name", "") for m in data.get("models", [])]
    if not any("qwen3-embedding" in n for n in model_names):
        print(
            "ERROR: qwen3-embedding:8b not found. Run `ollama pull qwen3-embedding:8b`",
            file=sys.stderr,
        )
        sys.exit(1)


def _get_db_stats(db_path: Path) -> dict:
    """Get key stats from the production DB."""
    db = sqlite3.connect(str(db_path))
    stats = {}
    try:
        stats["active_memories"] = db.execute(
            "SELECT COUNT(*) FROM memories WHERE superseded_by IS NULL"
        ).fetchone()[0]
        stats["total_memories"] = db.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0]
        stats["entities"] = db.execute(
            "SELECT COUNT(*) FROM entities"
        ).fetchone()[0]
        stats["relationships"] = db.execute(
            "SELECT COUNT(*) FROM relationships WHERE valid_to IS NULL"
        ).fetchone()[0]
        stats["superseded"] = stats["total_memories"] - stats["active_memories"]
    except sqlite3.OperationalError as e:
        logger.warning("Could not get DB stats: %s", e)
    finally:
        db.close()
    return stats


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cognition benchmarks using production maasv DB"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to production DB (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Top-k for retrieval (default: 5)",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Enable LLM-as-judge for proactive relevance (requires ANTHROPIC_API_KEY)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.db_path.exists():
        print(f"ERROR: DB not found at {args.db_path}", file=sys.stderr)
        sys.exit(1)

    _check_ollama()

    # Copy production DB to temp directory (never modify original)
    tmp_dir = tempfile.mkdtemp(prefix="maasv_cognition_")
    tmp_db = Path(tmp_dir) / "maasv.db"
    logger.info("Copying production DB to %s", tmp_db)
    shutil.copy2(args.db_path, tmp_db)
    for suffix in ("-wal", "-shm"):
        src = args.db_path.parent / (args.db_path.name + suffix)
        if src.exists():
            shutil.copy2(src, tmp_dir)

    try:
        _init_maasv(tmp_db)
        db_stats = _get_db_stats(tmp_db)
        logger.info("DB stats: %s", db_stats)

        # Open a read-only connection for scoring queries
        db_conn = sqlite3.connect(str(tmp_db))
        db_conn.row_factory = sqlite3.Row

        results_path = Path(__file__).parent.parent / "results" / "cognition.json"

        # Run all categories
        all_results = {}

        all_results["temporal"] = _run_temporal(db_conn, args.limit)

        # Re-init for clean state
        _init_maasv(tmp_db)
        all_results["session_coherence"] = _run_session_coherence(db_conn, args.limit)

        _init_maasv(tmp_db)
        all_results["graph_traversal"] = _run_graph_traversal(db_conn, args.limit)

        _init_maasv(tmp_db)
        all_results["consolidation"] = _run_consolidation(db_conn, args.limit)

        _init_maasv(tmp_db)
        all_results["decay_protection"] = _run_decay_protection(db_conn, args.limit)

        _init_maasv(tmp_db)
        all_results["proactive"] = _run_proactive(db_conn, args.limit, use_llm=args.with_llm)

        db_conn.close()

        # Output
        _print_results(all_results, db_stats)
        _save_results(all_results, db_stats, args.limit, results_path)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("Cleaned up temp directory")


if __name__ == "__main__":
    main()

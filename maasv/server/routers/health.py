"""Health and stats endpoints."""

import logging
import time

from fastapi import APIRouter, Depends

from maasv.server.auth import require_auth

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
def health():
    """Health check: DB accessible."""
    from maasv.core.db import _db

    try:
        with _db() as db:
            db.execute("SELECT 1").fetchone()
        return {"status": "healthy"}
    except Exception:
        logger.exception("Health check failed")
        return {"status": "unhealthy"}


@router.get("/stats", dependencies=[Depends(require_auth)])
def stats():
    """Detailed stats: counts by category, retrieval latency probe."""
    from maasv.core.db import _db
    from maasv.core.retrieval import find_similar_memories

    with _db() as db:
        # Memory counts by category
        category_rows = db.execute("""
            SELECT category, COUNT(*) as count
            FROM memories
            WHERE superseded_by IS NULL
            GROUP BY category
            ORDER BY count DESC
        """).fetchall()

        # Entity counts by type
        entity_rows = db.execute("""
            SELECT entity_type, COUNT(*) as count
            FROM entities
            GROUP BY entity_type
            ORDER BY count DESC
        """).fetchall()

        # Total counts
        total_memories = db.execute(
            "SELECT COUNT(*) FROM memories WHERE superseded_by IS NULL"
        ).fetchone()[0]
        total_entities = db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        total_relationships = db.execute(
            "SELECT COUNT(*) FROM relationships WHERE valid_to IS NULL"
        ).fetchone()[0]
        total_superseded = db.execute(
            "SELECT COUNT(*) FROM memories WHERE superseded_by IS NOT NULL"
        ).fetchone()[0]

    # Retrieval latency probe (only if we have memories)
    latency_ms = None
    if total_memories > 0:
        start = time.perf_counter()
        find_similar_memories("test query", limit=3)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)

    # Wisdom stats
    wisdom_stats = None
    try:
        from maasv.core.wisdom import get_stats
        wisdom_stats = get_stats()
    except Exception:
        pass

    return {
        "memories": {
            "total_active": total_memories,
            "total_superseded": total_superseded,
            "by_category": {row["category"]: row["count"] for row in category_rows},
        },
        "entities": {
            "total": total_entities,
            "by_type": {row["entity_type"]: row["count"] for row in entity_rows},
        },
        "relationships": {
            "total_active": total_relationships,
        },
        "retrieval_latency_ms": latency_ms,
        "wisdom": wisdom_stats,
    }

"""
Graph Reorganization - Sleep-Time Compute

Optimizes the knowledge graph for faster retrieval:
- Tracks access patterns
- Pre-computes common traversal paths
- Cleans up stale/orphaned data
"""

import logging
import json
from typing import Callable
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("maasv.lifecycle.reorganize")


def run_reorganize_job(data: dict, cancel_check: Callable[[], bool]) -> dict:
    """Run a graph reorganization job."""
    mode = data.get("mode", "incremental")

    results = {"optimizations": [], "cleaned": 0, "paths_cached": 0}

    if cancel_check():
        return {**results, "cancelled": True}

    paths_cached = _cache_common_paths()
    results["paths_cached"] = paths_cached
    results["optimizations"].append("cached_common_paths")

    if cancel_check():
        return {**results, "cancelled": True}

    if mode == "full":
        cleaned = _cleanup_orphans()
        results["cleaned"] = cleaned
        results["optimizations"].append("cleaned_orphans")

    return results


def _cache_common_paths() -> int:
    """Pre-compute and cache common traversal paths."""
    from maasv.core.graph import find_entity_by_name, get_entity_relationships

    cached = 0

    try:
        # Find the primary user entity — use config's known_entities if available
        import maasv
        config = maasv.get_config()

        # Look for the first "person" in known_entities
        primary_person = None
        for name, etype in config.known_entities.items():
            if etype == "person":
                primary_person = name
                break

        if not primary_person:
            logger.debug("[Reorganize] No primary person in known_entities, skipping path caching")
            return 0

        person = find_entity_by_name(primary_person)
        if not person:
            logger.debug(f"[Reorganize] No {primary_person} entity found, skipping path caching")
            return 0

        rels = get_entity_relationships(person["id"], predicate=None, direction="outgoing")

        family_predicates = {"spouse", "child", "married_to", "parent_of", "sibling"}
        family_members = [r for r in rels if r.get("predicate") in family_predicates]
        if family_members:
            _store_cached_path("primary_family", family_members)
            cached += 1

        project_rels = [r for r in rels if r.get("predicate") == "works_on"]
        if project_rels:
            _store_cached_path("primary_projects", project_rels)
            cached += 1

        logger.info(f"[Reorganize] Cached {cached} common paths")
        return cached

    except Exception as e:
        logger.warning(f"[Reorganize] Failed to cache paths: {e}")
        return 0


def _store_cached_path(path_name: str, relationships: list[dict]):
    """Store a cached path for fast retrieval."""
    from maasv.core.db import get_db

    db = get_db()
    try:
        db.execute("""
            CREATE TABLE IF NOT EXISTS cached_paths (
                name TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                cached_at TEXT NOT NULL,
                expires_at TEXT
            )
        """)

        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=1)

        simplified = []
        for r in relationships:
            simplified.append({
                "id": r.get("id"),
                "predicate": r.get("predicate"),
                "object_id": r.get("object_id"),
                "object_name": r.get("object_name"),
                "object_type": r.get("object_type"),
                "object_value": r.get("object_value")
            })

        db.execute("""
            INSERT OR REPLACE INTO cached_paths (name, data, cached_at, expires_at)
            VALUES (?, ?, ?, ?)
        """, (path_name, json.dumps(simplified), now.isoformat(), expires.isoformat()))

        db.commit()
    except Exception as e:
        logger.warning(f"[Reorganize] Failed to store cached path '{path_name}': {e}")
    finally:
        db.close()


def _cleanup_orphans() -> int:
    """Clean up orphaned entities (no relationships, created >7 days ago).

    Uses a single atomic DELETE ... WHERE NOT EXISTS to avoid TOCTOU races
    between the SELECT and DELETE.
    """
    from maasv.core.db import get_db

    db = get_db()
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")

        cursor = db.execute("""
            DELETE FROM entities
            WHERE created_at < ?
            AND NOT EXISTS (
                SELECT 1 FROM relationships r
                WHERE r.subject_id = entities.id OR r.object_id = entities.id
            )
        """, (cutoff,))
        deleted = cursor.rowcount
        db.commit()

        if deleted:
            logger.info(f"[Reorganize] Cleaned {deleted} orphaned entities")

        return deleted

    except Exception as e:
        logger.warning(f"[Reorganize] Failed to cleanup orphans: {e}")
        return 0
    finally:
        db.close()

"""
Memory Hygiene - Sleep-Time Cleanup

Runs during idle periods to:
1. Deduplicate memories with high embedding similarity (>threshold)
2. Prune stale, low-confidence memories
3. Consolidate clusters of related memories

All operations are audited and can be run in dry-run mode.
"""

import logging
import json
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger("maasv.lifecycle.memory_hygiene")

# Hard cap on memories loaded into hygiene jobs to bound memory usage.
# At 10K memories with 1024-dim embeddings, pre-loaded embeddings ≈ 40MB.
MAX_HYGIENE_MEMORIES = 10_000


@dataclass
class HygieneStats:
    """Statistics from a hygiene run."""
    duplicates_found: int = 0
    duplicates_merged: int = 0
    stale_found: int = 0
    stale_pruned: int = 0
    rel_duplicates_found: int = 0
    rel_duplicates_removed: int = 0
    entity_dupes_found: int = 0
    entity_dupes_merged: int = 0
    clusters_found: int = 0
    clusters_consolidated: int = 0
    errors: list = field(default_factory=list)
    dry_run: bool = True
    backup_path: Optional[str] = None
    started_at: str = ""
    completed_at: str = ""


def run_memory_hygiene_job(data: dict, cancel_check: Callable[[], bool]) -> dict:
    """
    Run a memory hygiene job.

    Args:
        data: {
            "mode": "full" | "incremental",
            "dry_run": bool (default True),
            "dedup": bool (default True),
            "prune": bool (default True),
            "consolidate": bool (default False - expensive)
        }
        cancel_check: Function to check if job should stop

    Returns:
        {"stats": HygieneStats as dict, "cancelled": bool}
    """
    mode = data.get("mode", "incremental")
    dry_run = data.get("dry_run", True)
    do_dedup = data.get("dedup", True)
    do_prune = data.get("prune", True)
    do_consolidate = data.get("consolidate", False)

    stats = HygieneStats(
        dry_run=dry_run,
        started_at=datetime.now(timezone.utc).isoformat()
    )

    if cancel_check():
        return {"stats": _stats_to_dict(stats), "cancelled": True}

    # Create backup before any destructive operations
    if not dry_run:
        backup_path = _create_backup()
        if backup_path:
            stats.backup_path = str(backup_path)
            logger.info(f"[MemoryHygiene] Created backup: {backup_path}")
        else:
            logger.error("[MemoryHygiene] Failed to create backup, aborting")
            stats.errors.append("Failed to create backup")
            return {"stats": _stats_to_dict(stats), "cancelled": False}

    # Step 1: Deduplicate
    if do_dedup and not cancel_check():
        try:
            dedup_stats = _deduplicate_memories(dry_run, cancel_check)
            stats.duplicates_found = dedup_stats["found"]
            stats.duplicates_merged = dedup_stats["merged"]
            logger.info(f"[MemoryHygiene] Dedup: found {stats.duplicates_found}, merged {stats.duplicates_merged}")
        except Exception as e:
            logger.error(f"[MemoryHygiene] Dedup failed: {e}", exc_info=True)
            stats.errors.append(f"Dedup error: {e}")

    if cancel_check():
        stats.completed_at = datetime.now(timezone.utc).isoformat()
        return {"stats": _stats_to_dict(stats), "cancelled": True}

    # Step 2: Prune stale
    if do_prune and not cancel_check():
        try:
            prune_stats = _prune_stale_memories(dry_run, cancel_check)
            stats.stale_found = prune_stats["found"]
            stats.stale_pruned = prune_stats["pruned"]
            logger.info(f"[MemoryHygiene] Prune: found {stats.stale_found}, pruned {stats.stale_pruned}")
        except Exception as e:
            logger.error(f"[MemoryHygiene] Prune failed: {e}", exc_info=True)
            stats.errors.append(f"Prune error: {e}")

    if cancel_check():
        stats.completed_at = datetime.now(timezone.utc).isoformat()
        return {"stats": _stats_to_dict(stats), "cancelled": True}

    # Step 3: Deduplicate relationships
    if not cancel_check():
        try:
            rel_dedup_stats = _deduplicate_relationships(dry_run)
            stats.rel_duplicates_found = rel_dedup_stats["found"]
            stats.rel_duplicates_removed = rel_dedup_stats["removed"]
            logger.info(f"[MemoryHygiene] Rel dedup: found {stats.rel_duplicates_found} groups, removed {stats.rel_duplicates_removed}")
        except Exception as e:
            logger.error(f"[MemoryHygiene] Relationship dedup failed: {e}", exc_info=True)
            stats.errors.append(f"Relationship dedup error: {e}")

    if cancel_check():
        stats.completed_at = datetime.now(timezone.utc).isoformat()
        return {"stats": _stats_to_dict(stats), "cancelled": True}

    # Step 4: Deduplicate entities (normalization-based only — conservative)
    if not cancel_check():
        try:
            ent_dedup_stats = _deduplicate_entities(dry_run)
            stats.entity_dupes_found = ent_dedup_stats["found"]
            stats.entity_dupes_merged = ent_dedup_stats["merged"]
            logger.info(f"[MemoryHygiene] Entity dedup: found {stats.entity_dupes_found} dupes in {ent_dedup_stats['clusters']} clusters, merged {stats.entity_dupes_merged}")
        except Exception as e:
            logger.error(f"[MemoryHygiene] Entity dedup failed: {e}", exc_info=True)
            stats.errors.append(f"Entity dedup error: {e}")

    if cancel_check():
        stats.completed_at = datetime.now(timezone.utc).isoformat()
        return {"stats": _stats_to_dict(stats), "cancelled": True}

    # Step 5: Consolidate (only in full mode, expensive)
    if do_consolidate and mode == "full" and not cancel_check():
        try:
            consolidate_stats = _consolidate_clusters(dry_run, cancel_check)
            stats.clusters_found = consolidate_stats["found"]
            stats.clusters_consolidated = consolidate_stats["consolidated"]
            logger.info(f"[MemoryHygiene] Consolidate: found {stats.clusters_found}, consolidated {stats.clusters_consolidated}")
        except Exception as e:
            logger.error(f"[MemoryHygiene] Consolidate failed: {e}", exc_info=True)
            stats.errors.append(f"Consolidate error: {e}")

    stats.completed_at = datetime.now(timezone.utc).isoformat()
    _log_hygiene_run(stats)

    return {"stats": _stats_to_dict(stats), "cancelled": False}


def _create_backup() -> Optional[Path]:
    """Create a backup of the database before modifications, retaining only the last N.

    Uses sqlite3 Connection.backup() for a consistent snapshot even under
    concurrent writes (safe with WAL mode), instead of filesystem copy.
    """
    import sqlite3
    import maasv

    config = maasv.get_config()

    if not config.backup_dir:
        logger.warning("[MemoryHygiene] No backup_dir configured, skipping backup")
        return None

    try:
        import os

        backup_dir = config.backup_dir / "memory_hygiene"
        backup_dir.mkdir(parents=True, exist_ok=True)
        # Restrict backup directory permissions (owner only)
        try:
            os.chmod(backup_dir, 0o700)
        except OSError:
            pass

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"pre_hygiene_{timestamp}.db"

        src = sqlite3.connect(str(config.db_path))
        dst = sqlite3.connect(str(backup_path))
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()

        # Restrict backup file permissions (owner read/write only)
        try:
            os.chmod(backup_path, 0o600)
        except OSError:
            pass

        # Enforce retention: keep only the last N backups
        _enforce_backup_retention(backup_dir, config.max_hygiene_backups)

        return backup_path
    except Exception as e:
        logger.error(f"[MemoryHygiene] Backup failed: {e}")
        return None


def _enforce_backup_retention(backup_dir: Path, max_backups: int):
    """Delete old hygiene backups, keeping only the most recent max_backups."""
    try:
        backups = sorted(
            backup_dir.glob("pre_hygiene_*.db"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        for old_backup in backups[max_backups:]:
            try:
                old_backup.unlink()
                logger.info(f"[MemoryHygiene] Removed old backup: {old_backup.name}")
            except FileNotFoundError:
                pass  # Already deleted by another process/thread

    except Exception as e:
        logger.warning(f"[MemoryHygiene] Backup retention cleanup failed: {e}")


def _is_protected(memory: dict) -> bool:
    """Check if a memory is protected from deletion."""
    import maasv

    config = maasv.get_config()

    category = memory.get("category", "").lower()
    subject = (memory.get("subject") or "").lower()

    # Never delete protected categories
    if category in config.protected_categories:
        return True

    # Never delete memories about protected subjects
    if subject in config.protected_subjects:
        return True

    return False


def _is_protected_from_dedup(memory: dict) -> bool:
    """
    Check if a memory is protected from deduplication (merging).

    Only identity memories are protected — we want to merge duplicates
    in other categories (family, preference, etc.) into better versions.
    """
    category = memory.get("category", "").lower()
    if category == "identity":
        return True

    return False


def _deduplicate_memories(dry_run: bool, cancel_check: Callable[[], bool]) -> dict:
    """
    Find and merge duplicate memories using embedding similarity.

    Cross-category: compares ALL active memories (not just within same category).
    Uses Union-Find to cluster transitively similar memories (A≈B and B≈C → one cluster).

    Keeper selection (in order):
    1. Longer content (more information)
    2. Higher importance
    3. More recent (created_at)

    Access count transfer: keeper gets max(keeper.access_count, max(removed.access_count))
    so frequently-accessed duplicates don't lose their retrieval boost.
    """
    import maasv
    from maasv.core.db import get_db

    config = maasv.get_config()
    similarity_threshold = config.similarity_threshold

    stats = {"found": 0, "merged": 0, "clusters": []}
    db = get_db()

    try:
        # Get active memories, capped to bound memory usage
        memories = db.execute("""
            SELECT id, content, category, subject, confidence, metadata,
                   created_at, importance, access_count
            FROM memories
            WHERE superseded_by IS NULL
            ORDER BY created_at DESC
            LIMIT ?
        """, (MAX_HYGIENE_MEMORIES,)).fetchall()

        memories = [dict(m) for m in memories]
        mem_by_id = {m["id"]: m for m in memories}
        logger.info(f"[MemoryHygiene] Checking {len(memories)} memories for duplicates (cross-category)")

        # --- Phase 1: Find all duplicate pairs ---
        duplicate_pairs = []  # (id_a, id_b, similarity)
        seen_pairs = set()

        for mem in memories:
            if cancel_check():
                break
            if _is_protected_from_dedup(mem):
                continue

            embedding_row = db.execute(
                "SELECT embedding FROM memory_vectors WHERE id = ?",
                (mem["id"],)
            ).fetchone()

            if not embedding_row:
                continue

            # Search across ALL categories (no category filter)
            similar = db.execute("""
                SELECT v.id, v.distance
                FROM memory_vectors v
                JOIN memories m ON v.id = m.id
                WHERE m.superseded_by IS NULL
                AND m.id != ?
                AND v.embedding MATCH ?
                AND k = 10
                ORDER BY distance
            """, (mem["id"], embedding_row["embedding"])).fetchall()

            for row in similar:
                if cancel_check():
                    break

                distance = row["distance"]
                similarity = 1 - (distance ** 2 / 2)

                if similarity > similarity_threshold:
                    other_id = row["id"]
                    pair_key = tuple(sorted([mem["id"], other_id]))
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        duplicate_pairs.append((mem["id"], other_id, similarity))

        logger.info(f"[MemoryHygiene] Found {len(duplicate_pairs)} duplicate pairs")

        if not duplicate_pairs:
            return stats

        # --- Phase 2: Union-Find to build transitive clusters ---
        parent = {}

        def find(x):
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])  # path compression
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for id_a, id_b, _ in duplicate_pairs:
            # Only union if neither is protected from dedup
            mem_a = mem_by_id.get(id_a)
            mem_b = mem_by_id.get(id_b)
            if mem_a and mem_b and not _is_protected_from_dedup(mem_a) and not _is_protected_from_dedup(mem_b):
                union(id_a, id_b)

        # Group into clusters
        clusters = {}
        for id_a, id_b, _ in duplicate_pairs:
            for mid in (id_a, id_b):
                root = find(mid)
                if root not in clusters:
                    clusters[root] = set()
                clusters[root].add(mid)

        # --- Phase 3: For each cluster, pick keeper and merge ---
        for root, member_ids in clusters.items():
            if cancel_check():
                break

            members = [mem_by_id[mid] for mid in member_ids if mid in mem_by_id]
            if len(members) < 2:
                continue

            # Skip clusters where all members are protected
            non_protected = [m for m in members if not _is_protected_from_dedup(m)]
            if len(non_protected) < 2:
                continue

            # Keeper selection: longest content → highest importance → most recent
            members.sort(key=lambda m: (
                len(m.get("content") or ""),
                m.get("importance") or 0.5,
                m.get("created_at") or "",
            ), reverse=True)

            keeper = members[0]
            to_remove = [m for m in members[1:] if not _is_protected_from_dedup(m)]

            if not to_remove:
                continue

            # Access count transfer: keeper gets the max across the whole cluster
            max_access = max(m.get("access_count") or 0 for m in members)

            stats["found"] += len(to_remove)
            cluster_info = {
                "keeper_id": keeper["id"],
                "keeper_content": keeper["content"][:120],
                "keeper_access_count": keeper.get("access_count") or 0,
                "inherited_access_count": max_access,
                "removed": [
                    {
                        "id": m["id"],
                        "content": m["content"][:120],
                        "category": m["category"],
                        "access_count": m.get("access_count") or 0,
                    }
                    for m in to_remove
                ],
            }
            stats["clusters"].append(cluster_info)

            if not dry_run:
                for removed in to_remove:
                    # Mark as superseded
                    db.execute("""
                        UPDATE memories
                        SET superseded_by = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (keeper["id"], removed["id"]))

                    # Merge metadata (guard against non-dict JSON values)
                    keep_meta = json.loads(keeper["metadata"]) if keeper.get("metadata") else {}
                    if not isinstance(keep_meta, dict):
                        keep_meta = {}
                    remove_meta = json.loads(removed["metadata"]) if removed.get("metadata") else {}
                    if not isinstance(remove_meta, dict):
                        remove_meta = {}
                    if remove_meta:
                        merged_meta = {**remove_meta, **keep_meta}
                        db.execute("""
                            UPDATE memories
                            SET metadata = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE id = ?
                        """, (json.dumps(merged_meta), keeper["id"]))

                    stats["merged"] += 1

                # Transfer access count to keeper
                if max_access > (keeper.get("access_count") or 0):
                    db.execute("""
                        UPDATE memories
                        SET access_count = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (max_access, keeper["id"]))

        if not dry_run:
            db.commit()

        logger.info(
            f"[MemoryHygiene] Dedup complete: {len(stats['clusters'])} clusters, "
            f"{stats['merged']} merged"
        )

    finally:
        db.close()

    return stats


def _prune_stale_memories(dry_run: bool, cancel_check: Callable[[], bool]) -> dict:
    """
    Remove stale, low-value memories.

    Candidates for pruning:
    - Confidence < min_confidence_threshold
    - Created > stale_days ago
    - Category not in protected list
    - Subject not in protected list
    """
    import maasv
    from maasv.core.db import get_db

    config = maasv.get_config()

    stats = {"found": 0, "pruned": 0, "candidates": []}
    db = get_db()

    try:
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=config.stale_days)).strftime("%Y-%m-%d %H:%M:%S")

        # Find stale, low-confidence memories
        candidates = db.execute("""
            SELECT id, content, category, subject, confidence, created_at
            FROM memories
            WHERE superseded_by IS NULL
            AND confidence < ?
            AND created_at < ?
        """, (config.min_confidence_threshold, cutoff_date)).fetchall()

        candidates = [dict(c) for c in candidates]
        logger.info(f"[MemoryHygiene] Found {len(candidates)} prune candidates")

        for mem in candidates:
            if cancel_check():
                break

            if _is_protected(mem):
                continue

            stats["found"] += 1
            stats["candidates"].append({
                "id": mem["id"],
                "content": mem["content"][:100],
                "category": mem["category"],
                "confidence": mem["confidence"],
                "age_days": (datetime.now(timezone.utc) - datetime.fromisoformat(mem["created_at"])).days
            })

            if not dry_run:
                # Actually delete
                db.execute("DELETE FROM memory_vectors WHERE id = ?", (mem["id"],))
                db.execute("DELETE FROM memories WHERE id = ?", (mem["id"],))
                stats["pruned"] += 1

        if not dry_run:
            db.commit()

    finally:
        db.close()

    return stats


def _deduplicate_relationships(dry_run: bool) -> dict:
    """
    Remove duplicate relationships — same (subject_id, predicate, object_id) or
    (subject_id, predicate, object_value) with multiple active rows.

    Keeps the row with highest confidence per group, deletes the rest.
    """
    from maasv.core.db import get_db

    stats = {"found": 0, "removed": 0}
    db = get_db()

    try:
        # Entity-to-entity duplicates
        groups = db.execute("""
            SELECT subject_id, predicate, object_id, count(*) as cnt
            FROM relationships
            WHERE valid_to IS NULL AND object_id IS NOT NULL
            GROUP BY subject_id, predicate, object_id
            HAVING count(*) > 1
        """).fetchall()

        stats["found"] += len(groups)

        for g in groups:
            rows = db.execute("""
                SELECT id, confidence FROM relationships
                WHERE subject_id = ? AND predicate = ? AND object_id = ?
                AND valid_to IS NULL
                ORDER BY confidence DESC, created_at ASC
            """, (g["subject_id"], g["predicate"], g["object_id"])).fetchall()

            if len(rows) < 2:
                continue

            to_delete = [row["id"] for row in rows[1:]]
            if not dry_run:
                placeholders = ",".join("?" * len(to_delete))
                db.execute(f"DELETE FROM relationships WHERE id IN ({placeholders})", to_delete)
            stats["removed"] += len(to_delete)

        # Attribute-value duplicates
        val_groups = db.execute("""
            SELECT subject_id, predicate, object_value, count(*) as cnt
            FROM relationships
            WHERE valid_to IS NULL AND object_id IS NULL AND object_value IS NOT NULL
            GROUP BY subject_id, predicate, object_value
            HAVING count(*) > 1
        """).fetchall()

        stats["found"] += len(val_groups)

        for g in val_groups:
            rows = db.execute("""
                SELECT id, confidence FROM relationships
                WHERE subject_id = ? AND predicate = ? AND object_value = ?
                AND valid_to IS NULL AND object_id IS NULL
                ORDER BY confidence DESC, created_at ASC
            """, (g["subject_id"], g["predicate"], g["object_value"])).fetchall()

            if len(rows) < 2:
                continue

            to_delete = [row["id"] for row in rows[1:]]
            if not dry_run:
                placeholders = ",".join("?" * len(to_delete))
                db.execute(f"DELETE FROM relationships WHERE id IN ({placeholders})", to_delete)
            stats["removed"] += len(to_delete)

        if not dry_run:
            db.commit()

    finally:
        db.close()

    return stats


def _deduplicate_entities(dry_run: bool) -> dict:
    """
    Find and merge near-duplicate entities using normalization-based matching.

    Conservative: only uses normalize_entity_name() to find exact matches after
    normalization (hyphens→underscores, strip plurals/domains/qualifiers).
    No fuzzy matching — that's too risky without human review.

    Groups entities by type, then by normalized name within each type.
    For each group of 2+, selects keeper (most relationships → shortest name →
    highest access → most recent) and merges the rest via merge_entity().

    Returns:
        {"found": int, "merged": int, "clusters": int}
    """
    from maasv.core.db import get_db
    from maasv.core.graph import merge_entity, normalize_entity_name

    stats = {"found": 0, "merged": 0, "clusters": 0}
    db = get_db()

    try:
        # Load all entities
        all_entities = [dict(r) for r in db.execute(
            "SELECT id, name, entity_type, canonical_name, access_count, created_at "
            "FROM entities"
        ).fetchall()]

        if len(all_entities) < 2:
            return stats

        # Get relationship counts per entity
        rel_count_rows = db.execute("""
            SELECT entity_id, SUM(cnt) as total FROM (
                SELECT subject_id as entity_id, COUNT(*) as cnt
                FROM relationships WHERE valid_to IS NULL GROUP BY subject_id
                UNION ALL
                SELECT object_id as entity_id, COUNT(*) as cnt
                FROM relationships WHERE valid_to IS NULL AND object_id IS NOT NULL
                GROUP BY object_id
            ) GROUP BY entity_id
        """).fetchall()
        rel_counts = {r["entity_id"]: r["total"] for r in rel_count_rows}

        # Group by (entity_type, normalized_name) to find duplicates
        from collections import defaultdict
        groups = defaultdict(list)
        for ent in all_entities:
            norm = normalize_entity_name(ent["canonical_name"])
            key = (ent["entity_type"], norm)
            groups[key].append(ent)

        # Process groups with 2+ members
        for (etype, norm_name), members in groups.items():
            if len(members) < 2:
                continue

            stats["clusters"] += 1
            stats["found"] += len(members) - 1  # all but keeper

            if dry_run:
                names = [m["canonical_name"] for m in members]
                logger.info(
                    f"[MemoryHygiene] Entity dedup (dry): {etype}/{norm_name} "
                    f"→ {len(members)} members: {names}"
                )
                continue

            # Keeper selection: most rels → shortest name → highest access → most recent
            members.sort(key=lambda m: (
                rel_counts.get(m["id"], 0),
                -len(m.get("canonical_name") or ""),
                m.get("access_count") or 0,
                m.get("created_at") or "",
            ), reverse=True)

            keeper = members[0]
            dup_ids = [m["id"] for m in members[1:]]

            try:
                merge_stats = merge_entity(keeper["id"], dup_ids)
                stats["merged"] += merge_stats["entities_deleted"]
                logger.info(
                    f"[MemoryHygiene] Entity dedup: merged {dup_ids} into "
                    f"{keeper['canonical_name']} ({merge_stats['entities_deleted']} deleted, "
                    f"{merge_stats['relationships_updated']} rels updated)"
                )
            except Exception as e:
                logger.error(
                    f"[MemoryHygiene] Entity dedup: failed merging into "
                    f"{keeper['canonical_name']}: {e}", exc_info=True
                )

    finally:
        db.close()

    return stats


def _consolidate_clusters(dry_run: bool, cancel_check: Callable[[], bool]) -> dict:
    """
    Find clusters of related memories and consolidate into stronger single memories.

    This is expensive (O(n^2) embedding comparisons) so only runs in full mode.
    Clusters are memories with:
    - Same subject
    - Similarity > cluster_similarity threshold
    """
    import struct
    import maasv
    from maasv.core.db import get_db
    from maasv.core.store import store_memory

    config = maasv.get_config()

    stats = {"found": 0, "consolidated": 0, "clusters": []}
    db = get_db()

    try:
        # Get memories grouped by subject (only those with subjects), capped
        memories = db.execute("""
            SELECT id, content, category, subject, confidence, metadata, created_at
            FROM memories
            WHERE superseded_by IS NULL
            AND subject IS NOT NULL
            AND subject != ''
            ORDER BY subject, created_at DESC
            LIMIT ?
        """, (MAX_HYGIENE_MEMORIES,)).fetchall()

        memories = [dict(m) for m in memories]

        # Pre-load embeddings from DB instead of recomputing (O(n) reads vs O(n²) API calls)
        embedding_cache = {}
        for mem in memories:
            row = db.execute(
                "SELECT embedding FROM memory_vectors WHERE id = ?", (mem["id"],)
            ).fetchone()
            if row and row["embedding"]:
                # Deserialize float32 binary blob
                blob = row["embedding"]
                embedding_cache[mem["id"]] = struct.unpack(f'{len(blob)//4}f', blob)

        # Group by subject
        by_subject = {}
        for mem in memories:
            subj = mem["subject"].lower()
            if subj not in by_subject:
                by_subject[subj] = []
            by_subject[subj].append(mem)

        for subject, mems in by_subject.items():
            if cancel_check():
                break

            if len(mems) < 3:  # Need at least 3 to form a meaningful cluster
                continue

            # Find clusters using vector similarity
            # Simple greedy clustering
            clusters = []
            used = set()

            for mem in mems:
                if cancel_check():
                    break
                if mem["id"] in used:
                    continue
                if mem["id"] not in embedding_cache:
                    continue

                cluster = [mem]
                used.add(mem["id"])

                query_embedding = embedding_cache[mem["id"]]

                for other in mems:
                    if other["id"] in used:
                        continue
                    if other["id"] not in embedding_cache:
                        continue

                    other_embedding = embedding_cache[other["id"]]
                    # Cosine similarity (assuming normalized)
                    similarity = sum(a * b for a, b in zip(query_embedding, other_embedding))

                    if similarity > config.cluster_similarity:
                        cluster.append(other)
                        used.add(other["id"])

                if len(cluster) >= 3:
                    clusters.append(cluster)
                    stats["found"] += 1

            # Consolidate clusters
            for cluster in clusters:
                if cancel_check():
                    break

                # Create consolidated content
                contents = [m["content"] for m in cluster]
                consolidated_content = _summarize_cluster(contents, subject)

                if not consolidated_content:
                    continue

                # Get highest confidence
                max_confidence = max(m.get("confidence", 1.0) for m in cluster)
                category = cluster[0]["category"]

                stats["clusters"].append({
                    "subject": subject,
                    "count": len(cluster),
                    "ids": [m["id"] for m in cluster],
                    "consolidated": consolidated_content[:200]
                })

                if not dry_run:
                    # Create new consolidated memory
                    new_id = store_memory(
                        content=consolidated_content,
                        category=category,
                        subject=subject.title(),
                        source="consolidation",
                        confidence=max_confidence,
                        metadata={"consolidated_from": [m["id"] for m in cluster]}
                    )

                    # Mark old ones as superseded (unless protected)
                    for mem in cluster:
                        if not _is_protected(mem):
                            db.execute("""
                                UPDATE memories
                                SET superseded_by = ?, updated_at = CURRENT_TIMESTAMP
                                WHERE id = ?
                            """, (new_id, mem["id"]))

                    stats["consolidated"] += 1

        if not dry_run:
            db.commit()

    finally:
        db.close()

    return stats


def _summarize_cluster(contents: list[str], subject: str) -> Optional[str]:
    """
    Create a summary of clustered memories.

    For now, just concatenate unique facts. Could use LLM for smarter summarization.
    """
    # Deduplicate very similar content
    unique = []
    for content in contents:
        content_lower = content.lower().strip()
        is_dup = False
        for existing in unique:
            if content_lower in existing.lower() or existing.lower() in content_lower:
                is_dup = True
                break
        if not is_dup:
            unique.append(content)

    if len(unique) == 1:
        return unique[0]

    # Simple concatenation for now
    return f"[Consolidated] About {subject}: " + " | ".join(unique[:5])


def _log_hygiene_run(stats: HygieneStats):
    """Log hygiene run results for audit."""
    import maasv

    config = maasv.get_config()
    log_path = config.hygiene_log_path

    if not log_path:
        logger.debug("[MemoryHygiene] No hygiene_log_path configured, skipping log file")
        return

    try:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if log_path.exists():
            with open(log_path) as f:
                log = json.load(f)
        else:
            log = {"runs": []}

        log["runs"].append(_stats_to_dict(stats))

        # Keep last 100 runs
        log["runs"] = log["runs"][-100:]

        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

    except Exception as e:
        logger.error(f"[MemoryHygiene] Failed to log run: {e}")


def _stats_to_dict(stats: HygieneStats) -> dict:
    """Convert HygieneStats to dict for JSON serialization."""
    return {
        "duplicates_found": stats.duplicates_found,
        "duplicates_merged": stats.duplicates_merged,
        "stale_found": stats.stale_found,
        "stale_pruned": stats.stale_pruned,
        "rel_duplicates_found": stats.rel_duplicates_found,
        "rel_duplicates_removed": stats.rel_duplicates_removed,
        "entity_dupes_found": stats.entity_dupes_found,
        "entity_dupes_merged": stats.entity_dupes_merged,
        "clusters_found": stats.clusters_found,
        "clusters_consolidated": stats.clusters_consolidated,
        "errors": stats.errors,
        "dry_run": stats.dry_run,
        "backup_path": stats.backup_path,
        "started_at": stats.started_at,
        "completed_at": stats.completed_at
    }


# Convenience function for manual runs
def run_hygiene(
    mode: str = "incremental",
    dry_run: bool = True,
    dedup: bool = True,
    prune: bool = True,
    consolidate: bool = False
) -> dict:
    """
    Run memory hygiene manually (not as a sleep job).

    Args:
        mode: "incremental" or "full"
        dry_run: If True, only report what would be done
        dedup: Run deduplication
        prune: Run stale pruning
        consolidate: Run cluster consolidation (expensive)

    Returns:
        Stats dict
    """
    return run_memory_hygiene_job(
        data={
            "mode": mode,
            "dry_run": dry_run,
            "dedup": dedup,
            "prune": prune,
            "consolidate": consolidate
        },
        cancel_check=lambda: False
    )

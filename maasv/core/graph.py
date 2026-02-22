"""
maasv Knowledge Graph

Entity CRUD, relationship management, graph queries, entity profiles.
All graph memory operations live here.
"""

import logging
import re
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

from maasv.core.db import _db, _record_entity_access, _escape_like

logger = logging.getLogger(__name__)

# Entity name character allowlist: alphanumeric, spaces, hyphens, underscores, periods, apostrophes
_ENTITY_NAME_RE = re.compile(r"[^a-zA-Z0-9 \-_.']+")
MAX_ENTITY_NAME_LENGTH = 200
MAX_OBJECT_VALUE_LENGTH = 2000

# Valid predicates — union of all predicates used across the codebase
VALID_PREDICATES = {
    # Location
    "located_in", "lives_in", "visited", "located_at",
    # Projects
    "works_on", "manages", "created", "owns",
    # Organization
    "works_at",
    # Technology
    "uses_tech", "built_with", "runs_on", "hosted_on", "depends_on", "written_in",
    # Family / social
    "parent_of", "child_of", "married_to", "sibling_of", "friend_of",
    "works_with", "colleague_of", "spouse", "child", "sibling",
    # Attributes
    "has_email", "has_phone", "has_birthday", "has_age",
    # Causal
    "caused_by", "led_to", "resulted_in", "motivated_by",
    "enabled_by", "blocked_by", "chose_over",
    # Inference
    "has_reference", "inferred_as",
    # Integration / usage
    "integrates_with", "integrated_via", "used_for", "uses",
    "monitors", "integrates",
    # Ownership / property
    "owns_pet", "has_property_in",
    # Social
    "interested_in", "collaborates_with",
}


def _sanitize_entity_name(name: str) -> str:
    """Sanitize entity name to allowed characters only.

    Strips characters not in [a-zA-Z0-9 \\-_.']. Collapses whitespace.
    Truncates to MAX_ENTITY_NAME_LENGTH.
    Raises ValueError if result is empty or single-character.
    """
    sanitized = _ENTITY_NAME_RE.sub("", name).strip()
    sanitized = re.sub(r"\s+", " ", sanitized)
    if len(sanitized) < 2:
        raise ValueError(f"Entity name too short after sanitization: {name!r} -> {sanitized!r}")
    return sanitized[:MAX_ENTITY_NAME_LENGTH]


def _clamp_confidence(value, default: float = 0.5) -> float:
    """Clamp confidence to [0.0, 1.0]. Coerce non-numeric to default."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, result))


# ============================================================================
# ENTITY OPERATIONS
# ============================================================================

def create_entity(
    name: str,
    entity_type: str,
    canonical_name: Optional[str] = None,
    metadata: Optional[dict] = None
) -> str:
    """Create a new entity in the knowledge graph.

    If an entity with the same (canonical_name, entity_type) already exists
    (race condition with find_or_create_entity), returns the existing entity's ID.
    """
    name = _sanitize_entity_name(name)
    entity_id = f"ent_{uuid.uuid4().hex[:12]}"

    if canonical_name is None:
        canonical_name = name.lower().strip().replace(" ", "_")

    with _db() as db:
        try:
            db.execute("""
                INSERT INTO entities (id, name, entity_type, canonical_name, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                entity_id, name, entity_type, canonical_name,
                json.dumps(metadata) if metadata else None
            ))
            db.commit()
        except sqlite3.IntegrityError:
            # Unique constraint on (canonical_name, entity_type) — return existing
            row = db.execute(
                "SELECT id FROM entities WHERE canonical_name = ? AND entity_type = ?",
                (canonical_name, entity_type)
            ).fetchone()
            if row:
                return row["id"]
            raise  # Different constraint violation, re-raise
    return entity_id


def get_entity(entity_id: str) -> Optional[dict]:
    """Get an entity by ID."""
    with _db() as db:
        row = db.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()

    if row:
        result = dict(row)
        if result.get('metadata'):
            result['metadata'] = json.loads(result['metadata'])
        return result
    return None


def find_entity_by_name(name: str, entity_type: Optional[str] = None) -> Optional[dict]:
    """Find an entity by name (case-insensitive)."""
    canonical = name.lower().strip().replace(" ", "_")

    with _db() as db:
        if entity_type:
            row = db.execute(
                "SELECT * FROM entities WHERE canonical_name = ? AND entity_type = ?",
                (canonical, entity_type)
            ).fetchone()
        else:
            row = db.execute(
                "SELECT * FROM entities WHERE canonical_name = ?",
                (canonical,)
            ).fetchone()

        if row:
            result = dict(row)
            _record_entity_access(db, [result['id']])
            if result.get('metadata'):
                result['metadata'] = json.loads(result['metadata'])
            return result
    return None


def normalize_entity_name(canonical_name: str) -> str:
    """
    Normalize a canonical_name for duplicate detection.

    Used by hygiene cycle (entity dedup) and write-time prevention.

    Steps:
    1. Lowercase + strip
    2. Replace hyphens with underscores
    3. Strip parenthetical qualifiers: "foo_(bar_baz)" -> "foo"
    4. Strip domain suffixes (.sh, .dev, .js, .io, .ai, .py, etc.)
    5. Strip trailing "s" if len > 4 (basic depluralization)
    """
    name = canonical_name.lower().strip()
    name = name.replace("-", "_")
    name = re.sub(r"_?\(.*?\)$", "", name)

    for suffix in (".sh", ".dev", ".js", ".io", ".ai", ".py", ".rs", ".go"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break

    if len(name) > 4 and name.endswith("s") and not name.endswith("ss"):
        name = name[:-1]

    return name


def find_or_create_entity(
    name: str,
    entity_type: str,
    metadata: Optional[dict] = None
) -> str:
    """
    Find existing entity or create new one. Returns entity ID.

    Lookup order:
    1. Exact canonical_name match (fast, existing behavior)
    2. Normalized name match within same entity_type (prevents near-duplicates
       like "react-native" when "react_native" already exists)
    """
    name = _sanitize_entity_name(name)

    # 1. Exact match (any type)
    existing = find_entity_by_name(name)
    if existing:
        return existing['id']

    # 2. Normalized match (same type only — don't merge across types)
    # NOTE: O(n) scan of all entities for this type. Fine at <1K entities.
    # For 10K+, add a normalized_name column with an index.
    incoming_norm = normalize_entity_name(name.lower().strip().replace(" ", "_"))
    with _db() as db:
        candidates = db.execute(
            "SELECT id, canonical_name FROM entities WHERE entity_type = ?",
            (entity_type,)
        ).fetchall()
        for row in candidates:
            if normalize_entity_name(row["canonical_name"]) == incoming_norm:
                _record_entity_access(db, [row["id"]])
                logger.debug(
                    f"[find_or_create_entity] Normalized match: '{name}' -> "
                    f"existing '{row['canonical_name']}' (type={entity_type})"
                )
                return row["id"]

    return create_entity(name, entity_type, metadata=metadata)


def merge_entity(keeper_id: str, duplicate_ids: list[str]) -> dict:
    """
    Merge duplicate entities into a single keeper entity.

    Reassigns all relationships from duplicates to keeper, transfers access_count,
    merges metadata, and deletes the duplicates. FTS triggers handle cleanup.

    Args:
        keeper_id: Entity ID to keep
        duplicate_ids: List of entity IDs to merge into keeper

    Returns:
        Stats dict: {relationships_updated, entities_deleted, rel_dupes_removed}
    """
    if not duplicate_ids:
        return {"relationships_updated": 0, "entities_deleted": 0, "rel_dupes_removed": 0}

    # Guard against keeper being in the duplicate list (would delete the keeper)
    duplicate_ids = [d for d in duplicate_ids if d != keeper_id]
    if not duplicate_ids:
        return {"relationships_updated": 0, "entities_deleted": 0, "rel_dupes_removed": 0}

    stats = {"relationships_updated": 0, "entities_deleted": 0, "rel_dupes_removed": 0}

    with _db() as db:
        # Verify keeper exists
        keeper = db.execute("SELECT * FROM entities WHERE id = ?", (keeper_id,)).fetchone()
        if not keeper:
            raise ValueError(f"Keeper entity {keeper_id} not found")
        keeper = dict(keeper)

        # Verify all duplicates exist
        all_dup_ids = []
        for dup_id in duplicate_ids:
            dup = db.execute("SELECT * FROM entities WHERE id = ?", (dup_id,)).fetchone()
            if dup:
                all_dup_ids.append(dup_id)
            else:
                logger.warning(f"Duplicate entity {dup_id} not found, skipping")

        if not all_dup_ids:
            return stats

        placeholders = ",".join("?" * len(all_dup_ids))

        # 1. Reassign subject_id relationships
        result = db.execute(
            f"UPDATE relationships SET subject_id = ? WHERE subject_id IN ({placeholders})",
            [keeper_id] + all_dup_ids
        )
        stats["relationships_updated"] += result.rowcount

        # 2. Reassign object_id relationships
        result = db.execute(
            f"UPDATE relationships SET object_id = ? WHERE object_id IN ({placeholders})",
            [keeper_id] + all_dup_ids
        )
        stats["relationships_updated"] += result.rowcount

        # 3. Transfer access_count — keeper gets max across all
        all_ids = [keeper_id] + all_dup_ids
        all_placeholders = ",".join("?" * len(all_ids))
        max_access_row = db.execute(
            f"SELECT MAX(COALESCE(access_count, 0)) as max_ac FROM entities WHERE id IN ({all_placeholders})",
            all_ids
        ).fetchone()
        max_access = max_access_row["max_ac"] or 0

        if max_access > (keeper.get("access_count") or 0):
            db.execute(
                "UPDATE entities SET access_count = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (max_access, keeper_id)
            )

        # 4. Merge metadata — keeper wins conflicts, preserve unique keys from duplicates
        keeper_meta = json.loads(keeper["metadata"]) if keeper.get("metadata") else {}
        if not isinstance(keeper_meta, dict):
            keeper_meta = {}

        for dup_id in all_dup_ids:
            dup = db.execute("SELECT metadata FROM entities WHERE id = ?", (dup_id,)).fetchone()
            if dup and dup["metadata"]:
                dup_meta = json.loads(dup["metadata"])
                if isinstance(dup_meta, dict):
                    # Duplicate keys fill in gaps, keeper wins conflicts
                    merged = {**dup_meta, **keeper_meta}
                    keeper_meta = merged

        if keeper_meta:
            db.execute(
                "UPDATE entities SET metadata = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (json.dumps(keeper_meta), keeper_id)
            )

        # 5. Delete duplicate entities (FTS trigger cleans up automatically)
        db.execute(
            f"DELETE FROM entities WHERE id IN ({placeholders})",
            all_dup_ids
        )
        stats["entities_deleted"] = len(all_dup_ids)

        db.commit()

    # 6. Relationship dedup — merging may have created duplicate triples
    from maasv.lifecycle.memory_hygiene import _deduplicate_relationships
    rel_stats = _deduplicate_relationships(dry_run=False)
    stats["rel_dupes_removed"] = rel_stats["removed"]

    return stats


def search_entities(
    query: str,
    entity_type: Optional[str] = None,
    limit: int = 10
) -> list[dict]:
    """Search entities using FTS."""
    with _db() as db:
        try:
            if entity_type:
                rows = db.execute("""
                    SELECT e.*
                    FROM entities_fts f
                    JOIN entities e ON f.rowid = e.rowid
                    WHERE entities_fts MATCH ?
                    AND e.entity_type = ?
                    ORDER BY rank
                    LIMIT ?
                """, (query, entity_type, limit)).fetchall()
            else:
                rows = db.execute("""
                    SELECT e.*
                    FROM entities_fts f
                    JOIN entities e ON f.rowid = e.rowid
                    WHERE entities_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """, (query, limit)).fetchall()
        except Exception:
            escaped_query = _escape_like(query)
            if entity_type:
                rows = db.execute("""
                    SELECT * FROM entities
                    WHERE name LIKE ? ESCAPE '\\' AND entity_type = ?
                    LIMIT ?
                """, (f"%{escaped_query}%", entity_type, limit)).fetchall()
            else:
                rows = db.execute("""
                    SELECT * FROM entities WHERE name LIKE ? ESCAPE '\\' LIMIT ?
                """, (f"%{escaped_query}%", limit)).fetchall()

    results = []
    for row in rows:
        result = dict(row)
        if result.get('metadata'):
            result['metadata'] = json.loads(result['metadata'])
        results.append(result)

    return results


def get_entities_by_type(entity_type: str, limit: int = 50) -> list[dict]:
    """Get all entities of a given type."""
    with _db() as db:
        rows = db.execute(
            "SELECT * FROM entities WHERE entity_type = ? ORDER BY name LIMIT ?",
            (entity_type, limit)
        ).fetchall()

    results = []
    for row in rows:
        result = dict(row)
        if result.get('metadata'):
            result['metadata'] = json.loads(result['metadata'])
        results.append(result)

    return results


# ============================================================================
# RELATIONSHIP OPERATIONS
# ============================================================================

def add_relationship(
    subject_id: str,
    predicate: str,
    object_id: Optional[str] = None,
    object_value: Optional[str] = None,
    valid_from: Optional[str] = None,
    confidence: float = 1.0,
    source: Optional[str] = None,
    metadata: Optional[dict] = None,
    origin: Optional[str] = None,
    origin_interface: Optional[str] = None,
) -> str:
    """Add a temporal relationship between entities.

    Deduplicates: if an active relationship with the same (subject_id, predicate, object_id)
    or (subject_id, predicate, object_value) already exists, updates confidence if higher
    and returns the existing relationship ID instead of creating a duplicate.
    """
    if object_id is None and object_value is None:
        raise ValueError("Must provide either object_id or object_value")

    # Task 5: Predicate allowlist (extended by config.extra_predicates)
    import maasv
    allowed = VALID_PREDICATES | maasv.get_config().extra_predicates
    if predicate not in allowed:
        raise ValueError(f"Unknown predicate: {predicate!r}")

    # Task 4: Confidence clamping
    confidence = _clamp_confidence(confidence)

    # Task 3: Object value length cap
    if object_value is not None:
        object_value = str(object_value)[:MAX_OBJECT_VALUE_LENGTH]

    # Task 23: Cap metadata JSON size
    if metadata is not None:
        meta_json = json.dumps(metadata)
        if len(meta_json) > 10_000:
            raise ValueError(f"Relationship metadata JSON exceeds 10K chars ({len(meta_json)})")

    if valid_from is None:
        valid_from = datetime.now(timezone.utc).isoformat()

    with _db() as db:
        # Check for existing active relationship with same triple
        if object_id is not None:
            existing = db.execute("""
                SELECT id, confidence FROM relationships
                WHERE subject_id = ? AND predicate = ? AND object_id = ?
                AND valid_to IS NULL
                LIMIT 1
            """, (subject_id, predicate, object_id)).fetchone()
        else:
            existing = db.execute("""
                SELECT id, confidence FROM relationships
                WHERE subject_id = ? AND predicate = ? AND object_value = ?
                AND valid_to IS NULL
                LIMIT 1
            """, (subject_id, predicate, object_value)).fetchone()

        if existing:
            # Already exists — update confidence if new is higher
            if confidence > existing["confidence"]:
                db.execute(
                    "UPDATE relationships SET confidence = ? WHERE id = ?",
                    (confidence, existing["id"])
                )
                db.commit()
            return existing["id"]

        # No existing match — insert new
        rel_id = f"rel_{uuid.uuid4().hex[:12]}"
        try:
            db.execute("""
                INSERT INTO relationships
                (id, subject_id, predicate, object_id, object_value, valid_from, confidence, source, metadata, origin, origin_interface)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rel_id, subject_id, predicate, object_id, object_value,
                valid_from, confidence, source,
                json.dumps(metadata) if metadata else None,
                origin, origin_interface,
            ))
            db.commit()
        except sqlite3.IntegrityError:
            # Partial unique index conflict — another thread created the same active relationship
            if object_id is not None:
                row = db.execute("""
                    SELECT id FROM relationships
                    WHERE subject_id = ? AND predicate = ? AND object_id = ?
                    AND valid_to IS NULL
                    LIMIT 1
                """, (subject_id, predicate, object_id)).fetchone()
            else:
                row = db.execute("""
                    SELECT id FROM relationships
                    WHERE subject_id = ? AND predicate = ? AND object_value = ?
                    AND valid_to IS NULL
                    LIMIT 1
                """, (subject_id, predicate, object_value)).fetchone()
            if row:
                return row["id"]
            raise  # Different constraint violation, re-raise
    return rel_id


def expire_relationship(
    relationship_id: str,
    valid_to: Optional[str] = None
) -> bool:
    """Mark a relationship as expired (no longer current)."""
    if valid_to is None:
        valid_to = datetime.now(timezone.utc).isoformat()

    with _db() as db:
        cursor = db.execute(
            "UPDATE relationships SET valid_to = ? WHERE id = ? AND valid_to IS NULL",
            (valid_to, relationship_id)
        )
        updated = cursor.rowcount > 0
        db.commit()
    return updated


def get_entity_relationships(
    entity_id: str,
    include_expired: bool = False,
    predicate: Optional[str] = None,
    direction: str = "both"
) -> list[dict]:
    """Get all relationships for an entity."""
    results = []
    queries = []
    params_list = []

    if direction in ("outgoing", "both"):
        query = """
            SELECT r.*,
                   e_subj.name as subject_name, e_subj.entity_type as subject_type,
                   e_obj.name as object_name, e_obj.entity_type as object_type
            FROM relationships r
            JOIN entities e_subj ON r.subject_id = e_subj.id
            LEFT JOIN entities e_obj ON r.object_id = e_obj.id
            WHERE r.subject_id = ?
        """
        params = [entity_id]
        if not include_expired:
            query += " AND r.valid_to IS NULL"
        if predicate:
            query += " AND r.predicate = ?"
            params.append(predicate)
        queries.append(query)
        params_list.append(params)

    if direction in ("incoming", "both"):
        query = """
            SELECT r.*,
                   e_subj.name as subject_name, e_subj.entity_type as subject_type,
                   e_obj.name as object_name, e_obj.entity_type as object_type
            FROM relationships r
            JOIN entities e_subj ON r.subject_id = e_subj.id
            LEFT JOIN entities e_obj ON r.object_id = e_obj.id
            WHERE r.object_id = ?
        """
        params = [entity_id]
        if not include_expired:
            query += " AND r.valid_to IS NULL"
        if predicate:
            query += " AND r.predicate = ?"
            params.append(predicate)
        queries.append(query)
        params_list.append(params)

    seen_ids = set()
    with _db() as db:
        for query, params in zip(queries, params_list):
            rows = db.execute(query, params).fetchall()
            for row in rows:
                row_dict = dict(row)
                if row_dict['id'] not in seen_ids:
                    if row_dict.get('metadata'):
                        row_dict['metadata'] = json.loads(row_dict['metadata'])
                    results.append(row_dict)
                    seen_ids.add(row_dict['id'])

        # Track access on the queried entity itself
        _record_entity_access(db, [entity_id])

    return results


# Causal predicate sets for chain traversal
_FORWARD_CAUSAL = {"led_to", "resulted_in", "enabled_by"}
_BACKWARD_CAUSAL = {"caused_by", "motivated_by", "blocked_by"}
_ALL_CAUSAL = _FORWARD_CAUSAL | _BACKWARD_CAUSAL | {"chose_over"}


def get_causal_chain(
    entity_id: str,
    direction: str = "both",
    max_hops: int = 3
) -> list[dict]:
    """
    Traverse causal edges from an entity.

    Args:
        entity_id: Starting entity
        direction: "forward" (led_to, resulted_in, enabled_by),
                   "backward" (caused_by, motivated_by, blocked_by),
                   or "both"
        max_hops: Maximum traversal depth

    Returns:
        Ordered list of {"entity": {...}, "relationship": {...}, "hop": int}
    """
    if direction == "forward":
        predicates = _FORWARD_CAUSAL
    elif direction == "backward":
        predicates = _BACKWARD_CAUSAL
    else:
        predicates = _ALL_CAUSAL

    chain = []
    visited = {entity_id}
    frontier = [entity_id]

    with _db() as db:
        for hop in range(1, max_hops + 1):
            next_frontier = []
            for current_id in frontier:
                placeholders = ",".join("?" * len(predicates))
                rows = db.execute(f"""
                    SELECT r.*,
                           e.id as next_entity_id, e.name as next_entity_name,
                           e.entity_type as next_entity_type
                    FROM relationships r
                    JOIN entities e ON (
                        CASE WHEN r.subject_id = ? THEN r.object_id ELSE r.subject_id END
                    ) = e.id
                    WHERE (r.subject_id = ? OR r.object_id = ?)
                    AND r.predicate IN ({placeholders})
                    AND r.valid_to IS NULL
                """, [current_id, current_id, current_id] + list(predicates)).fetchall()

                for row in rows:
                    row_dict = dict(row)
                    next_id = row_dict['next_entity_id']
                    if next_id and next_id not in visited:
                        visited.add(next_id)
                        next_frontier.append(next_id)
                        chain.append({
                            "entity": {
                                "id": next_id,
                                "name": row_dict['next_entity_name'],
                                "type": row_dict['next_entity_type'],
                            },
                            "relationship": {
                                "id": row_dict['id'],
                                "predicate": row_dict['predicate'],
                                "subject_id": row_dict['subject_id'],
                                "object_id": row_dict['object_id'],
                                "confidence": row_dict['confidence'],
                            },
                            "hop": hop,
                        })

            frontier = next_frontier
            if not frontier:
                break

    return chain


def update_relationship_value(
    subject_id: str,
    predicate: str,
    new_value: str,
    source: Optional[str] = None
) -> tuple[Optional[str], str]:
    """Update a relationship by expiring the old one and creating a new one.

    Uses a single connection/transaction: find current -> expire old -> insert new.
    """
    import maasv
    allowed = VALID_PREDICATES | maasv.get_config().extra_predicates
    if predicate not in allowed:
        raise ValueError(f"Unknown predicate: {predicate!r}")

    new_value = str(new_value)[:MAX_OBJECT_VALUE_LENGTH]
    now_iso = datetime.now(timezone.utc).isoformat()

    with _db() as db:
        # 1. Find current active relationship
        current = db.execute("""
            SELECT id FROM relationships
            WHERE subject_id = ? AND predicate = ? AND valid_to IS NULL
        """, (subject_id, predicate)).fetchone()

        old_id = None
        if current:
            old_id = current['id']
            # 2. Expire the old relationship
            db.execute(
                "UPDATE relationships SET valid_to = ? WHERE id = ? AND valid_to IS NULL",
                (now_iso, old_id)
            )

        # 3. Insert the new relationship
        new_id = f"rel_{uuid.uuid4().hex[:12]}"
        db.execute("""
            INSERT INTO relationships
            (id, subject_id, predicate, object_value, valid_from, confidence, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (new_id, subject_id, predicate, new_value, now_iso, 1.0, source))

        db.commit()

    return (old_id, new_id)


def graph_query(
    subject_type: Optional[str] = None,
    predicate: Optional[str] = None,
    object_type: Optional[str] = None,
    include_expired: bool = False,
    limit: int = 50
) -> list[dict]:
    """Query the graph with pattern matching."""
    query = """
        SELECT r.*,
               e_subj.name as subject_name, e_subj.entity_type as subject_type,
               e_obj.name as object_name, e_obj.entity_type as object_type
        FROM relationships r
        JOIN entities e_subj ON r.subject_id = e_subj.id
        LEFT JOIN entities e_obj ON r.object_id = e_obj.id
        WHERE 1=1
    """
    params = []

    if not include_expired:
        query += " AND r.valid_to IS NULL"
    if subject_type:
        query += " AND e_subj.entity_type = ?"
        params.append(subject_type)
    if predicate:
        query += " AND r.predicate = ?"
        params.append(predicate)
    if object_type:
        query += " AND e_obj.entity_type = ?"
        params.append(object_type)

    query += " ORDER BY r.created_at DESC LIMIT ?"
    params.append(limit)

    with _db() as db:
        rows = db.execute(query, params).fetchall()

    results = []
    for row in rows:
        row_dict = dict(row)
        if row_dict.get('metadata'):
            row_dict['metadata'] = json.loads(row_dict['metadata'])
        results.append(row_dict)

    return results


def get_entity_profile(entity_id: str) -> dict:
    """Get a complete profile for an entity including all current relationships."""
    entity = get_entity(entity_id)
    if not entity:
        return {}

    relationships = get_entity_relationships(entity_id, include_expired=False)

    profile = {
        "entity": entity,
        "relationships": {},
        "related_entities": []
    }

    related_ids = set()
    for rel in relationships:
        pred = rel['predicate']
        if pred not in profile['relationships']:
            profile['relationships'][pred] = []

        entry = {
            "id": rel['id'],
            "valid_from": rel['valid_from'],
            "confidence": rel['confidence']
        }

        if rel['object_id']:
            entry['entity_id'] = rel['object_id']
            entry['entity_name'] = rel.get('object_name')
            entry['entity_type'] = rel.get('object_type')
            related_ids.add(rel['object_id'])
        else:
            entry['value'] = rel['object_value']

        profile['relationships'][pred].append(entry)

    for eid in related_ids:
        related = get_entity(eid)
        if related:
            profile['related_entities'].append(related)

    return profile

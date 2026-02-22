"""
maasv Memory Store (slim)

Memory CRUD operations: store, supersede, get, delete, update metadata.
Database infra lives in db.py, retrieval in retrieval.py, graph in graph.py.
"""

import logging
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from maasv.core.db import _db, get_embedding, serialize_embedding

logger = logging.getLogger(__name__)

# Public API input limits
MAX_CONTENT_LENGTH = 50_000
MAX_CATEGORY_LENGTH = 50
MAX_METADATA_JSON_LENGTH = 10_000


def store_memory(
    content: str,
    category: str,
    subject: Optional[str] = None,
    source: str = "manual",
    confidence: float = 1.0,
    metadata: Optional[dict] = None,
    dedup_threshold: float = 0.05,
    origin: Optional[str] = None,
    origin_interface: Optional[str] = None,
) -> str:
    """
    Store a new memory with embedding, with dedup check.

    Args:
        content: The fact or memory to store
        category: Type of memory (family, preference, project, decision, etc.)
        subject: Who/what this is about (e.g., "John", "ProjectX")
        source: Where this came from (manual, conversation, extracted)
        confidence: How confident we are (0.0-1.0)
        metadata: Additional structured data
        dedup_threshold: Vector distance below which a memory is considered duplicate
        origin: What system created this (e.g., "claude", "chatgpt", "salesforce")
        origin_interface: Specific client/interface (e.g., "claude-code", "claude-desktop", "codex")

    Returns:
        Memory ID (existing ID if duplicate found)
    """
    # Task 23: Input validation
    if len(content) > MAX_CONTENT_LENGTH:
        content = content[:MAX_CONTENT_LENGTH]
    if len(category) > MAX_CATEGORY_LENGTH:
        category = category[:MAX_CATEGORY_LENGTH]
    if metadata is not None:
        meta_json = json.dumps(metadata)
        if len(meta_json) > MAX_METADATA_JSON_LENGTH:
            raise ValueError(
                f"Metadata JSON exceeds {MAX_METADATA_JSON_LENGTH} chars ({len(meta_json)})"
            )

    # Task 4: Confidence clamping
    from maasv.core.graph import _clamp_confidence
    confidence = _clamp_confidence(confidence)

    # Compute embedding first (needed for both dedup check and storage)
    embedding = get_embedding(content)

    with _db() as db:
        # Dedup check: find near-duplicate via vector similarity
        try:
            rows = db.execute(
                """
                SELECT v.id, m.content, m.category, distance
                FROM memory_vectors v
                JOIN memories m ON v.id = m.id
                WHERE m.superseded_by IS NULL
                AND v.embedding MATCH ?
                AND k = 3
                ORDER BY distance
                """,
                (serialize_embedding(embedding),)
            ).fetchall()

            for row in rows:
                if row['distance'] < dedup_threshold and row['category'] == category:
                    logger.info(
                        f"Dedup: skipping store, near-duplicate found: {row['id']} (dist={row['distance']:.4f})"
                    )
                    return row['id']
        except Exception:
            logger.debug("Dedup check failed, proceeding with store", exc_info=True)

        # No duplicate found — insert
        memory_id = f"mem_{uuid.uuid4().hex[:12]}"

        db.execute("""
            INSERT INTO memories (id, content, category, subject, source, confidence, metadata, origin, origin_interface)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory_id,
            content,
            category,
            subject,
            source,
            confidence,
            json.dumps(metadata) if metadata else None,
            origin,
            origin_interface,
        ))

        db.execute(
            "INSERT INTO memory_vectors (id, embedding) VALUES (?, ?)",
            (memory_id, serialize_embedding(embedding))
        )

        db.commit()

    return memory_id


def supersede_memory(
    old_id: str,
    new_content: str,
    source: str = "correction",
    force: bool = False,
    origin: Optional[str] = None,
    origin_interface: Optional[str] = None,
) -> str:
    """Mark an old memory as superseded and create a new one.

    Uses a single connection/transaction: read old -> insert new -> update old.
    Rejects superseding memories in protected categories unless force=True.
    Origin fields default to the old memory's values if not provided.
    """
    from maasv.core.graph import _clamp_confidence

    # Compute embedding outside the transaction (CPU work, no DB needed)
    embedding = get_embedding(new_content)

    with _db() as db:
        # 1. Read the old memory
        old = db.execute(
            "SELECT category, subject, metadata, confidence, origin, origin_interface FROM memories WHERE id = ?",
            (old_id,)
        ).fetchone()

        if not old:
            raise ValueError(f"Memory {old_id} not found")

        if not force:
            import maasv
            protected = maasv.get_config().protected_categories
            if old["category"] in protected:
                raise ValueError(
                    f"Cannot supersede memory in protected category '{old['category']}'. "
                    f"Use force=True to override."
                )

        old = dict(old)
        category = old['category']
        subject = old['subject']
        confidence = _clamp_confidence(old.get('confidence', 1.0))
        metadata = json.loads(old['metadata']) if old['metadata'] else None
        # Inherit origin from old memory if not explicitly provided
        if origin is None:
            origin = old.get('origin')
        if origin_interface is None:
            origin_interface = old.get('origin_interface')

        # 2. Dedup check against existing active memories
        new_id = None
        try:
            rows = db.execute(
                """
                SELECT v.id, m.content, m.category, distance
                FROM memory_vectors v
                JOIN memories m ON v.id = m.id
                WHERE m.superseded_by IS NULL
                AND v.embedding MATCH ?
                AND k = 3
                ORDER BY distance
                """,
                (serialize_embedding(embedding),)
            ).fetchall()

            for row in rows:
                if row['distance'] < 0.05 and row['category'] == category:
                    new_id = row['id']
                    break
        except Exception:
            logger.debug("Dedup check failed in supersede, proceeding with store", exc_info=True)

        # 3. Insert new memory if no duplicate found
        if new_id is None:
            new_id = f"mem_{uuid.uuid4().hex[:12]}"
            db.execute("""
                INSERT INTO memories (id, content, category, subject, source, confidence, metadata, origin, origin_interface)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                new_id, new_content, category, subject, source, confidence,
                json.dumps(metadata) if metadata else None,
                origin, origin_interface,
            ))
            db.execute(
                "INSERT INTO memory_vectors (id, embedding) VALUES (?, ?)",
                (new_id, serialize_embedding(embedding))
            )

        # 4. Mark old memory as superseded
        db.execute(
            "UPDATE memories SET superseded_by = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (new_id, old_id)
        )
        db.commit()

    return new_id


def get_all_active(category: Optional[str] = None, limit: Optional[int] = None) -> list[dict]:
    """Get all active (non-superseded) memories.

    Args:
        category: Filter by category (optional)
        limit: Maximum number of memories to return (optional, default unlimited)
    """
    query = "SELECT * FROM memories WHERE superseded_by IS NULL"
    params: list = []

    if category:
        query += " AND category = ?"
        params.append(category)

    query += " ORDER BY created_at DESC"

    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    with _db() as db:
        rows = db.execute(query, params).fetchall()

    return [dict(row) for row in rows]


def get_recent_memories(
    hours: int = 48,
    categories: Optional[list[str]] = None,
    limit: int = 50
) -> list[dict]:
    """Get recent memories from the last N hours."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")

    query = "SELECT * FROM memories WHERE superseded_by IS NULL AND created_at >= ?"
    params: list = [cutoff]

    if categories:
        placeholders = ",".join("?" * len(categories))
        query += f" AND category IN ({placeholders})"
        params.extend(categories)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with _db() as db:
        rows = db.execute(query, params).fetchall()

    return [dict(row) for row in rows]


def delete_memory(memory_id: str, force: bool = False) -> bool:
    """Permanently delete a memory.

    Rejects deletion of memories in protected categories unless force=True.
    """
    with _db() as db:
        if not force:
            row = db.execute(
                "SELECT category FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if row:
                import maasv
                protected = maasv.get_config().protected_categories
                if row["category"] in protected:
                    raise ValueError(
                        f"Cannot delete memory in protected category '{row['category']}'. "
                        f"Use force=True to override."
                    )

        db.execute("DELETE FROM memory_vectors WHERE id = ?", (memory_id,))
        cursor = db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        deleted = cursor.rowcount > 0
        db.commit()

    return deleted


def update_memory_metadata(memory_id: str, metadata_updates: dict) -> bool:
    """Update metadata for an existing memory (merge, not replace)."""
    with _db() as db:
        row = db.execute(
            "SELECT metadata FROM memories WHERE id = ?",
            (memory_id,)
        ).fetchone()

        if not row:
            return False

        current = json.loads(row['metadata']) if row['metadata'] else {}
        current.update(metadata_updates)

        db.execute(
            "UPDATE memories SET metadata = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (json.dumps(current), memory_id)
        )
        db.commit()
    return True

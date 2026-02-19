"""
maasv Database Infrastructure

Connection management, schema initialization, migrations, embedding helpers,
and access tracking. All other core modules import from here for DB access.
"""

import logging
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional, Callable

logger = logging.getLogger(__name__)


def _apply_pragmas(db: sqlite3.Connection):
    """Apply standard SQLite pragmas for safety and concurrency."""
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=5000")
    db.execute("PRAGMA foreign_keys=ON")


def _get_db_path():
    """Get the configured database path."""
    import maasv
    return maasv.get_config().db_path


def _get_embed_dims():
    """Get the configured embedding dimensions."""
    import maasv
    return maasv.get_config().embed_dims


def get_db() -> sqlite3.Connection:
    """Get database connection with sqlite-vec loaded.

    NOTE: Each call opens a new connection and loads sqlite-vec. Functions use
    _db() context manager for auto-close. At scale (>10K memories), consider
    connection pooling to reduce overhead. Current hot path (find_similar_memories)
    reuses a single connection for all 3 signals.
    """
    import sqlite_vec

    db = sqlite3.connect(str(_get_db_path()))
    db.row_factory = sqlite3.Row
    _apply_pragmas(db)

    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    return db


@contextmanager
def _db():
    """Context manager for database connections — ensures close on exception."""
    db = get_db()
    try:
        yield db
    finally:
        db.close()


def get_plain_db() -> sqlite3.Connection:
    """Get database connection WITHOUT sqlite-vec.

    Used by modules that don't need vector search (e.g., wisdom.py).
    """
    db = sqlite3.connect(str(_get_db_path()))
    db.row_factory = sqlite3.Row
    _apply_pragmas(db)
    return db


@contextmanager
def _plain_db():
    """Context manager for plain database connections (no sqlite-vec)."""
    db = get_plain_db()
    try:
        yield db
    finally:
        db.close()


def _ensure_migration_table(db: sqlite3.Connection):
    """Create the schema_migrations tracking table if it doesn't exist."""
    db.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            description TEXT NOT NULL,
            applied_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.commit()


def run_migration(db: sqlite3.Connection, version: int, description: str, migrate_fn: Callable[[sqlite3.Connection], None]):
    """
    Run a schema migration if it hasn't been applied yet.

    Checks schema_migrations for the version. If not present, runs migrate_fn
    inside a transaction and records the version. If already applied, skips silently.

    Args:
        db: Open database connection (with sqlite-vec loaded)
        version: Integer migration version (must be unique, monotonically increasing)
        description: Human-readable description of what this migration does
        migrate_fn: Callable that takes a db connection and performs the migration
    """
    existing = db.execute(
        "SELECT version FROM schema_migrations WHERE version = ?", (version,)
    ).fetchone()
    if existing:
        return

    logger.info(f"Running migration {version}: {description}")
    try:
        migrate_fn(db)
        db.execute(
            "INSERT INTO schema_migrations (version, description) VALUES (?, ?)",
            (version, description)
        )
        db.commit()
        logger.info(f"Migration {version} applied successfully")
    except Exception:
        db.rollback()
        logger.error(f"Migration {version} failed, rolled back", exc_info=True)
        raise


def init_db():
    """Initialize database schema."""
    db = get_db()
    embed_dims = _get_embed_dims()

    # Core memories table
    db.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            category TEXT NOT NULL,
            subject TEXT,
            source TEXT NOT NULL DEFAULT 'manual',
            confidence REAL DEFAULT 1.0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            superseded_by TEXT,
            metadata TEXT
        )
    """)

    # Vector embeddings table (sqlite-vec)
    db.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors USING vec0(
            id TEXT PRIMARY KEY,
            embedding FLOAT[{embed_dims}]
        )
    """)

    # === GRAPH MEMORY TABLES ===
    db.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            canonical_name TEXT,
            metadata TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS relationships (
            id TEXT PRIMARY KEY,
            subject_id TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object_id TEXT,
            object_value TEXT,
            valid_from TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            valid_to TEXT,
            confidence REAL DEFAULT 1.0,
            source TEXT,
            metadata TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (subject_id) REFERENCES entities(id)
        )
    """)

    # Indexes for efficient graph queries
    db.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_relationships_subject ON relationships(subject_id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_relationships_object ON relationships(object_id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_relationships_predicate ON relationships(predicate)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_relationships_valid ON relationships(valid_to)")

    # Full-text search for entities
    db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
            name,
            entity_type,
            canonical_name,
            content='entities',
            content_rowid='rowid'
        )
    """)

    # Triggers for entity FTS sync
    db.execute("""
        CREATE TRIGGER IF NOT EXISTS entities_ai AFTER INSERT ON entities BEGIN
            INSERT INTO entities_fts(rowid, name, entity_type, canonical_name)
            VALUES (NEW.rowid, NEW.name, NEW.entity_type, NEW.canonical_name);
        END
    """)

    db.execute("""
        CREATE TRIGGER IF NOT EXISTS entities_ad AFTER DELETE ON entities BEGIN
            INSERT INTO entities_fts(entities_fts, rowid, name, entity_type, canonical_name)
            VALUES ('delete', OLD.rowid, OLD.name, OLD.entity_type, OLD.canonical_name);
        END
    """)

    db.execute("""
        CREATE TRIGGER IF NOT EXISTS entities_au AFTER UPDATE ON entities BEGIN
            INSERT INTO entities_fts(entities_fts, rowid, name, entity_type, canonical_name)
            VALUES ('delete', OLD.rowid, OLD.name, OLD.entity_type, OLD.canonical_name);
            INSERT INTO entities_fts(rowid, name, entity_type, canonical_name)
            VALUES (NEW.rowid, NEW.name, NEW.entity_type, NEW.canonical_name);
        END
    """)

    # Full-text search for memories
    db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            content,
            category,
            subject,
            content='memories',
            content_rowid='rowid'
        )
    """)

    # Triggers to keep FTS in sync
    db.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content, category, subject)
            VALUES (NEW.rowid, NEW.content, NEW.category, NEW.subject);
        END
    """)

    db.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, category, subject)
            VALUES ('delete', OLD.rowid, OLD.content, OLD.category, OLD.subject);
        END
    """)

    db.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, category, subject)
            VALUES ('delete', OLD.rowid, OLD.content, OLD.category, OLD.subject);
            INSERT INTO memories_fts(rowid, content, category, subject)
            VALUES (NEW.rowid, NEW.content, NEW.category, NEW.subject);
        END
    """)

    db.commit()

    # --- Migration tracking and schema migrations ---
    _ensure_migration_table(db)

    # Version 0: Mark the Phase 0 baseline as tracked
    run_migration(db, 0, "Phase 0 baseline", lambda _db: None)

    # Version 1: Bitemporal columns
    def _migrate_bitemporal(db: sqlite3.Connection):
        db.execute("ALTER TABLE memories ADD COLUMN ingested_at TEXT DEFAULT NULL")
        db.execute("UPDATE memories SET ingested_at = created_at WHERE ingested_at IS NULL")
        db.execute("ALTER TABLE relationships ADD COLUMN ingested_at TEXT DEFAULT NULL")
        db.execute("UPDATE relationships SET ingested_at = created_at WHERE ingested_at IS NULL")
        db.execute("ALTER TABLE relationships ADD COLUMN change_reason TEXT DEFAULT NULL")

    run_migration(db, 1, "Bitemporal columns (ingested_at, change_reason)", _migrate_bitemporal)

    # Version 2: Importance scoring + decay columns
    def _migrate_importance(db: sqlite3.Connection):
        db.execute("ALTER TABLE memories ADD COLUMN importance REAL DEFAULT 0.5")
        db.execute("ALTER TABLE memories ADD COLUMN access_count INTEGER DEFAULT 0")
        db.execute("ALTER TABLE memories ADD COLUMN last_accessed_at TEXT DEFAULT NULL")
        # Backfill: family/identity = 1.0, decision = 0.8
        db.execute("UPDATE memories SET importance = 1.0 WHERE category IN ('family', 'identity')")
        db.execute("UPDATE memories SET importance = 0.8 WHERE category = 'decision'")

    run_migration(db, 2, "Importance scoring + decay columns", _migrate_importance)

    # Version 3: Entity access tracking columns
    # Previously added lazily by reorganize.py, but _record_entity_access
    # and merge_entity depend on them.
    def _migrate_entity_access(db: sqlite3.Connection):
        # These columns may already exist (previously added lazily by reorganize.py).
        # Only suppress the specific "duplicate column" error.
        for col_def in [
            "ALTER TABLE entities ADD COLUMN access_count INTEGER DEFAULT 0",
            "ALTER TABLE entities ADD COLUMN last_accessed_at TEXT DEFAULT NULL",
        ]:
            try:
                db.execute(col_def)
            except sqlite3.OperationalError as e:
                if "duplicate column" in str(e).lower():
                    pass  # Already exists, expected
                else:
                    raise

    run_migration(db, 3, "Entity access tracking columns", _migrate_entity_access)

    # Version 4: Dedup constraints — unique indexes on entities and active relationships
    def _migrate_dedup_constraints(db: sqlite3.Connection):
        # --- Clean up existing entity duplicates (same canonical_name + entity_type) ---
        dup_groups = db.execute("""
            SELECT canonical_name, entity_type
            FROM entities
            GROUP BY canonical_name, entity_type
            HAVING COUNT(*) > 1
        """).fetchall()

        for g in dup_groups:
            rows = db.execute("""
                SELECT id FROM entities
                WHERE canonical_name = ? AND entity_type = ?
                ORDER BY rowid ASC
            """, (g["canonical_name"], g["entity_type"])).fetchall()

            keeper_id = rows[0]["id"]
            dup_ids = [r["id"] for r in rows[1:]]
            if dup_ids:
                ph = ",".join("?" * len(dup_ids))
                db.execute(
                    f"UPDATE relationships SET subject_id = ? WHERE subject_id IN ({ph})",
                    [keeper_id] + dup_ids
                )
                db.execute(
                    f"UPDATE relationships SET object_id = ? WHERE object_id IN ({ph})",
                    [keeper_id] + dup_ids
                )
                db.execute(f"DELETE FROM entities WHERE id IN ({ph})", dup_ids)
                logger.info(
                    f"Migration 4: merged {len(dup_ids)} duplicate entities for "
                    f"'{g['canonical_name']}' ({g['entity_type']})"
                )

        # --- Clean up existing active relationship duplicates ---
        # Entity-to-entity: keep highest confidence per (subject_id, predicate, object_id)
        rel_groups = db.execute("""
            SELECT subject_id, predicate, object_id
            FROM relationships
            WHERE valid_to IS NULL AND object_id IS NOT NULL
            GROUP BY subject_id, predicate, object_id
            HAVING COUNT(*) > 1
        """).fetchall()

        for g in rel_groups:
            rows = db.execute("""
                SELECT id FROM relationships
                WHERE subject_id = ? AND predicate = ? AND object_id = ?
                AND valid_to IS NULL
                ORDER BY confidence DESC, created_at ASC
            """, (g["subject_id"], g["predicate"], g["object_id"])).fetchall()
            to_delete = [r["id"] for r in rows[1:]]
            if to_delete:
                ph = ",".join("?" * len(to_delete))
                db.execute(f"DELETE FROM relationships WHERE id IN ({ph})", to_delete)

        # Value-based: keep highest confidence per (subject_id, predicate, object_value)
        val_groups = db.execute("""
            SELECT subject_id, predicate, object_value
            FROM relationships
            WHERE valid_to IS NULL AND object_id IS NULL AND object_value IS NOT NULL
            GROUP BY subject_id, predicate, object_value
            HAVING COUNT(*) > 1
        """).fetchall()

        for g in val_groups:
            rows = db.execute("""
                SELECT id FROM relationships
                WHERE subject_id = ? AND predicate = ? AND object_value = ?
                AND valid_to IS NULL AND object_id IS NULL
                ORDER BY confidence DESC, created_at ASC
            """, (g["subject_id"], g["predicate"], g["object_value"])).fetchall()
            to_delete = [r["id"] for r in rows[1:]]
            if to_delete:
                ph = ",".join("?" * len(to_delete))
                db.execute(f"DELETE FROM relationships WHERE id IN ({ph})", to_delete)

        # --- Create unique indexes ---
        db.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_unique_canonical
            ON entities(canonical_name, entity_type)
        """)
        db.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_rel_active_entity
            ON relationships(subject_id, predicate, object_id)
            WHERE valid_to IS NULL AND object_id IS NOT NULL
        """)
        db.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_rel_active_value
            ON relationships(subject_id, predicate, object_value)
            WHERE valid_to IS NULL AND object_value IS NOT NULL
        """)

    run_migration(db, 4, "Dedup constraints — unique indexes on entities and active relationships", _migrate_dedup_constraints)

    db.close()

    # Set restrictive file permissions on the database (owner read/write only).
    # SQLite stores data in plaintext — file permissions are the first line of defense.
    import os
    db_path = _get_db_path()
    try:
        os.chmod(db_path, 0o600)
        # WAL and SHM files too, if they exist
        for suffix in ("-wal", "-shm"):
            wal_path = str(db_path) + suffix
            if os.path.exists(wal_path):
                os.chmod(wal_path, 0o600)
    except OSError:
        logger.debug("Could not set restrictive permissions on database files")


# ============================================================================
# EMBEDDING HELPERS
# ============================================================================

def get_embedding(text: str) -> list[float]:
    """Get embedding vector for a document/memory via the configured EmbedProvider."""
    import maasv
    return maasv.get_embed().embed(text)


def get_query_embedding(text: str) -> list[float]:
    """Get embedding vector for a search query (may use instruction prefix)."""
    import maasv
    return maasv.get_embed().embed_query(text)


def serialize_embedding(embedding: list[float]) -> bytes:
    """Convert embedding to binary format for sqlite-vec."""
    from sqlite_vec import serialize_float32
    return serialize_float32(embedding)


# ============================================================================
# ACCESS TRACKING
# ============================================================================

def _record_memory_access(db: sqlite3.Connection, memory_ids: list[str]):
    """Increment access_count and set last_accessed_at for retrieved memories."""
    if not memory_ids:
        return
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    placeholders = ",".join("?" * len(memory_ids))
    try:
        db.execute(
            f"UPDATE memories SET access_count = access_count + 1, last_accessed_at = ? "
            f"WHERE id IN ({placeholders})",
            [now] + memory_ids
        )
        db.commit()
    except Exception:
        logger.warning("Failed to record memory access", exc_info=True)


def _record_entity_access(db: sqlite3.Connection, entity_ids: list[str]):
    """Increment access_count and set last_accessed_at for retrieved entities."""
    if not entity_ids:
        return
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    placeholders = ",".join("?" * len(entity_ids))
    try:
        db.execute(
            f"UPDATE entities SET access_count = access_count + 1, last_accessed_at = ? "
            f"WHERE id IN ({placeholders})",
            [now] + entity_ids
        )
        db.commit()
    except Exception:
        logger.warning("Failed to record entity access", exc_info=True)


def _escape_like(value: str) -> str:
    """Escape LIKE wildcards (%, _) in a value for safe use in LIKE patterns."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


_FTS5_SPECIAL_RE = re.compile(r'[\"*()^]')
_FTS5_OPERATOR_RE = re.compile(r'\b(NEAR|NOT)\b', re.IGNORECASE)


def _sanitize_fts_input(text: str) -> str:
    """Strip FTS5 special syntax from raw text for safe use in MATCH queries.

    Removes characters that cause syntax errors (", *, (, ), ^) and operators
    that can produce unexpected results (NEAR, NOT). Preserves OR/AND since
    they are harmless (just modify query semantics) and used by callers.
    """
    text = _FTS5_SPECIAL_RE.sub(' ', text)
    text = _FTS5_OPERATOR_RE.sub(' ', text)
    return ' '.join(text.split())

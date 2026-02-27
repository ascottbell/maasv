"""
maasv Commitments — First-class tracked promises, follow-ups, and delegations.

Commitments are things Adam has agreed to do, is waiting on, or has delegated.
They have deadlines, escalation policies, and verification rules. The executive
loop checks them periodically and nudges when things are overdue.

Unlike generic entities/relationships, commitments have their own state machine
and query patterns (overdue, by-owner, escalation-ready) that warrant dedicated
storage.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from maasv.core.db import _plain_db as _db

logger = logging.getLogger(__name__)


class CommitmentType(str, Enum):
    """What kind of commitment this is."""

    PROMISE = "promise"  # Adam promised someone something
    FOLLOW_UP = "follow_up"  # Adam needs to follow up on something
    WAITING_ON = "waiting_on"  # Adam is waiting for someone else
    DELEGATION = "delegation"  # Adam delegated something to someone/Doris
    DRAFT_TO_SEND = "draft_to_send"  # Adam has a draft that needs sending
    REMINDER = "reminder"  # Time-based reminder (not a promise to anyone)


class CommitmentStatus(str, Enum):
    """Lifecycle state of a commitment."""

    OPEN = "open"  # Created, not yet started
    IN_PROGRESS = "in_progress"  # Actively being worked on
    BLOCKED = "blocked"  # Can't proceed, waiting on something
    COMPLETED = "completed"  # Done, verified
    EXPIRED = "expired"  # Deadline passed without completion
    CANCELLED = "cancelled"  # Explicitly cancelled


class DeadlineType(str, Enum):
    """How strict the deadline is."""

    HARD = "hard"  # Must be met (legal, contractual, etc.)
    SOFT = "soft"  # Should be met but some flexibility
    NONE = "none"  # No specific deadline


@dataclass
class Commitment:
    """A tracked commitment with full lifecycle metadata."""

    commitment_type: CommitmentType
    owner: str  # who owns this: "adam", "gabby", "doris", or a person's name
    subject: str  # what this is about
    status: CommitmentStatus = CommitmentStatus.OPEN
    deadline: Optional[str] = None  # ISO datetime
    deadline_type: DeadlineType = DeadlineType.NONE
    next_action: str = ""  # what needs to happen next
    verification_rule: str = ""  # how do we know it's done
    escalation_policy: str = ""  # what happens if deadline passes
    source: str = "conversation"  # where this came from
    related_entity_ids: list[str] = field(default_factory=list)
    context: str = ""  # additional context
    id: str = field(default_factory=lambda: f"cmt_{uuid.uuid4().hex[:12]}")
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    event_time: Optional[str] = None  # when the commitment was made in the real world
    ingested_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Optional[dict] = None


# ============================================================================
# TABLE MANAGEMENT
# ============================================================================


def ensure_commitments_table():
    """Create the commitments table if it doesn't exist.

    Called from the migration system in db.py — not standalone.
    This is provided for testing convenience.
    """
    with _db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS commitments (
                id TEXT PRIMARY KEY,
                commitment_type TEXT NOT NULL,
                owner TEXT NOT NULL,
                subject TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                deadline TEXT,
                deadline_type TEXT NOT NULL DEFAULT 'none',
                next_action TEXT DEFAULT '',
                verification_rule TEXT DEFAULT '',
                escalation_policy TEXT DEFAULT '',
                source TEXT NOT NULL DEFAULT 'conversation',
                related_entity_ids TEXT,
                context TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                completed_at TEXT,
                event_time TEXT,
                ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)

        # Indexes for common query patterns
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_commitments_status
            ON commitments(status)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_commitments_owner
            ON commitments(owner)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_commitments_deadline
            ON commitments(deadline)
            WHERE deadline IS NOT NULL
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_commitments_type
            ON commitments(commitment_type)
        """)

        # FTS for searching commitments by subject/context
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS commitments_fts USING fts5(
                subject,
                context,
                next_action,
                content='commitments',
                content_rowid='rowid'
            )
        """)

        # FTS sync triggers
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS commitments_ai AFTER INSERT ON commitments BEGIN
                INSERT INTO commitments_fts(rowid, subject, context, next_action)
                VALUES (NEW.rowid, NEW.subject, NEW.context, NEW.next_action);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS commitments_ad AFTER DELETE ON commitments BEGIN
                INSERT INTO commitments_fts(commitments_fts, rowid, subject, context, next_action)
                VALUES ('delete', OLD.rowid, OLD.subject, OLD.context, OLD.next_action);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS commitments_au AFTER UPDATE ON commitments BEGIN
                INSERT INTO commitments_fts(commitments_fts, rowid, subject, context, next_action)
                VALUES ('delete', OLD.rowid, OLD.subject, OLD.context, OLD.next_action);
                INSERT INTO commitments_fts(rowid, subject, context, next_action)
                VALUES (NEW.rowid, NEW.subject, NEW.context, NEW.next_action);
            END
        """)

        conn.commit()


# ============================================================================
# CRUD OPERATIONS
# ============================================================================


def _commitment_from_row(row) -> Commitment:
    """Convert a database row to a Commitment dataclass."""
    d = dict(row)
    entity_ids = json.loads(d["related_entity_ids"]) if d.get("related_entity_ids") else []
    metadata = json.loads(d["metadata"]) if d.get("metadata") else None

    return Commitment(
        id=d["id"],
        commitment_type=CommitmentType(d["commitment_type"]),
        owner=d["owner"],
        subject=d["subject"],
        status=CommitmentStatus(d["status"]),
        deadline=d.get("deadline"),
        deadline_type=DeadlineType(d.get("deadline_type", "none")),
        next_action=d.get("next_action", ""),
        verification_rule=d.get("verification_rule", ""),
        escalation_policy=d.get("escalation_policy", ""),
        source=d.get("source", "conversation"),
        related_entity_ids=entity_ids,
        context=d.get("context", ""),
        created_at=d["created_at"],
        updated_at=d["updated_at"],
        completed_at=d.get("completed_at"),
        event_time=d.get("event_time"),
        ingested_at=d.get("ingested_at", d["created_at"]),
        metadata=metadata,
    )


def create(commitment: Commitment) -> str:
    """Create a new commitment. Returns the commitment ID."""
    with _db() as conn:
        conn.execute(
            """
            INSERT INTO commitments
            (id, commitment_type, owner, subject, status, deadline, deadline_type,
             next_action, verification_rule, escalation_policy, source,
             related_entity_ids, context, created_at, updated_at,
             completed_at, event_time, ingested_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                commitment.id,
                commitment.commitment_type.value,
                commitment.owner,
                commitment.subject,
                commitment.status.value,
                commitment.deadline,
                commitment.deadline_type.value,
                commitment.next_action,
                commitment.verification_rule,
                commitment.escalation_policy,
                commitment.source,
                json.dumps(commitment.related_entity_ids) if commitment.related_entity_ids else None,
                commitment.context,
                commitment.created_at,
                commitment.updated_at,
                commitment.completed_at,
                commitment.event_time,
                commitment.ingested_at,
                json.dumps(commitment.metadata) if commitment.metadata else None,
            ),
        )
        conn.commit()

    logger.info(f"Created commitment {commitment.id}: [{commitment.commitment_type.value}] {commitment.subject}")
    return commitment.id


def get(commitment_id: str) -> Optional[Commitment]:
    """Get a commitment by ID."""
    with _db() as conn:
        row = conn.execute("SELECT * FROM commitments WHERE id = ?", (commitment_id,)).fetchone()
        if row:
            return _commitment_from_row(row)
    return None


def update_status(commitment_id: str, status: CommitmentStatus, details: str = "") -> bool:
    """Update commitment status. Handles completed_at timestamp automatically."""
    now = datetime.now(timezone.utc).isoformat()

    with _db() as conn:
        updates = {"status": status.value, "updated_at": now}

        if status == CommitmentStatus.COMPLETED:
            updates["completed_at"] = now

        if details:
            # Append to context
            row = conn.execute("SELECT context FROM commitments WHERE id = ?", (commitment_id,)).fetchone()
            if row:
                existing = row["context"] or ""
                separator = "\n---\n" if existing else ""
                updates["context"] = f"{existing}{separator}[{now}] Status -> {status.value}: {details}"

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [commitment_id]

        cursor = conn.execute(f"UPDATE commitments SET {set_clause} WHERE id = ?", values)
        conn.commit()
        updated = cursor.rowcount > 0

    if updated:
        logger.info(f"Commitment {commitment_id} -> {status.value}")
    return updated


def update_next_action(commitment_id: str, next_action: str) -> bool:
    """Update the next action for a commitment."""
    now = datetime.now(timezone.utc).isoformat()

    with _db() as conn:
        cursor = conn.execute(
            "UPDATE commitments SET next_action = ?, updated_at = ? WHERE id = ?",
            (next_action, now, commitment_id),
        )
        conn.commit()
        return cursor.rowcount > 0


def delete(commitment_id: str) -> bool:
    """Permanently delete a commitment."""
    with _db() as conn:
        cursor = conn.execute("DELETE FROM commitments WHERE id = ?", (commitment_id,))
        conn.commit()
        return cursor.rowcount > 0


# ============================================================================
# QUERY OPERATIONS
# ============================================================================

# Active statuses — commitments that still need attention
_ACTIVE_STATUSES = (
    CommitmentStatus.OPEN.value,
    CommitmentStatus.IN_PROGRESS.value,
    CommitmentStatus.BLOCKED.value,
)


def get_active(limit: int = 50) -> list[Commitment]:
    """Get all active (non-terminal) commitments, ordered by deadline urgency."""
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM commitments
            WHERE status IN (?, ?, ?)
            ORDER BY
                CASE WHEN deadline IS NOT NULL THEN 0 ELSE 1 END,
                deadline ASC,
                created_at ASC
            LIMIT ?
            """,
            (*_ACTIVE_STATUSES, limit),
        ).fetchall()

    return [_commitment_from_row(row) for row in rows]


def get_overdue() -> list[Commitment]:
    """Get commitments past their deadline that are still active."""
    now = datetime.now(timezone.utc).isoformat()

    with _db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM commitments
            WHERE status IN (?, ?, ?)
            AND deadline IS NOT NULL
            AND deadline < ?
            ORDER BY deadline ASC
            """,
            (*_ACTIVE_STATUSES, now),
        ).fetchall()

    return [_commitment_from_row(row) for row in rows]


def get_upcoming(hours: int = 24) -> list[Commitment]:
    """Get active commitments with deadlines in the next N hours."""
    now = datetime.now(timezone.utc)
    from datetime import timedelta

    cutoff = (now + timedelta(hours=hours)).isoformat()
    now_iso = now.isoformat()

    with _db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM commitments
            WHERE status IN (?, ?, ?)
            AND deadline IS NOT NULL
            AND deadline >= ?
            AND deadline <= ?
            ORDER BY deadline ASC
            """,
            (*_ACTIVE_STATUSES, now_iso, cutoff),
        ).fetchall()

    return [_commitment_from_row(row) for row in rows]


def get_by_owner(owner: str, include_completed: bool = False) -> list[Commitment]:
    """Get commitments owned by a specific person."""
    with _db() as conn:
        if include_completed:
            rows = conn.execute(
                "SELECT * FROM commitments WHERE owner = ? ORDER BY created_at DESC",
                (owner,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM commitments
                WHERE owner = ? AND status IN (?, ?, ?)
                ORDER BY
                    CASE WHEN deadline IS NOT NULL THEN 0 ELSE 1 END,
                    deadline ASC,
                    created_at ASC
                """,
                (owner, *_ACTIVE_STATUSES),
            ).fetchall()

    return [_commitment_from_row(row) for row in rows]


def get_by_type(commitment_type: CommitmentType, include_completed: bool = False) -> list[Commitment]:
    """Get commitments of a specific type."""
    with _db() as conn:
        if include_completed:
            rows = conn.execute(
                "SELECT * FROM commitments WHERE commitment_type = ? ORDER BY created_at DESC",
                (commitment_type.value,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM commitments
                WHERE commitment_type = ? AND status IN (?, ?, ?)
                ORDER BY deadline ASC, created_at ASC
                """,
                (commitment_type.value, *_ACTIVE_STATUSES),
            ).fetchall()

    return [_commitment_from_row(row) for row in rows]


def check_escalations() -> list[Commitment]:
    """Get commitments that are overdue and have escalation policies.

    This is what the executive loop calls periodically to decide what
    to nag Adam about.
    """
    now = datetime.now(timezone.utc).isoformat()

    with _db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM commitments
            WHERE status IN (?, ?, ?)
            AND deadline IS NOT NULL
            AND deadline < ?
            AND escalation_policy != ''
            ORDER BY deadline ASC
            """,
            (*_ACTIVE_STATUSES, now),
        ).fetchall()

    return [_commitment_from_row(row) for row in rows]


def search(query: str, limit: int = 10) -> list[Commitment]:
    """Full-text search across commitment subjects and context."""
    from maasv.core.db import _sanitize_fts_input

    sanitized = _sanitize_fts_input(query)
    if not sanitized:
        return []

    with _db() as conn:
        try:
            rows = conn.execute(
                """
                SELECT c.* FROM commitments c
                JOIN commitments_fts fts ON c.rowid = fts.rowid
                WHERE commitments_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (sanitized, limit),
            ).fetchall()
            return [_commitment_from_row(row) for row in rows]
        except Exception:
            logger.debug("FTS search failed for commitments", exc_info=True)
            return []


def get_waiting_on(limit: int = 20) -> list[Commitment]:
    """Get things Adam is waiting on from other people."""
    return get_by_type(CommitmentType.WAITING_ON)[:limit]


def get_stats() -> dict:
    """Get summary statistics about commitments."""
    with _db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM commitments").fetchone()[0]
        active = conn.execute(
            "SELECT COUNT(*) FROM commitments WHERE status IN (?, ?, ?)",
            _ACTIVE_STATUSES,
        ).fetchone()[0]
        overdue_count = 0
        now = datetime.now(timezone.utc).isoformat()
        overdue_row = conn.execute(
            """
            SELECT COUNT(*) FROM commitments
            WHERE status IN (?, ?, ?) AND deadline IS NOT NULL AND deadline < ?
            """,
            (*_ACTIVE_STATUSES, now),
        ).fetchone()
        if overdue_row:
            overdue_count = overdue_row[0]

        by_type = conn.execute(
            """
            SELECT commitment_type, COUNT(*) as count
            FROM commitments
            WHERE status IN (?, ?, ?)
            GROUP BY commitment_type
            """,
            _ACTIVE_STATUSES,
        ).fetchall()

        by_owner = conn.execute(
            """
            SELECT owner, COUNT(*) as count
            FROM commitments
            WHERE status IN (?, ?, ?)
            GROUP BY owner
            ORDER BY count DESC
            """,
            _ACTIVE_STATUSES,
        ).fetchall()

    return {
        "total": total,
        "active": active,
        "overdue": overdue_count,
        "by_type": {row["commitment_type"]: row["count"] for row in by_type},
        "by_owner": {row["owner"]: row["count"] for row in by_owner},
    }


# ============================================================================
# FORMATTING
# ============================================================================


def format_for_prompt(commitments: list[Commitment], heading: str = "Active Commitments") -> str:
    """Format commitments for LLM prompt injection."""
    if not commitments:
        return ""

    lines = [f"## {heading}\n"]
    now = datetime.now(timezone.utc)

    for c in commitments:
        status_icon = {
            CommitmentStatus.OPEN: "[ ]",
            CommitmentStatus.IN_PROGRESS: "[~]",
            CommitmentStatus.BLOCKED: "[!]",
            CommitmentStatus.COMPLETED: "[x]",
            CommitmentStatus.EXPIRED: "[x]",
            CommitmentStatus.CANCELLED: "[-]",
        }.get(c.status, "[ ]")

        line = f"{status_icon} **{c.subject}** ({c.commitment_type.value}, owner: {c.owner})"

        if c.deadline:
            try:
                dl = datetime.fromisoformat(c.deadline)
                if dl.tzinfo is None:
                    dl = dl.replace(tzinfo=timezone.utc)
                delta = dl - now
                if delta.total_seconds() < 0:
                    line += f" **OVERDUE by {abs(delta.days)}d**"
                elif delta.days == 0:
                    hours = int(delta.total_seconds() // 3600)
                    line += f" (due in {hours}h)"
                else:
                    line += f" (due in {delta.days}d)"
            except ValueError:
                line += f" (deadline: {c.deadline})"

        lines.append(line)
        if c.next_action:
            lines.append(f"  Next: {c.next_action}")

    return "\n".join(lines)

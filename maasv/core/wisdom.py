"""
maasv Wisdom: Learnings from experience.

Captures reasoning before actions, tracks outcomes,
and incorporates feedback to improve future decisions.

Unlike factual memory (who is the user's spouse, what's their schedule), wisdom is
experiential — patterns of what works and what doesn't, learned over time.
"""

import logging
import sqlite3
import json
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

from maasv.core.db import _plain_db as _db, _sanitize_fts_input

logger = logging.getLogger(__name__)


@dataclass
class WisdomEntry:
    """A single piece of experiential wisdom."""
    action_type: str
    reasoning: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    action_data: Optional[dict] = None
    trigger: Optional[str] = None
    context: Optional[str] = None
    outcome: str = "pending"
    outcome_details: Optional[str] = None
    feedback_score: Optional[int] = None
    feedback_notes: Optional[str] = None
    feedback_at: Optional[str] = None
    tags: Optional[list[str]] = None
    superseded_by: Optional[str] = None


def ensure_wisdom_tables():
    """Create wisdom and wisdom_fts tables if they don't exist."""
    with _db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS wisdom (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                action_type TEXT,
                action_data TEXT,
                trigger TEXT,
                context TEXT,
                reasoning TEXT,
                outcome TEXT DEFAULT 'pending',
                outcome_details TEXT,
                feedback_score INTEGER,
                feedback_notes TEXT,
                feedback_at TEXT,
                tags TEXT,
                superseded_by TEXT
            )
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS wisdom_fts USING fts5(
                action_type,
                reasoning,
                context,
                feedback_notes,
                content='wisdom',
                content_rowid='rowid'
            )
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS wisdom_ai AFTER INSERT ON wisdom BEGIN
                INSERT INTO wisdom_fts(rowid, action_type, reasoning, context, feedback_notes)
                VALUES (NEW.rowid, NEW.action_type, NEW.reasoning, NEW.context, NEW.feedback_notes);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS wisdom_ad AFTER DELETE ON wisdom BEGIN
                INSERT INTO wisdom_fts(wisdom_fts, rowid, action_type, reasoning, context, feedback_notes)
                VALUES ('delete', OLD.rowid, OLD.action_type, OLD.reasoning, OLD.context, OLD.feedback_notes);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS wisdom_au AFTER UPDATE ON wisdom BEGIN
                INSERT INTO wisdom_fts(wisdom_fts, rowid, action_type, reasoning, context, feedback_notes)
                VALUES ('delete', OLD.rowid, OLD.action_type, OLD.reasoning, OLD.context, OLD.feedback_notes);
                INSERT INTO wisdom_fts(rowid, action_type, reasoning, context, feedback_notes)
                VALUES (NEW.rowid, NEW.action_type, NEW.reasoning, NEW.context, NEW.feedback_notes);
            END
        """)
        conn.commit()


def log_reasoning(
    action_type: str,
    reasoning: str,
    action_data: dict = None,
    trigger: str = None,
    context: str = None,
    tags: list[str] = None,
) -> str:
    """Log reasoning before taking an action. Returns wisdom ID for later feedback."""
    entry = WisdomEntry(
        action_type=action_type,
        reasoning=reasoning,
        action_data=action_data,
        trigger=trigger,
        context=context,
        tags=tags,
    )

    with _db() as conn:
        conn.execute(
            """
            INSERT INTO wisdom (id, timestamp, action_type, action_data, trigger,
                               context, reasoning, outcome, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id, entry.timestamp, entry.action_type,
                json.dumps(entry.action_data) if entry.action_data else None,
                entry.trigger, entry.context, entry.reasoning, entry.outcome,
                json.dumps(entry.tags) if entry.tags else None,
            )
        )
        conn.commit()
        return entry.id


def record_outcome(wisdom_id: str, outcome: str, details: str = None) -> bool:
    """Record the outcome of an action."""
    with _db() as conn:
        cursor = conn.execute(
            "UPDATE wisdom SET outcome = ?, outcome_details = ? WHERE id = ?",
            (outcome, details, wisdom_id)
        )
        conn.commit()
        return cursor.rowcount > 0


def add_feedback(wisdom_id: str, score: int, notes: str = None) -> bool:
    """Attach feedback to a wisdom entry. Score: 1-5."""
    if not 1 <= score <= 5:
        raise ValueError("Score must be between 1 and 5")

    with _db() as conn:
        cursor = conn.execute(
            "UPDATE wisdom SET feedback_score = ?, feedback_notes = ?, feedback_at = ? WHERE id = ?",
            (score, notes, datetime.now(timezone.utc).isoformat(), wisdom_id)
        )
        conn.commit()
        return cursor.rowcount > 0


def get_relevant_wisdom(
    action_type: str,
    limit: int = 5,
    include_unrated: bool = False,
) -> list[dict]:
    """Get relevant wisdom for a given action type, prioritizing failures."""
    with _db() as conn:
        if include_unrated:
            query = """
                SELECT * FROM wisdom
                WHERE action_type = ?
                ORDER BY
                    CASE WHEN feedback_score IS NULL THEN 1 ELSE 0 END,
                    CASE WHEN feedback_score <= 2 THEN 0 ELSE 1 END,
                    feedback_score DESC,
                    timestamp DESC
                LIMIT ?
            """
            rows = conn.execute(query, (action_type, limit)).fetchall()
        else:
            query = """
                SELECT * FROM wisdom
                WHERE action_type = ?
                  AND feedback_score IS NOT NULL
                ORDER BY
                    CASE WHEN feedback_score <= 2 THEN 0 ELSE 1 END,
                    feedback_score DESC,
                    timestamp DESC
                LIMIT ?
            """
            rows = conn.execute(query, (action_type, limit)).fetchall()

        return [_row_to_dict(row) for row in rows]


def search_wisdom(query: str, limit: int = 10) -> list[dict]:
    """Full-text search across reasoning, context, and feedback."""
    query = _sanitize_fts_input(query)
    if not query:
        return []

    with _db() as conn:
        try:
            rows = conn.execute(
                """
                SELECT w.* FROM wisdom w
                JOIN wisdom_fts fts ON w.rowid = fts.rowid
                WHERE wisdom_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit)
            ).fetchall()
            return [_row_to_dict(row) for row in rows]
        except sqlite3.OperationalError:
            logger.debug("FTS5 query failed in search_wisdom: %s", query, exc_info=True)
            return []


def get_wisdom_by_id(wisdom_id: str) -> Optional[dict]:
    """Get a specific wisdom entry by ID."""
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM wisdom WHERE id = ?", (wisdom_id,)
        ).fetchone()
        return _row_to_dict(row) if row else None


def get_recent_wisdom(limit: int = 10) -> list[dict]:
    """Get the most recent wisdom entries."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM wisdom ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [_row_to_dict(row) for row in rows]


def get_pending_feedback(limit: int = 20) -> list[dict]:
    """Get wisdom entries that haven't received feedback yet."""
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM wisdom
            WHERE feedback_score IS NULL
              AND outcome != 'pending'
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()
        return [_row_to_dict(row) for row in rows]


def format_wisdom_for_prompt(entries: list[dict]) -> str:
    """Format wisdom entries for inclusion in a prompt."""
    if not entries:
        return ""

    lines = ["## Learnings from past experience:\n"]
    for entry in entries:
        score = entry.get("feedback_score")
        if score and score <= 2:
            label = "LEARNED FROM MISTAKE"
        elif score and score >= 4:
            label = "WORKED WELL"
        else:
            label = "MIXED"

        lines.append(f"- **{label}** ({score}/5 rating)")
        lines.append(f"   Reasoning: {entry['reasoning'][:200]}...")
        if entry.get("feedback_notes"):
            lines.append(f"   Feedback: {entry['feedback_notes']}")
        lines.append("")

    return "\n".join(lines)


def delete_wisdom(wisdom_id: str) -> bool:
    """Delete a wisdom entry."""
    with _db() as conn:
        cursor = conn.execute("DELETE FROM wisdom WHERE id = ?", (wisdom_id,))
        conn.commit()
        return cursor.rowcount > 0


def update_wisdom(wisdom_id: str, **kwargs) -> bool:
    """Update fields on a wisdom entry.

    NOTE: Column names in set_clause come from allowed_fields intersection,
    not user input. If adding fields here, ensure names are safe SQL identifiers.
    """
    allowed_fields = {
        "reasoning", "context", "outcome", "outcome_details",
        "feedback_score", "feedback_notes", "tags"
    }

    updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
    if not updates:
        return False

    if "tags" in updates and isinstance(updates["tags"], list):
        updates["tags"] = json.dumps(updates["tags"])

    set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
    values = list(updates.values()) + [wisdom_id]

    with _db() as conn:
        cursor = conn.execute(
            f"UPDATE wisdom SET {set_clause} WHERE id = ?",
            values
        )
        conn.commit()
        return cursor.rowcount > 0


def get_stats() -> dict:
    """Get statistics about the wisdom database."""
    with _db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM wisdom").fetchone()[0]
        rated = conn.execute(
            "SELECT COUNT(*) FROM wisdom WHERE feedback_score IS NOT NULL"
        ).fetchone()[0]

        by_action = conn.execute(
            """
            SELECT action_type, COUNT(*) as count,
                   AVG(feedback_score) as avg_score
            FROM wisdom
            GROUP BY action_type
            ORDER BY count DESC
            """
        ).fetchall()

        avg_score = conn.execute(
            "SELECT AVG(feedback_score) FROM wisdom WHERE feedback_score IS NOT NULL"
        ).fetchone()[0]

        return {
            "total_entries": total,
            "rated_entries": rated,
            "unrated_entries": total - rated,
            "average_score": round(avg_score, 2) if avg_score else None,
            "by_action_type": [dict(row) for row in by_action],
        }


def _get_action_families() -> dict[str, list[str]]:
    """Get action families from config."""
    import maasv
    try:
        return maasv.get_config().action_families
    except RuntimeError:
        return {}


def _build_action_to_family() -> dict[str, str]:
    """Build reverse lookup from action type to family name.

    NOTE: Rebuilt on every call. Cheap for small configs (typical: <20 entries).
    Cache at module level if action_families grows large.
    """
    families = _get_action_families()
    mapping = {}
    for family, actions in families.items():
        for action in actions:
            mapping[action] = family
    return mapping


def get_action_family(action_type: str) -> Optional[str]:
    """Get the family name for an action type."""
    return _build_action_to_family().get(action_type)


def get_family_actions(action_type: str) -> list[str]:
    """Get all action types in the same family as the given action."""
    mapping = _build_action_to_family()
    family = mapping.get(action_type)
    if family:
        return _get_action_families()[family]
    return [action_type]


def should_query_wisdom(
    action_type: str,
    threshold_executions: int = 10,
    threshold_failure_rate: float = 0.1
) -> tuple[bool, str]:
    """Determine if we should query wisdom before executing an action."""
    with _db() as conn:
        stats = conn.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN outcome = 'failed' THEN 1 ELSE 0 END) as failures,
                SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as successes,
                SUM(CASE WHEN feedback_score IS NOT NULL AND feedback_score <= 2 THEN 1 ELSE 0 END) as negative_feedback
            FROM wisdom
            WHERE action_type = ?
            """,
            (action_type,)
        ).fetchone()

        total = stats["total"] or 0
        failures = stats["failures"] or 0
        negative_feedback = stats["negative_feedback"] or 0

        if total < threshold_executions:
            return True, f"learning ({total}/{threshold_executions} executions)"
        if negative_feedback > 0:
            return True, f"has {negative_feedback} negative feedback entries"
        if total > 0:
            failure_rate = failures / total
            if failure_rate > threshold_failure_rate:
                return True, f"high failure rate ({failure_rate:.0%})"

        return False, f"well-established ({total} executions, {failures} failures)"


def get_smart_wisdom(
    action_type: str,
    context: str = None,
    limit: int = 5,
) -> list[dict]:
    """Get relevant wisdom using smart matching across action families."""
    results = []

    with _db() as conn:
        # 1. Exact matches with failures or feedback
        exact_rows = conn.execute(
            """
            SELECT *, 1 as match_quality FROM wisdom
            WHERE action_type = ?
              AND (outcome = 'failed' OR feedback_score IS NOT NULL)
            ORDER BY
                CASE WHEN outcome = 'failed' THEN 0 ELSE 1 END,
                CASE WHEN feedback_score <= 2 THEN 0
                     WHEN feedback_score IS NULL THEN 2
                     ELSE 1 END,
                timestamp DESC
            LIMIT ?
            """,
            (action_type, limit)
        ).fetchall()
        results.extend([_row_to_dict(r) for r in exact_rows])

        # 2. Family matches
        remaining = limit - len(results)
        if remaining > 0:
            family_actions = get_family_actions(action_type)
            sibling_actions = [a for a in family_actions if a != action_type]

            if sibling_actions:
                placeholders = ",".join("?" * len(sibling_actions))
                family_rows = conn.execute(
                    f"""
                    SELECT *, 2 as match_quality FROM wisdom
                    WHERE action_type IN ({placeholders})
                      AND (outcome = 'failed' OR feedback_score IS NOT NULL)
                    ORDER BY
                        CASE WHEN outcome = 'failed' THEN 0 ELSE 1 END,
                        CASE WHEN feedback_score <= 2 THEN 0
                             WHEN feedback_score IS NULL THEN 2
                             ELSE 1 END,
                        timestamp DESC
                    LIMIT ?
                    """,
                    (*sibling_actions, remaining)
                ).fetchall()
                results.extend([_row_to_dict(r) for r in family_rows])

        # 3. Context-based search
        remaining = limit - len(results)
        if remaining > 0 and context:
            words = [_sanitize_fts_input(w) for w in context.split() if len(w) > 3][:3]
            words = [w for w in words if w]
            if words:
                search_query = " OR ".join(words)
                try:
                    context_rows = conn.execute(
                        """
                        SELECT w.*, 3 as match_quality FROM wisdom w
                        JOIN wisdom_fts fts ON w.rowid = fts.rowid
                        WHERE wisdom_fts MATCH ?
                          AND w.action_type != ?
                          AND (w.outcome = 'failed' OR w.feedback_score IS NOT NULL)
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (search_query, action_type, remaining)
                    ).fetchall()

                    existing_ids = {r["id"] for r in results}
                    for row in context_rows:
                        d = _row_to_dict(row)
                        if d["id"] not in existing_ids:
                            results.append(d)
                except sqlite3.OperationalError:
                    pass

        return results[:limit]


def format_smart_wisdom_for_prompt(entries: list[dict]) -> str:
    """Format smart wisdom entries for inclusion in a prompt."""
    if not entries:
        return ""

    lines = ["<wisdom>", "Past experiences relevant to this action:"]

    for entry in entries:
        action = entry.get("action_type", "unknown")
        score = entry.get("feedback_score")
        outcome = entry.get("outcome", "unknown")

        if outcome == "failed":
            indicator = "FAILED"
        elif score and score <= 2:
            indicator = "BAD"
        elif score and score >= 4:
            indicator = "GOOD"
        else:
            indicator = "OK"

        line = f"- [{indicator}] {action}"

        if entry.get("feedback_notes"):
            line += f": {entry['feedback_notes']}"
        elif entry.get("outcome_details") and outcome == "failed":
            line += f": {entry['outcome_details'][:100]}"
        elif entry.get("reasoning"):
            line += f": {entry['reasoning'][:100]}"

        lines.append(line)

    lines.append("</wisdom>")
    return "\n".join(lines)


# ============================================================================
# Escalation Wisdom
#
# Higher-level convenience functions for tracking escalation decisions.
# Useful for agents that monitor external sources (email, calendar, feeds)
# and need to learn which items warrant user notification.
# ============================================================================

def log_escalation_miss(
    source: str,
    description: str,
    sender: str = None,
    subject: str = None,
    why_important: str = None,
    tags: list[str] = None,
) -> str:
    """Log when something should have been escalated but wasn't."""
    action_type = f"{source}_escalation_miss"

    reasoning = f"Missed escalation: {description}"
    if why_important:
        reasoning += f". Feedback: {why_important}"

    context_parts = []
    if sender:
        context_parts.append(f"From: {sender}")
    if subject:
        context_parts.append(f"Subject: {subject}")
    context = "; ".join(context_parts) if context_parts else None

    entry_id = log_reasoning(
        action_type=action_type,
        reasoning=reasoning,
        action_data={"sender": sender, "subject": subject, "description": description},
        trigger=f"{source}_scout",
        context=context,
        tags=tags or [],
    )

    record_outcome(entry_id, "failed", f"Should have escalated: {description[:100]}")
    add_feedback(entry_id, score=1, notes=why_important or "Missed important escalation")

    return entry_id


def log_escalation_correct(
    source: str,
    description: str,
    tags: list[str] = None,
) -> str:
    """Log when an escalation decision was correct."""
    action_type = f"{source}_escalation_correct"

    entry_id = log_reasoning(
        action_type=action_type,
        reasoning=f"Correct escalation decision: {description}",
        trigger=f"{source}_scout",
        tags=tags or [],
    )

    record_outcome(entry_id, "success", description[:100])
    return entry_id


def get_escalation_patterns(source: str, limit: int = 10) -> list[dict]:
    """Get escalation patterns to help agents make better decisions."""
    with _db() as conn:
        miss_action = f"{source}_escalation_miss"
        rows = conn.execute(
            "SELECT * FROM wisdom WHERE action_type = ? ORDER BY timestamp DESC LIMIT ?",
            (miss_action, limit)
        ).fetchall()
        return [_row_to_dict(row) for row in rows]


def should_escalate_based_on_wisdom(
    source: str,
    sender: str = None,
    subject: str = None,
    snippet: str = None,
) -> tuple[bool, Optional[str]]:
    """Check if wisdom suggests this should be escalated based on past misses."""
    patterns = get_escalation_patterns(source, limit=20)

    if not patterns:
        return False, None

    current_text = " ".join(filter(None, [sender, subject, snippet])).lower()
    if not current_text:
        return False, None

    for pattern in patterns:
        action_data = pattern.get("action_data", {})
        if isinstance(action_data, str):
            try:
                action_data = json.loads(action_data)
            except Exception:
                action_data = {}

        pattern_sender = (action_data.get("sender") or "").lower()
        if pattern_sender and pattern_sender in current_text:
            return True, f"Previously missed escalation from similar sender: {pattern_sender}"

        pattern_tags = pattern.get("tags", [])
        if isinstance(pattern_tags, str):
            try:
                pattern_tags = json.loads(pattern_tags)
            except Exception:
                pattern_tags = []

        for tag in pattern_tags:
            if tag.lower() in current_text:
                return True, f"Matches past missed pattern: {tag}"

        pattern_subject = (action_data.get("subject") or "").lower()
        if pattern_subject:
            keywords = [w for w in pattern_subject.split() if len(w) > 3]
            for kw in keywords:
                if kw in current_text:
                    return True, f"Similar to missed escalation with subject containing '{kw}'"

    return False, None


def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a database row to a dictionary."""
    d = dict(row)
    if d.get("action_data"):
        try:
            d["action_data"] = json.loads(d["action_data"])
        except json.JSONDecodeError:
            pass
    if d.get("tags"):
        try:
            d["tags"] = json.loads(d["tags"])
        except json.JSONDecodeError:
            pass
    return d

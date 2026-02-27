"""
maasv After-Action Review — MetaAgent-style post-task analysis.

Tracks the full plan -> execute -> result chain for non-trivial tasks.
Each completed task gets a RunRecord capturing:
- Goal (what was the intent)
- Context snapshot (world state at task start)
- Tools used in sequence
- Outcome (success/failure/partial)
- User feedback (if provided)
- Time taken + tokens consumed

Extends the existing wisdom system (which tracks individual tool reasoning)
to cover full multi-step task sequences. The executive loop or Doris brain
calls start_run() before a task, adds tool steps, then completes it.

Storage: dedicated run_records table in the maasv DB, linked to wisdom
entries for individual tool invocations within the run.
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


class RunOutcome(str, Enum):
    """Outcome of a completed run."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"  # some goals achieved, others not
    CANCELLED = "cancelled"  # user or system cancelled
    PENDING = "pending"  # still in progress


@dataclass
class ToolStep:
    """A single tool invocation within a run."""

    tool_name: str
    arguments_summary: str  # not full args — just enough to understand intent
    result_summary: str  # truncated result
    success: bool
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_ms: int = 0
    wisdom_id: Optional[str] = None  # link to wisdom entry if one was created
    error: Optional[str] = None


@dataclass
class RunRecord:
    """A complete record of a multi-step task execution."""

    goal: str  # what was the intent
    id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex[:12]}")
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    outcome: RunOutcome = RunOutcome.PENDING
    outcome_details: str = ""
    context_snapshot: str = ""  # world state summary at start
    steps: list[ToolStep] = field(default_factory=list)
    model_used: str = ""  # which model handled this
    model_tier: str = ""  # local/haiku/sonnet/opus
    input_tokens: int = 0
    output_tokens: int = 0
    user_feedback_score: Optional[int] = None  # 1-5
    user_feedback_notes: Optional[str] = None
    source: str = ""  # what triggered this run (voice, chat, proactive, etc.)
    trigger: str = ""  # specific trigger event
    metadata: Optional[dict] = None


# ============================================================================
# TABLE MANAGEMENT
# ============================================================================


def ensure_run_records_table():
    """Create the run_records table if it doesn't exist.

    Called from the migration system in db.py or standalone for testing.
    """
    with _db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS run_records (
                id TEXT PRIMARY KEY,
                goal TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                outcome TEXT NOT NULL DEFAULT 'pending',
                outcome_details TEXT DEFAULT '',
                context_snapshot TEXT DEFAULT '',
                steps TEXT DEFAULT '[]',
                model_used TEXT DEFAULT '',
                model_tier TEXT DEFAULT '',
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                user_feedback_score INTEGER,
                user_feedback_notes TEXT,
                source TEXT DEFAULT '',
                trigger TEXT DEFAULT '',
                metadata TEXT
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_run_records_outcome
            ON run_records(outcome)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_run_records_started
            ON run_records(started_at)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_run_records_source
            ON run_records(source)
        """)

        # FTS for searching run goals and outcomes
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS run_records_fts USING fts5(
                goal,
                outcome_details,
                content='run_records',
                content_rowid='rowid'
            )
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS run_records_ai AFTER INSERT ON run_records BEGIN
                INSERT INTO run_records_fts(rowid, goal, outcome_details)
                VALUES (NEW.rowid, NEW.goal, NEW.outcome_details);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS run_records_ad AFTER DELETE ON run_records BEGIN
                INSERT INTO run_records_fts(run_records_fts, rowid, goal, outcome_details)
                VALUES ('delete', OLD.rowid, OLD.goal, OLD.outcome_details);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS run_records_au AFTER UPDATE ON run_records BEGIN
                INSERT INTO run_records_fts(run_records_fts, rowid, goal, outcome_details)
                VALUES ('delete', OLD.rowid, OLD.goal, OLD.outcome_details);
                INSERT INTO run_records_fts(rowid, goal, outcome_details)
                VALUES (NEW.rowid, NEW.goal, NEW.outcome_details);
            END
        """)

        conn.commit()


# ============================================================================
# RUN LIFECYCLE
# ============================================================================


def _run_from_row(row) -> RunRecord:
    """Convert a database row to a RunRecord."""
    d = dict(row)
    steps = json.loads(d.get("steps") or "[]")
    metadata = json.loads(d["metadata"]) if d.get("metadata") else None

    tool_steps = []
    for s in steps:
        tool_steps.append(ToolStep(
            tool_name=s.get("tool_name", ""),
            arguments_summary=s.get("arguments_summary", ""),
            result_summary=s.get("result_summary", ""),
            success=s.get("success", True),
            timestamp=s.get("timestamp", ""),
            duration_ms=s.get("duration_ms", 0),
            wisdom_id=s.get("wisdom_id"),
            error=s.get("error"),
        ))

    return RunRecord(
        id=d["id"],
        goal=d["goal"],
        started_at=d["started_at"],
        completed_at=d.get("completed_at"),
        outcome=RunOutcome(d.get("outcome", "pending")),
        outcome_details=d.get("outcome_details", ""),
        context_snapshot=d.get("context_snapshot", ""),
        steps=tool_steps,
        model_used=d.get("model_used", ""),
        model_tier=d.get("model_tier", ""),
        input_tokens=d.get("input_tokens", 0),
        output_tokens=d.get("output_tokens", 0),
        user_feedback_score=d.get("user_feedback_score"),
        user_feedback_notes=d.get("user_feedback_notes"),
        source=d.get("source", ""),
        trigger=d.get("trigger", ""),
        metadata=metadata,
    )


def _steps_to_json(steps: list[ToolStep]) -> str:
    """Serialize tool steps to JSON."""
    return json.dumps([
        {
            "tool_name": s.tool_name,
            "arguments_summary": s.arguments_summary,
            "result_summary": s.result_summary,
            "success": s.success,
            "timestamp": s.timestamp,
            "duration_ms": s.duration_ms,
            "wisdom_id": s.wisdom_id,
            "error": s.error,
        }
        for s in steps
    ])


def start_run(
    goal: str,
    context_snapshot: str = "",
    source: str = "",
    trigger: str = "",
    model_used: str = "",
    model_tier: str = "",
    metadata: Optional[dict] = None,
) -> str:
    """Start a new run. Returns the run ID.

    Call this before beginning a multi-step task. Then add tool steps
    as they execute, and call complete_run when done.
    """
    run = RunRecord(
        goal=goal,
        context_snapshot=context_snapshot,
        source=source,
        trigger=trigger,
        model_used=model_used,
        model_tier=model_tier,
        metadata=metadata,
    )

    with _db() as conn:
        conn.execute(
            """
            INSERT INTO run_records
            (id, goal, started_at, outcome, context_snapshot, steps,
             model_used, model_tier, source, trigger, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.id,
                run.goal,
                run.started_at,
                run.outcome.value,
                run.context_snapshot,
                "[]",
                run.model_used,
                run.model_tier,
                run.source,
                run.trigger,
                json.dumps(run.metadata) if run.metadata else None,
            ),
        )
        conn.commit()

    logger.info(f"Started run {run.id}: {goal[:80]}")
    return run.id


def add_step(
    run_id: str,
    tool_name: str,
    arguments_summary: str,
    result_summary: str,
    success: bool,
    duration_ms: int = 0,
    wisdom_id: Optional[str] = None,
    error: Optional[str] = None,
) -> bool:
    """Add a tool step to an in-progress run.

    Appends to the steps JSON array in the database.
    """
    step = ToolStep(
        tool_name=tool_name,
        arguments_summary=arguments_summary[:500],
        result_summary=result_summary[:500],
        success=success,
        duration_ms=duration_ms,
        wisdom_id=wisdom_id,
        error=error[:200] if error else None,
    )

    with _db() as conn:
        row = conn.execute("SELECT steps FROM run_records WHERE id = ?", (run_id,)).fetchone()
        if not row:
            logger.warning(f"Run {run_id} not found")
            return False

        steps = json.loads(row["steps"] or "[]")
        steps.append({
            "tool_name": step.tool_name,
            "arguments_summary": step.arguments_summary,
            "result_summary": step.result_summary,
            "success": step.success,
            "timestamp": step.timestamp,
            "duration_ms": step.duration_ms,
            "wisdom_id": step.wisdom_id,
            "error": step.error,
        })

        conn.execute(
            "UPDATE run_records SET steps = ? WHERE id = ?",
            (json.dumps(steps), run_id),
        )
        conn.commit()

    return True


def complete_run(
    run_id: str,
    outcome: RunOutcome,
    outcome_details: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> bool:
    """Complete a run with its outcome.

    Sets completed_at, outcome, token counts. Calculates total
    duration from started_at to now.
    """
    now = datetime.now(timezone.utc).isoformat()

    with _db() as conn:
        cursor = conn.execute(
            """
            UPDATE run_records
            SET completed_at = ?, outcome = ?, outcome_details = ?,
                input_tokens = ?, output_tokens = ?
            WHERE id = ? AND outcome = 'pending'
            """,
            (now, outcome.value, outcome_details, input_tokens, output_tokens, run_id),
        )
        conn.commit()
        updated = cursor.rowcount > 0

    if updated:
        logger.info(f"Completed run {run_id}: {outcome.value}")
    return updated


def add_feedback(run_id: str, score: int, notes: str = "") -> bool:
    """Add user feedback to a completed run. Score: 1-5."""
    if not 1 <= score <= 5:
        raise ValueError("Score must be between 1 and 5")

    with _db() as conn:
        cursor = conn.execute(
            "UPDATE run_records SET user_feedback_score = ?, user_feedback_notes = ? WHERE id = ?",
            (score, notes, run_id),
        )
        conn.commit()
        return cursor.rowcount > 0


# ============================================================================
# QUERIES
# ============================================================================


def get_run(run_id: str) -> Optional[RunRecord]:
    """Get a run by ID."""
    with _db() as conn:
        row = conn.execute("SELECT * FROM run_records WHERE id = ?", (run_id,)).fetchone()
        if row:
            return _run_from_row(row)
    return None


def get_recent_runs(limit: int = 20, outcome: Optional[RunOutcome] = None) -> list[RunRecord]:
    """Get recent runs, optionally filtered by outcome."""
    with _db() as conn:
        if outcome:
            rows = conn.execute(
                "SELECT * FROM run_records WHERE outcome = ? ORDER BY started_at DESC LIMIT ?",
                (outcome.value, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM run_records ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

    return [_run_from_row(row) for row in rows]


def get_failed_runs(limit: int = 20) -> list[RunRecord]:
    """Get recent failed runs for debugging/learning."""
    return get_recent_runs(limit=limit, outcome=RunOutcome.FAILURE)


def get_runs_by_source(source: str, limit: int = 20) -> list[RunRecord]:
    """Get runs triggered by a specific source."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM run_records WHERE source = ? ORDER BY started_at DESC LIMIT ?",
            (source, limit),
        ).fetchall()

    return [_run_from_row(row) for row in rows]


def search_runs(query: str, limit: int = 10) -> list[RunRecord]:
    """Full-text search across run goals and outcomes."""
    from maasv.core.db import _sanitize_fts_input

    sanitized = _sanitize_fts_input(query)
    if not sanitized:
        return []

    with _db() as conn:
        try:
            rows = conn.execute(
                """
                SELECT r.* FROM run_records r
                JOIN run_records_fts fts ON r.rowid = fts.rowid
                WHERE run_records_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (sanitized, limit),
            ).fetchall()
            return [_run_from_row(row) for row in rows]
        except Exception:
            logger.debug("FTS search failed for run_records", exc_info=True)
            return []


def get_pending_runs() -> list[RunRecord]:
    """Get runs that are still in progress (not completed)."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM run_records WHERE outcome = 'pending' ORDER BY started_at ASC"
        ).fetchall()

    return [_run_from_row(row) for row in rows]


# ============================================================================
# ANALYSIS
# ============================================================================


def get_stats() -> dict:
    """Get aggregate statistics about runs."""
    with _db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM run_records").fetchone()[0]

        by_outcome = conn.execute(
            "SELECT outcome, COUNT(*) as count FROM run_records GROUP BY outcome"
        ).fetchall()

        by_source = conn.execute(
            "SELECT source, COUNT(*) as count FROM run_records WHERE source != '' GROUP BY source ORDER BY count DESC"
        ).fetchall()

        by_tier = conn.execute(
            "SELECT model_tier, COUNT(*) as count FROM run_records WHERE model_tier != '' GROUP BY model_tier"
        ).fetchall()

        avg_tokens = conn.execute(
            "SELECT AVG(input_tokens + output_tokens) as avg_total FROM run_records WHERE outcome != 'pending'"
        ).fetchone()

        avg_score = conn.execute(
            "SELECT AVG(user_feedback_score) FROM run_records WHERE user_feedback_score IS NOT NULL"
        ).fetchone()

        # Avg steps per completed run
        completed = conn.execute(
            "SELECT steps FROM run_records WHERE outcome != 'pending'"
        ).fetchall()
        step_counts = []
        for row in completed:
            steps = json.loads(row["steps"] or "[]")
            step_counts.append(len(steps))

    return {
        "total_runs": total,
        "by_outcome": {row["outcome"]: row["count"] for row in by_outcome},
        "by_source": {row["source"]: row["count"] for row in by_source},
        "by_tier": {row["model_tier"]: row["count"] for row in by_tier},
        "avg_tokens_per_run": round(avg_tokens[0] or 0, 0),
        "avg_feedback_score": round(avg_score[0], 2) if avg_score[0] else None,
        "avg_steps_per_run": round(sum(step_counts) / len(step_counts), 1) if step_counts else 0,
    }


def get_tool_success_rates(limit: int = 20) -> dict[str, dict]:
    """Analyze tool success rates across all runs.

    Returns per-tool stats: total_calls, successes, failures, success_rate.
    """
    with _db() as conn:
        rows = conn.execute("SELECT steps FROM run_records").fetchall()

    tool_stats: dict[str, dict] = {}
    for row in rows:
        steps = json.loads(row["steps"] or "[]")
        for step in steps:
            name = step.get("tool_name", "unknown")
            if name not in tool_stats:
                tool_stats[name] = {"total": 0, "success": 0, "failure": 0}
            tool_stats[name]["total"] += 1
            if step.get("success", True):
                tool_stats[name]["success"] += 1
            else:
                tool_stats[name]["failure"] += 1

    # Calculate rates and sort by total calls
    for stats in tool_stats.values():
        stats["success_rate"] = stats["success"] / stats["total"] if stats["total"] > 0 else 0

    sorted_tools = dict(
        sorted(tool_stats.items(), key=lambda x: x[1]["total"], reverse=True)[:limit]
    )
    return sorted_tools


def get_failure_patterns(limit: int = 10) -> list[dict]:
    """Analyze common failure patterns across runs.

    Returns the most common tool sequences that led to failures.
    """
    with _db() as conn:
        rows = conn.execute(
            "SELECT goal, steps, outcome_details FROM run_records WHERE outcome = 'failure' ORDER BY started_at DESC LIMIT ?",
            (limit * 2,),  # over-fetch to get enough after grouping
        ).fetchall()

    patterns = []
    for row in rows:
        steps = json.loads(row["steps"] or "[]")
        failed_tools = [s["tool_name"] for s in steps if not s.get("success", True)]
        if failed_tools:
            patterns.append({
                "goal": row["goal"][:100],
                "failed_tools": failed_tools,
                "details": row["outcome_details"][:200] if row["outcome_details"] else "",
                "total_steps": len(steps),
            })

    return patterns[:limit]


# ============================================================================
# FORMATTING
# ============================================================================


def format_run_for_prompt(run: RunRecord) -> str:
    """Format a run record for LLM prompt injection."""
    lines = [f"**Run: {run.goal}**"]
    lines.append(f"Outcome: {run.outcome.value}")

    if run.steps:
        lines.append(f"Steps ({len(run.steps)}):")
        for i, step in enumerate(run.steps, 1):
            status = "OK" if step.success else "FAIL"
            lines.append(f"  {i}. [{status}] {step.tool_name}: {step.arguments_summary[:60]}")
            if step.error:
                lines.append(f"     Error: {step.error[:80]}")

    if run.outcome_details:
        lines.append(f"Details: {run.outcome_details[:200]}")

    if run.user_feedback_score:
        lines.append(f"Feedback: {run.user_feedback_score}/5")
        if run.user_feedback_notes:
            lines.append(f"  Notes: {run.user_feedback_notes[:100]}")

    return "\n".join(lines)


def format_stats_for_prompt(stats: dict) -> str:
    """Format run statistics for LLM context."""
    lines = ["## Task Execution History"]
    lines.append(f"Total runs: {stats['total_runs']}")

    by_outcome = stats.get("by_outcome", {})
    if by_outcome:
        parts = [f"{k}: {v}" for k, v in by_outcome.items()]
        lines.append(f"Outcomes: {', '.join(parts)}")

    if stats.get("avg_feedback_score"):
        lines.append(f"Avg feedback: {stats['avg_feedback_score']}/5")

    if stats.get("avg_steps_per_run"):
        lines.append(f"Avg steps/run: {stats['avg_steps_per_run']}")

    return "\n".join(lines)

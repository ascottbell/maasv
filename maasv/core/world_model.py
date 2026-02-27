"""
maasv World Model — Queryable state of a person's life.

The world model sits on top of maasv's knowledge graph and provides:
- Rich entity schemas for people, projects, routines, locations, devices
- Bi-temporal queries (event_time + ingestion_time)
- Confidence tracking and staleness detection
- Current activity hypotheses
- Relevance decay implementation

This is not a replacement for the KG — it's a semantic layer that makes
the KG queryable in terms of "what is true right now" rather than
"what triples exist in the graph."
"""

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from maasv.core.db import _db, _plain_db
from maasv.core.graph import (
    add_relationship,
    create_entity,
    expire_relationship,
    find_entity_by_name,
    find_or_create_entity,
    get_entity,
    get_entity_profile,
    get_entity_relationships,
    get_entities_by_type,
    graph_query,
    search_entities,
)

logger = logging.getLogger(__name__)


# ============================================================================
# ENTITY SCHEMAS — Rich metadata schemas per entity type
# ============================================================================

# These define what metadata fields are expected for each entity type.
# The actual storage is in the generic entities.metadata JSON column.
# Schemas are for validation/documentation, not enforcement — we don't
# reject entities missing fields, we just know what to look for.

ENTITY_SCHEMAS = {
    "person": {
        "fields": {
            "roles": "list[str]",  # friend, colleague, family, doctor, etc.
            "communication_norms": "str",  # e.g. "prefers text over email"
            "boundaries": "str",  # e.g. "don't contact after 9pm"
            "relationship_to_adam": "str",  # e.g. "wife", "coworker", "friend"
            "last_interaction": "str",  # ISO datetime
            "interaction_frequency": "str",  # daily, weekly, monthly, rare
            "email": "str",
            "phone": "str",
            "birthday": "str",  # ISO date
        },
    },
    "project": {
        "fields": {
            "state": "str",  # backlog, active, blocked, paused, done, archived
            "next_actions": "list[str]",
            "blockers": "list[str]",
            "deadline": "str",  # ISO datetime
            "owner": "str",  # who owns this
            "collaborators": "list[str]",
            "tech_stack": "list[str]",
            "repo_path": "str",  # local filesystem path
            "repo_url": "str",  # remote URL
        },
    },
    "routine": {
        "fields": {
            "schedule": "str",  # cron-like or description: "weekdays 7am"
            "day_pattern": "list[str]",  # ["monday", "wednesday", "friday"]
            "time_of_day": "str",  # morning, midday, afternoon, evening, night
            "duration_minutes": "int",
            "participants": "list[str]",  # entity names involved
            "exceptions": "list[str]",  # known exceptions/overrides
            "last_occurrence": "str",  # ISO datetime
        },
    },
    "location": {
        "fields": {
            "address": "str",
            "type": "str",  # home, office, school, restaurant, etc.
            "typical_occupants": "list[str]",
            "typical_activities": "list[str]",  # what happens here
            "current_occupants": "list[str]",  # who's there now (if known)
            "lat": "float",
            "lon": "float",
        },
    },
    "device": {
        "fields": {
            "device_type": "str",  # phone, laptop, speaker, light, etc.
            "capabilities": "list[str]",
            "state": "str",  # online, offline, error, unknown
            "failure_history": "list[str]",  # recent failures
            "last_health_check": "str",  # ISO datetime
            "integration": "str",  # homeassistant, apple, etc.
        },
    },
}


# ============================================================================
# CONFIDENCE AND STALENESS
# ============================================================================


def compute_decay(
    created_at: str,
    half_life_days: int = 30,
    now: Optional[datetime] = None,
) -> float:
    """Compute temporal decay factor for a fact.

    Uses exponential decay: factor = 0.5^(age_days / half_life_days).
    Returns a value in (0.0, 1.0] where 1.0 means brand new.

    Args:
        created_at: ISO datetime string of when the fact was established
        half_life_days: Days for decay to reach 0.5
        now: Current time (defaults to UTC now)

    Returns:
        Decay factor in (0.0, 1.0]
    """
    if now is None:
        now = datetime.now(timezone.utc)

    try:
        ts = datetime.fromisoformat(created_at)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return 0.5  # Unknown age, assume middle

    age_days = (now - ts).total_seconds() / 86400.0
    if age_days <= 0:
        return 1.0

    if half_life_days <= 0:
        return 0.0

    return math.pow(0.5, age_days / half_life_days)


def get_effective_confidence(
    base_confidence: float,
    created_at: str,
    half_life_days: int = 30,
    now: Optional[datetime] = None,
) -> float:
    """Compute effective confidence = base_confidence * decay_factor.

    A fact with confidence 1.0 that's 30 days old (with 30-day half life)
    has effective confidence 0.5.
    """
    decay = compute_decay(created_at, half_life_days, now)
    return base_confidence * decay


# ============================================================================
# STALENESS DETECTION
# ============================================================================


@dataclass
class StaleFact:
    """A fact that may be outdated."""

    entity_id: str
    entity_name: str
    entity_type: str
    relationship_id: Optional[str]
    predicate: Optional[str]
    value: Optional[str]
    confidence: float
    effective_confidence: float
    age_days: float
    last_verified: Optional[str]  # last_accessed_at as proxy


def get_stale_facts(threshold_days: int = 30, min_confidence: float = 0.3) -> list[StaleFact]:
    """Find facts whose effective confidence has decayed below a threshold.

    Checks both entity metadata staleness and relationship staleness.
    Returns facts ordered by effective confidence (most stale first).
    """
    import maasv

    config = maasv.get_config()
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=threshold_days)).isoformat()
    results = []

    with _db() as db:
        # Stale relationships: active relationships that haven't been updated recently
        rows = db.execute(
            """
            SELECT r.id as rel_id, r.subject_id, r.predicate, r.object_value,
                   r.confidence, r.valid_from, r.created_at,
                   e.name as entity_name, e.entity_type, e.last_accessed_at
            FROM relationships r
            JOIN entities e ON r.subject_id = e.id
            WHERE r.valid_to IS NULL
            AND r.created_at < ?
            ORDER BY r.created_at ASC
            LIMIT 200
            """,
            (cutoff,),
        ).fetchall()

        for row in rows:
            row = dict(row)
            # Get category-specific half life
            etype = row["entity_type"]
            half_life = config.category_half_life_days.get(etype, config.decay_half_life_days)
            eff_conf = get_effective_confidence(
                row["confidence"],
                row["created_at"],
                half_life,
                now,
            )
            if eff_conf < min_confidence:
                ts = datetime.fromisoformat(row["created_at"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age = (now - ts).total_seconds() / 86400.0

                results.append(
                    StaleFact(
                        entity_id=row["subject_id"],
                        entity_name=row["entity_name"],
                        entity_type=row["entity_type"],
                        relationship_id=row["rel_id"],
                        predicate=row["predicate"],
                        value=row["object_value"],
                        confidence=row["confidence"],
                        effective_confidence=eff_conf,
                        age_days=age,
                        last_verified=row["last_accessed_at"],
                    )
                )

    results.sort(key=lambda f: f.effective_confidence)
    return results


def get_stale_entities(threshold_days: int = 60) -> list[dict]:
    """Find entities that haven't been accessed or updated in a long time.

    These are candidates for verification — the world model should prompt
    questions like "Are you still working on X?" or "Is Y still at Z?"
    """
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=threshold_days)).strftime("%Y-%m-%d %H:%M:%S")

    with _db() as db:
        rows = db.execute(
            """
            SELECT id, name, entity_type, updated_at, last_accessed_at,
                   access_count, metadata
            FROM entities
            WHERE (last_accessed_at IS NULL OR last_accessed_at < ?)
            AND (updated_at IS NULL OR updated_at < ?)
            ORDER BY COALESCE(last_accessed_at, updated_at, created_at) ASC
            LIMIT 50
            """,
            (cutoff, cutoff),
        ).fetchall()

    results = []
    for row in rows:
        d = dict(row)
        if d.get("metadata"):
            d["metadata"] = json.loads(d["metadata"])

        # Compute age
        ref_time = d.get("last_accessed_at") or d.get("updated_at")
        if ref_time:
            try:
                ts = datetime.fromisoformat(ref_time)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                d["days_since_activity"] = (now - ts).total_seconds() / 86400.0
            except ValueError:
                d["days_since_activity"] = None
        else:
            d["days_since_activity"] = None

        results.append(d)

    return results


# ============================================================================
# WORLD STATE QUERIES
# ============================================================================


@dataclass
class EntityState:
    """Current state of a world model entity with computed fields."""

    id: str
    name: str
    entity_type: str
    metadata: dict
    relationships: list[dict]
    effective_confidence: float
    staleness_days: float
    last_activity: Optional[str]


def get_entity_state(entity_id: str) -> Optional[EntityState]:
    """Get full current state of an entity with confidence and staleness.

    Enriches the raw entity profile with world-model computed fields:
    effective confidence, staleness, and activity tracking.
    """
    import maasv

    config = maasv.get_config()
    profile = get_entity_profile(entity_id)
    if not profile or not profile.get("entity"):
        return None

    entity = profile["entity"]
    now = datetime.now(timezone.utc)

    # Compute staleness
    ref_time = entity.get("last_accessed_at") or entity.get("updated_at") or entity.get("created_at")
    staleness_days = 0.0
    if ref_time:
        try:
            ts = datetime.fromisoformat(ref_time)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            staleness_days = (now - ts).total_seconds() / 86400.0
        except ValueError:
            pass

    # Compute average effective confidence across relationships
    rels = get_entity_relationships(entity_id, include_expired=False)
    etype = entity.get("entity_type", "")
    half_life = config.category_half_life_days.get(etype, config.decay_half_life_days)

    confidences = []
    for rel in rels:
        created = rel.get("created_at", "")
        if created:
            eff = get_effective_confidence(rel.get("confidence", 1.0), created, half_life, now)
            confidences.append(eff)

    avg_confidence = sum(confidences) / len(confidences) if confidences else 1.0

    # Flatten relationships for output
    rel_list = []
    for pred, entries in profile.get("relationships", {}).items():
        for entry in entries:
            rel_list.append({"predicate": pred, **entry})

    return EntityState(
        id=entity["id"],
        name=entity["name"],
        entity_type=entity.get("entity_type", ""),
        metadata=entity.get("metadata") or {},
        relationships=rel_list,
        effective_confidence=avg_confidence,
        staleness_days=staleness_days,
        last_activity=ref_time,
    )


def get_people(limit: int = 50) -> list[dict]:
    """Get all person entities with their metadata."""
    return get_entities_by_type("person", limit=limit)


def get_projects(active_only: bool = True) -> list[dict]:
    """Get project entities, optionally filtered to active states."""
    projects = get_entities_by_type("project", limit=100)
    if active_only:
        active_states = {"active", "in_progress", "blocked"}
        projects = [
            p for p in projects
            if (p.get("metadata") or {}).get("state", "active") in active_states
        ]
    return projects


def get_locations() -> list[dict]:
    """Get all known locations."""
    return get_entities_by_type("location", limit=50)


def get_routines() -> list[dict]:
    """Get all routine entities."""
    return get_entities_by_type("routine", limit=50)


def get_devices() -> list[dict]:
    """Get all device entities."""
    return get_entities_by_type("device", limit=50)


# ============================================================================
# ACTIVITY HYPOTHESIS
# ============================================================================


@dataclass
class ActivityHypothesis:
    """Best guess at what Adam is currently doing."""

    activity: str  # e.g. "working", "family time", "sleeping"
    confidence: float  # 0.0-1.0
    location: Optional[str] = None  # best guess location
    reasoning: str = ""  # why we think this
    based_on: list[str] = field(default_factory=list)  # evidence sources


def get_current_activity(
    current_hour: Optional[int] = None,
    day_of_week: Optional[int] = None,
) -> ActivityHypothesis:
    """Estimate what Adam is likely doing right now.

    Combines:
    - Time-of-day heuristics (same as sitrep engine uses)
    - Day-of-week patterns
    - Known routines (if any match current time)
    - Calendar data (if available via entities)

    Args:
        current_hour: Override for testing (0-23)
        day_of_week: Override for testing (0=Monday, 6=Sunday)

    Returns:
        Best-guess ActivityHypothesis
    """
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("America/New_York")
    now = datetime.now(tz)
    hour = current_hour if current_hour is not None else now.hour
    weekday = day_of_week if day_of_week is not None else now.weekday()
    is_weekend = weekday >= 5

    evidence = [f"time={hour}:00", f"day={now.strftime('%A') if day_of_week is None else weekday}"]

    # Time-based heuristics
    if hour < 6:
        activity = "sleeping"
        location = "home"
        confidence = 0.9
    elif hour < 7:
        activity = "waking up"
        location = "home"
        confidence = 0.7
    elif hour < 9:
        if is_weekend:
            activity = "morning routine with family"
            confidence = 0.6
        else:
            activity = "getting kids ready for school"
            confidence = 0.7
        location = "home"
    elif hour < 12:
        if is_weekend:
            activity = "family time or personal projects"
            confidence = 0.5
            location = None
        else:
            activity = "working"
            confidence = 0.8
            location = "home office"
    elif hour < 13:
        activity = "lunch"
        confidence = 0.5
        location = None
    elif hour < 17:
        if is_weekend:
            activity = "family time or errands"
            confidence = 0.5
            location = None
        else:
            activity = "working"
            confidence = 0.8
            location = "home office"
    elif hour < 20:
        activity = "family time or dinner"
        confidence = 0.6
        location = "home"
    elif hour < 22:
        activity = "winding down, kids bedtime"
        confidence = 0.6
        location = "home"
    else:
        activity = "sleeping or about to sleep"
        location = "home"
        confidence = 0.7

    reasoning = f"Based on time ({hour}:00) and day ({'weekend' if is_weekend else 'weekday'})"

    # Check routines for this time slot
    routines = get_routines()
    for routine in routines:
        meta = routine.get("metadata") or {}
        time_of_day = meta.get("time_of_day", "")
        day_pattern = meta.get("day_pattern", [])

        # Simple time-of-day matching
        matches_time = False
        if time_of_day == "morning" and 6 <= hour < 12:
            matches_time = True
        elif time_of_day == "afternoon" and 12 <= hour < 17:
            matches_time = True
        elif time_of_day == "evening" and 17 <= hour < 22:
            matches_time = True

        if matches_time:
            # Check day pattern
            day_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            current_day = day_names[weekday]
            if not day_pattern or current_day in [d.lower() for d in day_pattern]:
                activity = f"{routine['name']} (routine)"
                confidence = min(confidence + 0.1, 0.95)
                evidence.append(f"routine:{routine['name']}")
                reasoning += f"; matches routine '{routine['name']}'"
                break

    return ActivityHypothesis(
        activity=activity,
        confidence=confidence,
        location=location,
        reasoning=reasoning,
        based_on=evidence,
    )


# ============================================================================
# BI-TEMPORAL QUERIES
# ============================================================================


def time_travel_query(
    entity_type: Optional[str] = None,
    predicate: Optional[str] = None,
    as_of: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """Query the knowledge graph as it existed at a point in time.

    Uses the bi-temporal model:
    - valid_from/valid_to: when the fact was true in the real world
    - ingested_at: when Doris learned about it

    If as_of is provided, returns facts that were:
    - Already ingested by that time (ingested_at <= as_of)
    - Valid at that time (valid_from <= as_of AND (valid_to IS NULL OR valid_to > as_of))

    Args:
        entity_type: Filter by entity type
        predicate: Filter by relationship predicate
        as_of: ISO datetime to query. If None, returns current state.
        limit: Max results

    Returns:
        List of relationship dicts with entity info
    """
    if as_of is None:
        # Current state — delegate to existing graph_query
        return graph_query(
            subject_type=entity_type,
            predicate=predicate,
            include_expired=False,
            limit=limit,
        )

    query = """
        SELECT r.*,
               e_subj.name as subject_name, e_subj.entity_type as subject_type,
               e_obj.name as object_name, e_obj.entity_type as object_type
        FROM relationships r
        JOIN entities e_subj ON r.subject_id = e_subj.id
        LEFT JOIN entities e_obj ON r.object_id = e_obj.id
        WHERE r.valid_from <= ?
        AND (r.valid_to IS NULL OR r.valid_to > ?)
        AND (r.ingested_at IS NULL OR r.ingested_at <= ?)
    """
    params: list = [as_of, as_of, as_of]

    if entity_type:
        query += " AND e_subj.entity_type = ?"
        params.append(entity_type)
    if predicate:
        query += " AND r.predicate = ?"
        params.append(predicate)

    query += " ORDER BY r.valid_from DESC LIMIT ?"
    params.append(limit)

    with _db() as db:
        rows = db.execute(query, params).fetchall()

    results = []
    for row in rows:
        row_dict = dict(row)
        if row_dict.get("metadata"):
            row_dict["metadata"] = json.loads(row_dict["metadata"])
        results.append(row_dict)

    return results


# ============================================================================
# ENTITY METADATA HELPERS
# ============================================================================


def update_entity_metadata(entity_id: str, updates: dict) -> bool:
    """Merge metadata updates into an entity's existing metadata.

    This is the primary way to update world model state for entities.
    For example, marking a project as "blocked" or updating a person's
    last_interaction time.

    Args:
        entity_id: Entity to update
        updates: Dict of metadata keys to set/overwrite

    Returns:
        True if entity was found and updated
    """
    with _db() as db:
        row = db.execute("SELECT metadata FROM entities WHERE id = ?", (entity_id,)).fetchone()
        if not row:
            return False

        current = json.loads(row["metadata"]) if row["metadata"] else {}
        current.update(updates)

        db.execute(
            "UPDATE entities SET metadata = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (json.dumps(current), entity_id),
        )
        db.commit()

    return True


def set_project_state(project_name: str, state: str) -> bool:
    """Convenience: update a project's state in the world model.

    Valid states: backlog, active, blocked, paused, done, archived.
    """
    valid_states = {"backlog", "active", "blocked", "paused", "done", "archived"}
    if state not in valid_states:
        raise ValueError(f"Invalid project state: {state!r}. Valid: {valid_states}")

    entity = find_entity_by_name(project_name, "project")
    if not entity:
        logger.warning(f"Project entity not found: {project_name}")
        return False

    return update_entity_metadata(entity["id"], {"state": state})


def record_interaction(person_name: str, interaction_time: Optional[str] = None) -> bool:
    """Record that Adam interacted with a person. Updates last_interaction."""
    if interaction_time is None:
        interaction_time = datetime.now(timezone.utc).isoformat()

    entity = find_entity_by_name(person_name, "person")
    if not entity:
        logger.warning(f"Person entity not found: {person_name}")
        return False

    return update_entity_metadata(entity["id"], {"last_interaction": interaction_time})


def update_device_state(device_name: str, state: str) -> bool:
    """Update a device's current state in the world model."""
    entity = find_entity_by_name(device_name, "device")
    if not entity:
        logger.warning(f"Device entity not found: {device_name}")
        return False

    updates = {
        "state": state,
        "last_health_check": datetime.now(timezone.utc).isoformat(),
    }
    return update_entity_metadata(entity["id"], updates)


# ============================================================================
# WORLD MODEL SUMMARY — for prompt injection
# ============================================================================


def get_world_summary(include_stale: bool = False) -> str:
    """Build a compact world state summary for LLM context.

    Returns a formatted string with:
    - Active projects and their states
    - Key people and recent interactions
    - Current activity hypothesis
    - Stale facts needing verification (optional)
    """
    lines = ["## World State\n"]

    # Activity hypothesis
    activity = get_current_activity()
    lines.append(f"**Current activity estimate:** {activity.activity} "
                 f"(confidence: {activity.confidence:.0%})")
    if activity.location:
        lines.append(f"**Likely location:** {activity.location}")
    lines.append("")

    # Active projects
    projects = get_projects(active_only=True)
    if projects:
        lines.append("**Active Projects:**")
        for p in projects[:10]:
            meta = p.get("metadata") or {}
            state = meta.get("state", "active")
            blockers = meta.get("blockers", [])
            line = f"- {p['name']} [{state}]"
            if blockers:
                line += f" (blocked: {', '.join(blockers[:2])})"
            lines.append(line)
        lines.append("")

    # Key people with recent interactions
    people = get_people(limit=20)
    recent_people = []
    for p in people:
        meta = p.get("metadata") or {}
        last = meta.get("last_interaction")
        if last:
            recent_people.append((p, last))

    if recent_people:
        recent_people.sort(key=lambda x: x[1], reverse=True)
        lines.append("**Recent interactions:**")
        for p, last in recent_people[:5]:
            meta = p.get("metadata") or {}
            role = meta.get("relationship_to_adam", "")
            role_str = f" ({role})" if role else ""
            lines.append(f"- {p['name']}{role_str}: last {last[:10]}")
        lines.append("")

    # Stale facts
    if include_stale:
        stale = get_stale_facts(threshold_days=30, min_confidence=0.3)
        if stale:
            lines.append(f"**Stale facts ({len(stale)} need verification):**")
            for sf in stale[:5]:
                lines.append(
                    f"- {sf.entity_name}.{sf.predicate} = {sf.value} "
                    f"(eff. confidence: {sf.effective_confidence:.0%}, "
                    f"age: {sf.age_days:.0f}d)"
                )
            lines.append("")

    return "\n".join(lines)

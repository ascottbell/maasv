# MAASV-23: Proactive Trigger Engine — Research

## Problem Statement

Today maasv is purely pull-based: Doris asks, maasv responds. Every interaction follows the same pattern — query comes in, retrieval runs, results go out. Nothing happens inside maasv unless someone pokes it.

This means Doris misses opportunities where the cognition layer *knows* something relevant happened but has no way to tell anyone:

1. **New person added to the graph** — Doris could proactively ask Adam for context ("You just mentioned Sarah from TerryAnn — want me to remember her role?") but doesn't know it happened until the next query.

2. **Relationship change** — Adam's birthday relationship to Levi gets updated, but Doris can't preemptively check if a reminder should be set.

3. **Memory contradiction detected** — Sleep-time review stores an insight that contradicts an existing memory, but nothing alerts Doris to resolve the conflict.

4. **Wisdom pattern emerges** — After 10 failed escalation decisions from the same email source, Doris should adjust her behavior, but only finds out when she happens to query wisdom before the next email.

5. **Graph reaches a threshold** — A project entity accumulates 20+ relationships in a week (a burst of activity), signaling Adam is deeply engaged with it — useful context Doris could surface without being asked.

6. **Memory decay approaching** — A low-confidence memory about a restaurant is about to be pruned by hygiene, but Adam mentioned it yesterday — the access should have boosted it, but the timing didn't line up.

The question is whether this belongs in maasv (the cognition layer) or in Doris (the agent). There are legitimate arguments both ways.

## Current State in maasv

### Event Sources That Already Exist

These are the places in maasv where something "happens" that could be a trigger source:

**Memory Storage** (`maasv/core/store.py`):
- `store_memory()` (line 24) — New memory created. Returns memory_id. Also does dedup check — if a near-duplicate is found (line 87-91), it returns the existing ID without storing. Both cases are events: "new memory stored" and "duplicate detected."
- `supersede_memory()` (line 126) — Old memory replaced by new one. This is a correction event — the system's understanding changed.
- `delete_memory()` (line 280) — Memory permanently removed.
- `update_memory_metadata()` (line 305) — Metadata changed on existing memory.

**Knowledge Graph** (`maasv/core/graph.py`):
- `create_entity()` (line 255) — New entity in the graph. Also handles IntegrityError for race conditions (line 279).
- `find_or_create_entity()` (line 352) — Either finds existing or creates new. The "create" path is an event.
- `add_relationship()` (line 628) — New relationship between entities. Handles dedup (returns existing if same triple exists, line 691-696). Both "new relationship" and "relationship confidence updated" are events.
- `expire_relationship()` (line 751) — Relationship marked as no longer current.
- `update_relationship_value()` (line 912) — Old relationship expired, new one created (temporal update).
- `merge_entity()` (line 388) — Duplicate entities consolidated. Complex event: relationships reassigned, entities deleted.

**Wisdom System** (`maasv/core/wisdom.py`):
- `log_reasoning()` (line 99) — New wisdom entry logged (action about to happen).
- `record_outcome()` (line 140) — Outcome recorded for an action.
- `add_feedback()` (line 150) — Human feedback attached to wisdom entry.
- `log_escalation_miss()` (line 539) — Missed escalation pattern recorded.

**Entity Extraction** (`maasv/server/routers/extraction.py`):
- `extract()` (line 17) — Entities and relationships extracted from text. Bulk event — multiple entities/relationships created at once.

### Background Infrastructure (Lifecycle)

**SleepWorker** (`maasv/lifecycle/worker.py`):
- Priority queue-based background worker (line 40), daemon thread.
- Job types: INFERENCE, REVIEW, REORGANIZE, MEMORY_HYGIENE, LEARN (line 19-24).
- Cancellable jobs via `threading.Event` (line 44).
- Idle monitor (line 189) — watches for activity gaps, triggers sleep work when idle, cancels when active.
- Singleton pattern via `get_sleep_worker()` (line 169).

This is the closest thing to an event-driven system maasv has today. The idle monitor is already an event-based pattern: "idle detected" triggers work, "activity detected" cancels it. But it's a simple binary state machine, not a general-purpose event bus.

**Inference** (`maasv/lifecycle/inference.py`): Resolves vague references in conversations, stores as relationships.

**Review** (`maasv/lifecycle/review.py`): Second-pass conversation analysis, stores insights as memories.

**Learn** (`maasv/lifecycle/learn.py`): Labels retrieval logs, trains the learned ranker, checks graduation readiness.

**Reorganize** (`maasv/lifecycle/reorganize.py`): Caches common graph paths, cleans orphaned entities.

**Memory Hygiene** (`maasv/lifecycle/memory_hygiene.py`): Dedup, prune, consolidate — the full maintenance pipeline.

### HTTP API (`maasv/server/`)

FastAPI server with auth (line 13, `main.py`). Routers for memory, graph, wisdom, extraction, health. All routes are synchronous `def` (not `async def`) — FastAPI runs them in a threadpool. No WebSocket support currently. No callback/webhook registration endpoints.

### What's Missing

1. **No event bus** — Functions just return values. There's no pub/sub, no observer pattern, no callback mechanism.
2. **No event log** — Changes happen and are gone. The `retrieval_log` table (migration 5) tracks queries but not mutations.
3. **No webhook infrastructure** — The server can receive HTTP requests but has no mechanism to *send* them.
4. **No trigger registration** — No way to say "when X happens, do Y."

## Trigger Patterns That Actually Matter

### High-Value Triggers for a Personal Assistant

These are triggers where the delay between "event happens" and "Doris finds out" actually costs something:

1. **New person mentioned** — Entity created with type "person." Doris could proactively ask: "Who is [name]? Want me to remember anything about them?" This is high-value because context captured at mention-time is richer than context recalled later.

2. **Relationship contradiction** — New relationship conflicts with existing one (e.g., "Adam lives_in Brooklyn" when "Adam lives_in Upper West Side" already exists). The predicate-normalization system (graph.py line 142-203) already does semantic matching — contradictions are a natural extension.

3. **Escalation pattern spike** — Wisdom logs N consecutive `*_escalation_miss` entries for a source. Doris should adjust her escalation threshold for that source without waiting for the next email.

4. **Memory cluster forming** — Multiple memories stored about the same subject in a short window. Signals a topic Adam is focused on — Doris could preemptively load relevant context.

5. **Birthday/date approaching** — A `has_birthday` relationship value is within N days. Classic reminder trigger.

6. **Correction chain** — Multiple `supersede_memory()` calls on the same topic in a session. Something is unstable in the knowledge base — might warrant human review.

7. **Sleep-time discovery** — Review job finds a high-confidence insight that changes existing understanding. Worth surfacing proactively rather than waiting for the next relevant query.

### Low-Value / Annoying Triggers

These sound useful in theory but would generate noise:

1. **Every memory store** — "I stored a memory!" is not useful. Doris already knows — she asked maasv to store it.

2. **Access count thresholds** — "Memory X was accessed 50 times!" Who cares? This is an internal optimization signal, not a user-facing event.

3. **Hygiene operations** — "I pruned 12 stale memories." Administrative noise. Log it, don't notify.

4. **Entity dedup events** — "I merged 'React Native' with 'react_native'." Internal cleanup. Only notify if it changes something user-visible.

5. **Learned ranker updates** — "Model retrained, NDCG improved 2%." This is telemetry, not a trigger.

6. **Every graph query result** — "Someone queried relationships for Adam." The consumer already knows — they made the query.

The distinction: **triggers should fire when maasv knows something the consumer doesn't**. If the consumer initiated the action, they already know.

## Architecture Options

### Option A: SQLite Triggers + Events Table (Polling)

Use SQLite's built-in `CREATE TRIGGER` to write to an `events` table on every INSERT/UPDATE/DELETE to core tables. Consumers poll the events table.

**How it works:**
```sql
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,      -- 'memory.stored', 'entity.created', etc.
    source_table TEXT NOT NULL,    -- 'memories', 'entities', 'relationships'
    source_id TEXT NOT NULL,       -- ID of the affected row
    data TEXT,                     -- JSON payload (old/new values)
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    consumed_at TEXT               -- NULL until a consumer marks it read
);

CREATE TRIGGER memory_insert_event AFTER INSERT ON memories
BEGIN
    INSERT INTO events (event_type, source_table, source_id, data)
    VALUES ('memory.stored', 'memories', NEW.id,
            json_object('category', NEW.category, 'subject', NEW.subject));
END;
```

Consumers poll: `SELECT * FROM events WHERE consumed_at IS NULL ORDER BY id LIMIT 50`.

**Pros:**
- Zero dependencies. SQLite triggers are built-in and atomic with the operation.
- Events table becomes a durable audit log for free.
- Works across all DB access patterns (direct library calls, HTTP API, sleep worker).
- Consumer doesn't need to be running when events occur — they accumulate.
- Already have 10 SQLite triggers in the codebase (FTS sync) — proven pattern.

**Cons:**
- Polling adds latency (up to poll interval, typically 1-5 seconds).
- SQLite triggers fire at the SQL level, not the Python level — can't easily trigger on "deduplicate detected" (which is a Python-level decision at store.py line 87).
- Trigger logic is SQL, which is limited for complex conditions.
- Must be careful about trigger overhead on high-throughput paths (embedding insertion is already the bottleneck).

**Assessment:** Solid foundation. The events table is a good idea regardless of delivery mechanism. It captures what happened at the data layer even if the Python process crashes. But it can't capture Python-level events (dedup decisions, normalization, business logic).

### Option B: In-Process Event Bus (Python Callbacks)

Simple observer pattern. Functions emit events, registered callbacks handle them.

**How it works:**
```python
# maasv/core/events.py
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

_listeners: dict[str, list[Callable]] = defaultdict(list)

@dataclass
class Event:
    event_type: str
    source_id: str
    data: dict

def emit(event: Event):
    for listener in _listeners.get(event.event_type, []):
        try:
            listener(event)
        except Exception:
            logger.error("Event listener failed", exc_info=True)

def on(event_type: str, callback: Callable):
    _listeners[event_type].append(callback)
```

Then in `store_memory()`:
```python
emit(Event("memory.stored", memory_id, {"category": category, "subject": subject}))
```

**Pros:**
- Captures Python-level events (dedup decisions, normalization, any business logic).
- Zero latency — callbacks fire synchronously (or async via queue).
- Simple to implement — ~50 lines of code.
- Easy to test — mock the emit function.
- No DB overhead.

**Cons:**
- Only works within the same Python process. If maasv is used as a library (which it is — Doris imports maasv directly), the consumer must register callbacks at init time.
- If the process crashes between the operation and the callback, the event is lost.
- No durability — if no one is listening, the event vanishes.
- Synchronous callbacks on the store path add latency to every `store_memory()` call.

**Assessment:** Good for v1 if combined with option A's events table for durability. The bus handles real-time notification; the table handles crash recovery and offline consumers.

### Option C: Webhook-Based Notification

Register HTTP endpoints that maasv POSTs to when events occur.

**How it works:**
```python
# POST /v1/triggers/register
{
    "url": "http://localhost:8001/webhooks/maasv",
    "event_types": ["memory.stored", "entity.created"],
    "secret": "hmac-secret-for-verification"
}
```

When an event fires, maasv POSTs to the registered URL with HMAC signature.

**Pros:**
- Decoupled — consumer can be any HTTP server in any language.
- Standard pattern (Mem0 does this — see Prior Art).
- Works across process boundaries.

**Cons:**
- Requires HTTP infrastructure: registration storage, delivery queue, retry logic, HMAC signing.
- Adds significant complexity for a local-first, single-consumer system.
- Latency: HTTP round-trip for every event.
- Reliability: what happens when the consumer is down? Need a delivery queue, retry logic, dead letter handling.
- Overkill for the current use case (Doris and maasv run on the same machine, same process).

**Assessment:** This is the right pattern if maasv ever becomes a multi-tenant service. For the current local-first, single-consumer setup, it's over-engineered. But the events table from Option A makes this easy to add later — just add a webhook delivery worker that reads from the events table.

### Option D: WAL-Based CDC

Monitor SQLite's Write-Ahead Log for changes.

**How it works:**
SQLite provides `sqlite3_update_hook()` which fires a callback on every INSERT/UPDATE/DELETE. Python's `sqlite3` module doesn't expose this directly, but `ctypes` or the `apsw` library can. Alternatively, use the trigger-based CDC pattern from `kevinconway/sqlite-cdc`.

**Pros:**
- Captures all changes without modifying any application code.
- Zero overhead on the write path (the WAL is already being written).

**Cons:**
- `sqlite3_update_hook()` is not exposed in Python's standard `sqlite3` module.
- Requires either `apsw` (another dependency) or `ctypes` hacking (fragile).
- Fires at row level — need to reconstruct higher-level events from raw row changes.
- Connection-specific — each `sqlite3.Connection` has its own hook. maasv opens a new connection per operation (`_db()` context manager, db.py line 62-68), so the hook would need to be registered on every connection open.
- The trigger-based CDC approach (sqlite-cdc) is functionally equivalent to Option A but with an external dependency.

**Assessment:** Interesting but impractical for maasv. The connection-per-operation pattern means hooks would need registration on every `get_db()` call, and the raw row-level events need significant enrichment to be useful. Option A (SQLite triggers) achieves the same thing more cleanly.

## Trigger Storm Prevention

This is critical. A naive trigger system on a memory store that processes 100 memories in a batch (e.g., entity extraction from a long conversation) would fire 100 events. If each event triggers a webhook to Doris, and Doris reacts to each one, you get a feedback loop.

### Debouncing

Group events within a time window. Instead of firing on every `memory.stored`, accumulate them and fire once per window:

```python
# "5 memories stored in the last 3 seconds about subject 'TerryAnn'"
# → single trigger: "memory_burst" with count=5, subject="TerryAnn"
```

Implementation: Timer-based. On first event, start a timer (e.g., 3 seconds). Accumulate events. When timer fires, emit the debounced/aggregated event. Reset. The SleepWorker's idle monitor (worker.py line 211-234) already does something similar — check every N seconds, act on state change.

### Rate Limiting

Per-event-type rate limit. E.g., max 1 `entity.created` trigger per 10 seconds per entity type. Excess events are logged but not delivered.

### Circuit Breakers

If a consumer fails to acknowledge N consecutive events, stop delivering and log a warning. Resume after a backoff period. Prevents hammering a dead consumer.

### Priority Levels

Not all events are equal:

- **P0 (immediate):** Contradiction detected, escalation pattern spike. Fire immediately, no debounce.
- **P1 (batched):** New entity created, relationship added. Debounce to once per 5-10 seconds.
- **P2 (digest):** Memory stored, metadata updated. Accumulate and deliver as a periodic summary (every 30-60 seconds).
- **P3 (log only):** Hygiene operations, ranker updates. Write to events table but never fire a trigger.

### Cascade Prevention

Triggers must not be able to trigger themselves. If a trigger callback stores a new memory (which fires another trigger), you get infinite recursion. Solution: events emitted from within a trigger callback are tagged `source=trigger` and don't fire triggers themselves. This is the standard pattern from rule engines (Drools' "no-loop" attribute).

### Max Event Age

Events older than N minutes are auto-consumed (marked as stale) to prevent unbounded table growth and prevent processing events that are no longer relevant.

## Prior Art

### Mem0 Webhooks
Mem0's platform (not open source) supports webhooks for memory events (`memory_add`, `memory_update`, `memory_delete`). Webhooks are configured per-project with event type filtering. Payload includes event details, memory content, and event type. This is the closest prior art to what MAASV-23 proposes. Key difference: Mem0 is a hosted platform with many consumers; maasv is local-first with one consumer.
- [Mem0 Webhooks Documentation](https://docs.mem0.ai/platform/features/webhooks)

### Letta / MemGPT Sleep-Time Agents
Letta (formerly MemGPT) introduced the concept of "sleep-time agents" that reorganize and improve memory during idle periods — the same pattern as maasv's SleepWorker. Their key insight: memory management should happen asynchronously, not during conversation. maasv already implements this. The trigger engine would extend it by making sleep-time discoveries actionable (not just stored).
- [Letta Sleep-Time Agents](https://docs.letta.com/concepts/memgpt/)
- [Letta Agent Memory Blog](https://www.letta.com/blog/agent-memory)

### Graphlit
Graphlit takes a broader infrastructure approach — comprehensive semantic infrastructure spanning all data sources, not just conversations. They don't have explicit triggers but their event-driven pipeline (ingest → enrich → index) is relevant. The enrichment step (auto-extracting entities, generating summaries) is similar to what a trigger could do.
- [Graphlit Survey of AI Agent Memory Frameworks](https://www.graphlit.com/blog/survey-of-ai-agent-memory-frameworks)

### IFTTT / Zapier
The "when X then Y" pattern is the most intuitive trigger model. IFTTT does single-step (one trigger, one action). Zapier does multi-step with conditional logic. For maasv, the IFTTT model is the right starting point — single trigger, single action, no branching. Complexity can come later.
- [Zapier vs IFTTT Comparison](https://zapier.com/blog/zapier-vs-ifttt/)

### SQLite CDC (kevinconway/sqlite-cdc)
Trigger-based change data capture for SQLite. Writes before/after state to a changelog table per tracked table. Demonstrates that SQLite triggers are a proven mechanism for event capture. maasv already uses 10 SQLite triggers for FTS sync — adding more for event capture is a natural extension.
- [sqlite-cdc GitHub](https://github.com/kevinconway/sqlite-cdc)

### Python eventsourcing Library
Full event sourcing library with SQLite backend. Overkill for maasv's needs (we don't need event replay or CQRS), but validates that SQLite is a viable event store for small-scale systems.
- [eventsourcing PyPI](https://pypi.org/project/eventsourcing/)
- [eventsourcing SQLite docs](https://eventsourcing.readthedocs.io/en/stable/topics/persistence.html)

### Rete Algorithm / Rule Engines
Drools, CLIPS, and the Rete algorithm are the gold standard for rule-based event processing. Rete sacrifices memory for speed — builds a discrimination network of rule conditions for O(1) matching. This is massive overkill for maasv (we'll have maybe 10-50 rules, not 10,000). But one useful pattern to borrow: **conflict resolution** — when multiple rules match, which one fires? Priority-based ordering is the simplest strategy.
- [Rete Algorithm Wikipedia](https://en.wikipedia.org/wiki/Rete_algorithm)
- [Python Rule Engines](https://www.nected.ai/blog/python-rule-engines-automate-and-enforce-with-python)

### CHI 2025: Proactive Conversational Agents with Inner Thoughts
Research paper on agents that formulate "inner thoughts" during conversations and seek the right moment to contribute. Key finding: proactive agents significantly outperformed reactive baselines across all evaluation metrics, but knowing when to stay silent is critical. Directly relevant to the trigger engine — the engine needs to understand not just *what* to trigger but *when* silence is the right response.
- [CHI 2025 Paper](https://dl.acm.org/doi/10.1145/3706598.3713760)
- [ArXiv](https://arxiv.org/html/2501.00383v2)

### Google Sensible Agent (UIST 2025)
Framework for unobtrusive proactive AR agents. Key insight: anticipate user intentions and determine the best *approach* to deliver assistance, minimizing disruption. Relevant principle: the trigger itself is only half the problem — the delivery mechanism matters as much.
- [Sensible Agent Paper](https://dl.acm.org/doi/10.1145/3746059.3747748)
- [Google Research Blog](https://research.google/blog/sensible-agent-a-framework-for-unobtrusive-interaction-with-proactive-ar-agents/)

### Webhook Delivery Best Practices
Exponential backoff with jitter for retries. At-least-once delivery with idempotency keys. Queue-first ingestion (acknowledge fast, process async). Circuit breakers for troubled destinations. These patterns are relevant if maasv ever needs webhook delivery, but for v1 the in-process event bus avoids all of this complexity.
- [Hookdeck: Webhooks at Scale](https://hookdeck.com/blog/webhooks-at-scale)

## Recommended v1 Scope

### The Honest Assessment

Before proposing an implementation, the fundamental question: **should the cognition layer be proactive, or should the agent just poll?**

**Arguments for triggers in maasv:**
- The cognition layer knows *when* data changes. The agent doesn't, unless it polls constantly.
- Some events are time-sensitive (contradiction detected, escalation pattern spike). Polling every 30 seconds might miss the window.
- Clean separation of concerns: maasv owns data events, Doris owns actions.
- The events table is useful as an audit log regardless of whether triggers fire.

**Arguments for polling from Doris:**
- Doris already polls maasv on every conversation turn. Most triggers would fire in that window anyway.
- Triggers add complexity to maasv — more code paths, more failure modes, more testing surface.
- The "proactive" behaviors Doris needs (birthday reminders, follow-ups) could be implemented as scheduled jobs in Doris that query maasv, no triggers required.
- maasv is a *library* embedded in Doris's process. An event bus between them is just a function call with extra steps.
- Complexity budget: maasv is already at 0.2.0 with 10 migrations. Every feature needs to justify its weight.

**My recommendation:** A thin events layer in maasv (events table + in-process bus), but **most of the "proactive" logic lives in Doris.** maasv should emit events. Doris should decide what to do with them. The cognition layer's job is to say "something changed." The agent's job is to decide if that matters.

### v1 Implementation

**Layer 1: Events table (SQLite triggers)**
- Add an `events` table to the schema.
- Add SQLite triggers on `memories`, `entities`, and `relationships` tables for INSERT/UPDATE/DELETE.
- Events accumulate durably. Consumers poll or use the in-process bus for real-time.
- Event types: `memory.stored`, `memory.superseded`, `memory.deleted`, `entity.created`, `entity.merged`, `relationship.added`, `relationship.expired`, `relationship.updated`, `wisdom.logged`, `wisdom.outcome`, `wisdom.feedback`.
- Automatic cleanup: events older than 24 hours or marked consumed are pruned by hygiene.

**Layer 2: In-process event bus (Python)**
- Simple `emit(event)` / `on(event_type, callback)` in `maasv/core/events.py`.
- Emit from Python code (store.py, graph.py, wisdom.py) for events that can't be captured by SQL triggers (dedup decisions, normalization, business logic).
- Also emit after SQL triggers fire (so consumers get both SQL-level and Python-level events via the same bus).
- Callbacks are async-safe: queued to a background thread, not blocking the store path.
- Cascade prevention: events from within callbacks tagged `source=trigger`, don't re-trigger.

**Layer 3: Trigger registration (deferred to v2)**
- v1 has no "when X do Y" rules. It just has events.
- Doris registers callbacks at maasv init time for events she cares about.
- The trigger *matching* logic (conditions, filters, debouncing) lives in Doris, not maasv.
- This keeps maasv simple and puts the complexity where it belongs — in the agent that understands user context.

### What v1 Does NOT Include

- No webhook delivery infrastructure.
- No rule engine or condition matching in maasv.
- No trigger CRUD API (registration, listing, deletion of triggers).
- No debouncing in maasv (Doris handles this).
- No trigger persistence (callbacks are registered in-process, lost on restart — Doris re-registers at startup).

## First Use Case in Doris

**Scenario: New person mentioned in conversation**

1. Adam says: "I had lunch with Sarah from the TerryAnn team today."
2. Doris extracts entities via maasv's extraction endpoint.
3. maasv's `find_or_create_entity()` creates a new entity for "Sarah" (type: person).
4. SQLite trigger writes to events table: `entity.created`, source_id=`ent_abc123`, data=`{"name": "Sarah", "entity_type": "person"}`.
5. In-process event bus emits `entity.created` event.
6. Doris's callback (registered at startup) receives the event.
7. Doris checks: is this a person? Was it just created (not found)? Is it associated with a known project?
8. Doris proactively says: "I noticed Sarah from TerryAnn. Want me to remember her role, or anything else about her?"
9. Adam: "She's the new PM replacing Mike."
10. Doris stores the relationship and updates the entity metadata.

**What this required from maasv:** An events table, a 50-line event bus, and ~10 SQLite triggers. Total: maybe 200 lines of new code.

**What this required from Doris:** A callback registration at startup, filtering logic (is this person new? relevant?), and a proactive prompt template. This is where the intelligence lives.

## Dependencies in maasv Codebase

### New Files
- `maasv/core/events.py` — Event bus implementation (~100 lines).

### Modified Files
- `maasv/core/db.py` — New migration (11) to create `events` table and SQLite triggers on `memories`, `entities`, `relationships`, `wisdom`.
- `maasv/core/store.py` — Add `emit()` calls after store, supersede, delete operations. Also emit on dedup detection (Python-level event).
- `maasv/core/graph.py` — Add `emit()` calls after entity create, relationship add/expire/update, entity merge.
- `maasv/core/wisdom.py` — Add `emit()` calls after log, outcome, feedback.
- `maasv/lifecycle/memory_hygiene.py` — Add `emit()` calls after dedup, prune, consolidate operations.
- `maasv/__init__.py` — Expose event bus in public API (e.g., `maasv.on("event_type", callback)`).
- `maasv/config.py` — Add `events_enabled: bool = True` and `events_max_age_hours: int = 24` config fields.

### No Changes Needed
- `maasv/server/` — No changes in v1. HTTP trigger registration is v2.
- `maasv/lifecycle/worker.py` — SleepWorker doesn't need changes. It already queues jobs; it doesn't need to be event-driven.
- `maasv/lifecycle/inference.py`, `review.py`, `learn.py`, `reorganize.py` — These are consumers of events (in the future), not producers. v1 doesn't change them.

## Risks

### Trigger Storms
If entity extraction processes a long document and creates 50 entities in rapid succession, that's 50 `entity.created` events. Doris's callback fires 50 times. If each one generates a proactive prompt, Adam gets buried.

**Mitigation:** Debouncing lives in Doris (not maasv). Doris accumulates events for 5 seconds before acting. The events table provides durability; the bus provides immediacy; Doris provides intelligence.

### Performance Impact on store() Path
The `store_memory()` path is already bottlenecked by embedding computation. Adding an `emit()` call adds negligible overhead if callbacks are queued (not synchronous). The SQLite trigger adds one INSERT per memory store — tested to be sub-millisecond.

The bigger risk is if someone registers a slow callback that blocks the emit queue. Mitigation: callbacks run on a separate thread with a timeout. Slow callbacks are logged and skipped.

### Complexity Budget
maasv is at v0.2.0 with a clean, understandable codebase. Every feature adds maintenance burden. The events system is ~200 lines of new code, which is modest. But it adds a new concept (events) that every future contributor needs to understand.

**Mitigation:** Make it opt-in (`events_enabled: bool = True` in config, but easy to disable). Keep the events module self-contained — no tentacles reaching into unrelated code. The emit calls in store.py/graph.py/wisdom.py should be single lines, not complex conditional logic.

### The "Build It and They Won't Come" Risk
The events table might accumulate events that nobody ever reads. The bus might have no registered listeners. This is waste — SQLite triggers firing for nothing, events table growing for nothing.

**Mitigation:** Events are cheap (one row per event, auto-pruned after 24 hours). The SQLite triggers are tiny (one INSERT). If nobody's listening, the cost is near-zero. And the events table doubles as an audit log, which has value independent of triggers.

### Maasv vs. Doris Boundary Confusion
If maasv starts "deciding" what's important (trigger conditions, debouncing, priority), it's crossing into agent territory. The cognition layer should be dumb about intent and smart about data.

**Mitigation:** v1 design keeps maasv dumb. It emits events. It doesn't interpret them. All filtering, debouncing, and action-taking lives in Doris. maasv is the nervous system; Doris is the brain.

## Open Questions

1. **Should the events table use autoincrement INTEGER or UUID primary keys?** Autoincrement is simpler and guarantees ordering. UUID is consistent with the rest of maasv's ID scheme. Leaning toward autoincrement — ordering matters for event consumers.

2. **Should events from SQLite triggers include the full row data or just the ID?** Full data avoids a round-trip query but makes the trigger SQL more complex and the events table larger. Just-ID is simpler but requires a follow-up query. Leaning toward a middle ground: include key fields (category, subject, entity_type) but not full content.

3. **Should the event bus support wildcard subscriptions?** E.g., `on("memory.*", callback)` to catch all memory events. Useful but adds complexity. Leaning toward yes — it's 5 extra lines of code and significantly more ergonomic.

4. **Should event emission be synchronous or asynchronous by default?** Synchronous is simpler to reason about (event is emitted after the DB commit). Async is faster (doesn't block the store path). Leaning toward async with an option for sync in tests.

5. **How should Doris re-register callbacks after a restart?** If maasv is a library (not a server), callbacks die with the process. Doris needs to call `maasv.on(...)` at startup every time. The events table provides durability — on restart, Doris can process any events that accumulated while she was down by polling the events table.

6. **What about the MCP server?** Doris's MCP server (`doris-memory`) wraps maasv. Should event registration happen at the MCP layer too? Probably yes — Claude Desktop and Claude Code both use MCP, and they'd benefit from knowing when memory changes. But MCP doesn't support server-initiated messages (it's request-response). This might require MCP SSE (Server-Sent Events) or a polling endpoint. Deferred to v2.

7. **Is 24 hours the right max event age?** Events older than 24 hours are unlikely to be actionable. But some use cases (weekly digests) might want longer retention. Make it configurable, default to 24 hours.

8. **Should the events table be excluded from database backups?** It's transient data, not core memory. Including it in backups wastes space. But it's also useful for debugging. Leaning toward including it — the overhead is minimal.

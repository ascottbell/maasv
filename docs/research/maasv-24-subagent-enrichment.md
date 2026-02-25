# MAASV-24: Subagent Orchestration for Enrichment -- Research

## Problem Statement

Today, a memory in maasv knows only what the extraction LLM can infer from the conversation text that produced it. When Adam says "I had lunch at Russ & Daughters on Houston Street," the extraction pipeline creates:

- Entity: "Russ & Daughters" (type: place, confidence: 0.9)
- Relationship: Adam -> visited -> Russ & Daughters

That's it. The memory has no idea that Russ & Daughters is a Jewish appetizing shop founded in 1914 on East Houston Street in the Lower East Side, that it's 0.4 miles from the F train at 2nd Ave, or that Adam has visited three other restaurants on the same block. It doesn't know the cuisine type, the neighborhood, or the phone number.

This matters for retrieval in concrete ways:

1. **Queries about categories fail.** "Where did I eat Jewish food recently?" won't match because "Jewish food" isn't in the memory -- the LLM might infer it from the name, but it's not stored as structured metadata.
2. **Location-based queries fail.** "What's near the office?" requires knowing that Russ & Daughters is on the Lower East Side, and where the office is.
3. **Entity disambiguation fails.** If there's also a "Russ & Daughters Cafe" (the separate Brooklyn Navy Yard location), the system has no way to distinguish them.
4. **Cross-referencing fails.** "What restaurants has Adam visited in the same neighborhood?" requires neighborhood metadata that doesn't exist.

The enrichment gap is the difference between what was said and what would be useful to know.

## Current State in maasv

### Extraction Pipeline

The extraction pipeline lives in `/Users/macmini/Projects/maasv/maasv/extraction/entity_extraction.py`. Here's the flow:

1. **Input**: A conversation summary string + topic (line 259-261)
2. **LLM Call**: `EntityExtractor.extract_from_summary()` sends the summary to Claude Haiku 4.5 with a structured extraction prompt (line 274-280). The prompt asks for entities (person, place, project, organization, event, technology) and relationships with predefined predicates.
3. **Parsing**: Response is parsed as JSON. Entities and relationships are capped (20 entities, 30 relationships per extraction -- lines 295-300).
4. **Storage**: `store_extracted_entities()` (line 310) stores entities via `find_or_create_entity()` and relationships via `add_relationship()`, with sanitization and garbage filtering.

The convenience function `extract_and_store_entities()` (line 449) wraps both steps.

### Entity Schema

Entities in the graph (from `/Users/macmini/Projects/maasv/maasv/core/graph.py`) have:
- `name`, `entity_type`, `canonical_name` (line 256-257)
- `metadata` (JSON dict -- line 256)
- Types: person, place, project, organization, event, technology (from the extraction prompt)

The metadata dict is the only extensible field. Currently, extraction stores `description`, `source`, and `confidence` in metadata (line 352-355). There's no schema enforcement on metadata -- it's a free-form JSON dict.

### Memory Schema

Memories (from `/Users/macmini/Projects/maasv/maasv/core/store.py`) have:
- `content`, `category`, `subject`, `source`, `confidence` (line 24-33)
- `metadata` (JSON dict)
- `origin`, `origin_interface`

No enrichment fields exist. The `metadata` dict is capped at 10,000 chars of JSON (line 20).

### Background Processing

The sleep worker (`/Users/macmini/Projects/maasv/maasv/lifecycle/worker.py`) runs five job types during idle periods:
- **Inference**: Resolves vague references ("that place") to specific entities (line 136-139)
- **Review**: Second-pass conversation analysis for patterns/preferences (line 141-143)
- **Reorganize**: Graph optimization, path caching, orphan cleanup (line 145-147)
- **Memory Hygiene**: Dedup, prune stale, consolidate clusters (line 148-151)
- **Learn**: Trains the learned ranker on retrieval feedback (line 153-155)

All jobs are cancellable via `cancel_check()` callbacks and run on a single daemon thread via a priority queue (max 50 jobs). The idle monitor triggers jobs after 30 seconds of inactivity (configurable via `idle_threshold_seconds`).

### Where Enrichment Would Plug In

The natural insertion points are:

1. **Post-extraction hook in `extract_and_store_entities()`** (line 449-458): After entities are stored, trigger enrichment for newly created entities.
2. **New SleepJob type** in the worker: `JobType.ENRICH` alongside the existing five types.
3. **Entity metadata updates** via `update_memory_metadata()` in store.py or direct SQL updates to entity metadata.

The existing `find_or_create_entity()` returns the entity ID, which enrichment would need to attach metadata to.

## What Enrichment Actually Helps

### High-Value Enrichments

These are enrichments that directly improve retrieval quality for a personal assistant:

**1. Entity Type Refinement / Category Tagging**
- Enriching "Russ & Daughters" with `{cuisine: "Jewish/deli", category: "restaurant"}` directly enables queries like "Where did I eat Jewish food?"
- This is the single highest-value enrichment because it creates new retrieval dimensions that don't exist in the raw text.
- Can be done cheaply with an LLM call using the entity name + any available description.
- Cost: ~$0.001-0.002 per entity (Haiku 4.5, ~500 input + 200 output tokens).

**2. Entity Descriptions / Summaries**
- A one-sentence description: "Russ & Daughters is a Jewish appetizing shop on the Lower East Side, known for smoked fish and bagels, established 1914."
- Improves vector search because the description gets embedded alongside the entity name.
- The extraction LLM already generates a `description` field, but it's based solely on conversation context. A web-informed description would be richer.
- Cost: One LLM call or one web lookup per entity.

**3. Neighborhood / Location Normalization**
- Normalizing "Houston Street" to "Lower East Side, Manhattan" enables location-based grouping.
- For a personal assistant centered on NYC + Hudson Valley, neighborhood tagging is extremely practical.
- Can use a simple lookup table for known neighborhoods rather than a full geocoding API.
- Cost: Negligible with a local lookup table. ~$0 if using Nominatim (free, 1 req/sec rate limit).

**4. Relationship Inference Between Existing Entities**
- After storing "Adam visited Russ & Daughters" and "Adam visited Katz's Deli," an enrichment pass could infer "both are in the Lower East Side" and "both serve Jewish food."
- This creates graph edges that improve graph-signal retrieval.
- Cost: LLM call over existing graph data, no external API needed.

### Medium-Value Enrichments

**5. Temporal Context**
- Knowing that an event happened "last Tuesday" and resolving that to "2026-02-17" makes time-based queries work.
- The extraction pipeline doesn't currently extract dates. Adding temporal resolution would help queries like "What did I do last week?"
- Cost: Can be done in the extraction prompt itself, near-zero marginal cost.

**6. Contact Information Normalization**
- Phone numbers, emails, addresses -- normalizing these into standard formats.
- Already partially supported via `has_email`, `has_phone` predicates.
- Cost: Regex-based, near-zero.

### Low-Value Enrichments

**7. Wikipedia/Wikidata Entity Linking**
- Linking "Russ & Daughters" to its Wikidata ID (Q7382847) sounds impressive but provides minimal retrieval benefit for a personal assistant.
- Wikidata IDs don't help with natural language queries. The structured data in Wikidata (founding date, coordinates) is useful, but the linking step itself is expensive and error-prone.
- Risk: High. Entity linking to knowledge bases has significant ambiguity -- "Python" could be the language, the snake, or the Monty Python troupe. For a personal assistant, the context is usually clear from the conversation, making external linking redundant.

**8. Full Company Enrichment (Clearbit-style)**
- Revenue, employee count, industry classification, social media links for every company mentioned.
- Overkill for personal assistant use. Adam doesn't need to know Anthropic's revenue when he mentions them in a conversation.
- Exception: If Adam is doing business due diligence (Gabby's M&A work context), this becomes high-value. But that's a specialized use case, not a default enrichment.
- Cost: $0.36+ per lookup via Clearbit (starting at $99/mo for 275 requests). Expensive for the marginal value.

**9. Full Geocoding (Lat/Lon Coordinates)**
- Converting addresses to lat/lon coordinates for distance calculations.
- Useful in theory, but maasv doesn't have a spatial index, and most personal assistant queries are neighborhood-level, not coordinate-level.
- Cost: Free via Nominatim, but adds latency and external dependency for minimal gain.

**10. Image/Media Enrichment**
- Fetching photos, logos, or maps for entities.
- No retrieval benefit. Might improve UI presentation, but maasv is a retrieval layer, not a UI.

### Cost/Benefit Analysis

Assuming ~50 new entities per day (based on Doris's typical usage of 10-20 conversations/day with 2-5 entities each):

| Enrichment | Cost/Entity | Daily Cost (50 ent) | Latency | Retrieval Impact |
|---|---|---|---|---|
| Category tagging (LLM) | $0.002 | $0.10 | ~0.5s | High |
| Description enrichment (LLM) | $0.002 | $0.10 | ~0.5s | Medium-High |
| Neighborhood normalization (local) | $0 | $0 | <1ms | High for location queries |
| Relationship inference (LLM) | $0.003 | $0.15 | ~1s | Medium |
| Temporal resolution (in extraction) | $0 | $0 | 0ms | Medium |
| Wikipedia linking | $0.002 | $0.10 | ~2s | Low |
| Company enrichment (Clearbit) | $0.36 | $18.00 | ~1s | Low (for personal use) |
| Full geocoding (Nominatim) | $0 | $0 | ~1s | Low (no spatial index) |

**Bottom line**: Category tagging + description enrichment + neighborhood normalization gives 80% of the retrieval improvement for <$0.20/day. Everything else is marginal or negative ROI.

## Architecture Options

### Option A: Synchronous Enrichment (Enrich During store())

```
store_memory() -> extract_entities() -> enrich_entities() -> return
```

**Pros:**
- Simple. No async complexity, no queue management.
- Enriched data is available immediately for the next query.
- Easy to test -- single codepath.

**Cons:**
- Adds 0.5-2s latency per memory store operation. For Doris, this means the user waits longer after each conversation turn.
- If enrichment fails, it blocks or complicates the store operation.
- Serial processing -- enrichment of entity A can't overlap with enrichment of entity B.

**Verdict:** Acceptable for cheap enrichments (category tagging, neighborhood lookup). Unacceptable for anything requiring web calls or multiple LLM calls.

### Option B: Async Background Enrichment (Store Fast, Enrich Later)

```
store_memory() -> extract_entities() -> queue_enrichment_job() -> return
                                                |
                                                v (background)
                                        enrich_entities() -> update_entity_metadata()
```

**Pros:**
- Store operation remains fast.
- Enrichment can be batched -- enrich 10 entities at once instead of one at a time.
- Natural fit with the existing SleepWorker pattern.
- Enrichment failures don't block memory storage.

**Cons:**
- Complexity: need to track enrichment state (pending, complete, failed).
- Stale window: queries between store and enrichment completion won't benefit from enriched data.
- Need to handle the case where enrichment runs but the entity was already deleted/merged by hygiene.

**Verdict:** Best fit for maasv's architecture. The SleepWorker already handles cancellable background jobs. The stale window is acceptable -- enrichment is a latent quality improvement, not a real-time requirement.

### Option C: On-Demand Enrichment (Enrich at Retrieval Time)

```
query() -> retrieve_candidates() -> enrich_if_needed() -> rank() -> return
```

**Pros:**
- Only enriches entities that are actually queried -- zero waste.
- No background infrastructure needed.

**Cons:**
- Adds latency to every query, not just storage.
- Retrieval is the hot path -- adding LLM calls here is unacceptable for Doris's voice interface (needs <2s response time).
- Enrichment results aren't cached for future queries unless you add a caching layer, which is just deferred background enrichment with extra steps.

**Verdict:** Bad idea. Retrieval latency is sacred in maasv. Don't touch it.

### Option D: Hybrid (Cheap Sync + Expensive Async)

```
store_memory() -> extract_entities() -> cheap_enrichment() -> queue_expensive_enrichment() -> return
                                              |                          |
                                    (neighborhood lookup,         (background)
                                     contact normalization)     LLM category tagging,
                                                                description enrichment
```

**Pros:**
- Cheap enrichments (lookup tables, regex) are available immediately.
- Expensive enrichments (LLM calls) don't block storage.
- Progressive enhancement -- data quality improves over time.

**Cons:**
- Two codepaths to maintain and test.
- Need to define "cheap" vs "expensive" boundary.

**Verdict:** This is the right answer. Neighborhood lookup and contact normalization are effectively free and should be synchronous. LLM-based enrichments should be async.

### Recommendation: Option D (Hybrid)

Phase 1 focuses on cheap sync enrichments. Phase 2 adds async LLM enrichments via the SleepWorker.

## Enrichment Sources

### Free / Local (Sync-Safe)

| Source | What It Provides | Latency | Cost |
|---|---|---|---|
| Local neighborhood lookup table | NYC neighborhood from address/place name | <1ms | $0 |
| Regex normalization | Phone/email/date standardization | <1ms | $0 |
| Existing graph context | Related entities already in the graph | <10ms | $0 |

### LLM-Based (Async)

| Source | What It Provides | Latency | Cost |
|---|---|---|---|
| Claude Haiku 4.5 | Category tagging, description enrichment, relationship inference | 0.5-1s | ~$0.002/call |
| Batch Haiku (if batch API used) | Same, but cheaper | 5-30min | ~$0.001/call |

### External APIs (Async, Optional)

| Source | What It Provides | Latency | Cost | Rate Limit |
|---|---|---|---|---|
| Nominatim (OSM) | Geocoding, reverse geocoding | 0.5-2s | Free | 1 req/sec |
| Wikipedia API | Entity descriptions | 0.3-1s | Free | Rate limited |
| Wikidata API | Structured entity data | 0.3-1s | Free | Rate limited |
| OpenCage | Geocoding + enriched data | 0.3-1s | Free tier: 2,500/day | 1 req/sec |

For a personal assistant doing ~50 entities/day, all free tiers are sufficient.

## Prior Art

### Mem0

Mem0's architecture (published April 2025 in [arxiv.org/abs/2504.19413](https://arxiv.org/abs/2504.19413)) stores memories as a directed labeled graph where nodes are entities and edges are relationships. Key enrichment-related features:

- **Entity Extractor + Relations Generator**: Two-step pipeline similar to maasv's extraction. Mem0 uses GPT-4o-mini with function calling for structured extraction.
- **Conflict Detector + Update Resolver**: When new information conflicts with existing graph data, an LLM decides whether to add, merge, invalidate, or skip. This is a form of enrichment -- existing entities get updated context from new conversations.
- **Graph Memory retrieval boost**: Mem0g (graph-enhanced) achieves ~2% higher scores than base Mem0 on their benchmarks. Modest but measurable.
- **No external enrichment**: Mem0 does not enrich entities from external sources. All entity data comes from conversation extraction.

**What maasv can learn from Mem0**: The conflict detection/resolution pattern is valuable. When enrichment data conflicts with existing metadata (e.g., enrichment says "Russ & Daughters is in Brooklyn" but existing data says "Houston Street"), the system needs a resolution strategy. Mem0's LLM-powered resolver is one approach.

Source: [Mem0 Research](https://mem0.ai/research), [Mem0 Graph Memory Blog](https://mem0.ai/blog/graph-memory-solutions-ai-agents)

### Zep / Graphiti

Zep's Graphiti engine (published January 2025 in [arxiv.org/abs/2501.13956](https://arxiv.org/abs/2501.13956)) is the closest prior art to what MAASV-24 proposes:

- **Three-tier graph**: Episode subgraph (raw messages), semantic entity subgraph (extracted entities + relationships), community subgraph (clusters of related entities). The community tier is effectively enrichment -- it groups entities into higher-level concepts.
- **Temporal enrichment**: Every edge has validity intervals (`t_valid`, `t_invalid`). This is temporal enrichment baked into the data model, not added after the fact.
- **Entity extraction with context window**: Processes current message + last N messages together. This provides better extraction than single-message processing.
- **Hybrid search**: Cosine similarity + BM25 + graph BFS, similar to maasv's 3-signal retrieval.
- **Custom entity ontologies**: Developers define entity types using Pydantic models. This enables domain-specific enrichment schemas.
- **Performance**: P95 retrieval latency of 300ms. 18.5% accuracy improvement over baselines.

**What maasv can learn from Zep**: The community subgraph concept -- automatically clustering entities into neighborhoods, categories, or themes -- is a form of enrichment that directly improves retrieval. maasv's reorganize job already caches common paths; extending this to entity clustering would be a natural fit.

Source: [Zep Paper](https://arxiv.org/html/2501.13956v1), [Graphiti GitHub](https://github.com/getzep/graphiti)

### LangMem

LangMem (LangChain's memory framework, launched 2025) has the most directly relevant enrichment architecture:

- **ReflectionExecutor**: Background worker that processes memories after conversation activity settles. Implements debouncing -- if 5 messages arrive in 10 seconds, it waits for a pause before processing them all at once. This is almost exactly what maasv's SleepWorker does with idle detection.
- **Memory enrichment process**: Searches for relevant existing memories, extracts new information, updates existing memories, maintains versioned history. This is consolidation + enrichment in one pass.
- **Background Memory Manager**: Dedicated worker that handles extraction, consolidation, and updates asynchronously.
- **Three memory types**: Semantic (facts), episodic (events), procedural (instructions). Enrichment can target any type.

**What maasv can learn from LangMem**: The debouncing pattern is smart. Rather than enriching each entity as it's created, batch entities from the same conversation and enrich them together. This reduces LLM calls (one call can categorize 10 entities) and provides better context (knowing all entities from a conversation helps disambiguate each one).

Source: [LangMem Conceptual Guide](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/), [LangMem Background Processing](https://langchain-ai.github.io/langmem/guides/delayed_processing/), [LangMem GitHub](https://github.com/langchain-ai/langmem)

### Multi-Agent Frameworks (CrewAI, AutoGen, LangGraph)

These frameworks provide orchestration patterns for specialized sub-agents, but their patterns are more about task decomposition than memory enrichment:

- **CrewAI**: Role-based agents (researcher, writer, reviewer) working in sequence or parallel. Relevant pattern: a "researcher" agent that enriches data before a "writer" agent uses it.
- **AutoGen**: Conversational multi-agent collaboration. Agents chat with each other to refine outputs. Overkill for enrichment -- enrichment is a simple input->output pipeline, not a conversation.
- **LangGraph**: Graph-based workflows with decision points and parallel processing. Most relevant for orchestrating multiple enrichment steps with conditional logic (e.g., only geocode if entity type is "place").

**What maasv should NOT do**: Don't build a full multi-agent framework for enrichment. The overhead of agent coordination, message passing, and conversation management is not justified for what is essentially a set of independent API calls. A simple enrichment pipeline with conditional steps is sufficient.

Source: [DataCamp Framework Comparison](https://www.datacamp.com/tutorial/crewai-vs-langgraph-vs-autogen), [Agent Orchestration Guide 2026](https://iterathon.tech/blog/ai-agent-orchestration-frameworks-2026)

## Risks and Failure Modes

### Enrichment Making Things Worse

This is the most important risk and the one most likely to be hand-waved away. Enrichment can actively degrade retrieval quality:

**1. Hallucinated Metadata**
- An LLM asked to categorize "The Smith" (a restaurant) might tag it as `{category: "person"}` because "Smith" is a surname.
- Hallucinated categories poison vector search -- the embedding of "restaurant" + hallucinated "person" metadata is worse than the embedding of just "restaurant."
- **Mitigation**: Confidence thresholds. Only store enrichment metadata above a threshold (e.g., 0.8). Include a `source: "enrichment"` field so it can be filtered or removed later.

**2. Stale Lookups**
- Enriching "Russ & Daughters" with hours and phone number from a web lookup. Six months later, the hours change. The memory system now has stale data that it presents as fact.
- **Mitigation**: Temporal validity. Enrichment metadata should have a `valid_until` timestamp. After expiry, the data is either re-enriched or flagged as potentially stale.

**3. Entity Confusion**
- "Apple" the company vs. "apple" the fruit. Web enrichment for "Apple" will return company data, but the conversation might have been about actual apples.
- **Mitigation**: Use conversation context when enriching, not just entity name. The extraction already provides a `description` field -- pass that to the enrichment LLM.

**4. Enrichment Pollution**
- Adding too much metadata makes vector embeddings noisier. If every entity has 20 metadata fields, the embedding captures all of them, diluting the signal from the fields that actually matter.
- **Mitigation**: Be selective. Store enrichment in metadata fields that are NOT embedded. Only embed the entity name + description, not the full metadata blob.

**5. Memory Poisoning**
- If enrichment stores hallucinated data, and that data gets used in future LLM prompts (e.g., entity profiles in retrieval context), it creates a feedback loop where the system reinforces its own mistakes.
- This is the "cross-session memory corruption" risk documented in the AI hallucination literature.
- **Mitigation**: Tag all enrichment-derived data with `source: "enrichment"` and `confidence`. Never present enrichment data as user-stated fact. In retrieval prompts, distinguish between "Adam said" and "system inferred."

### Failure Modes in Implementation

**6. Enrichment Queue Overflow**
- If enrichment is slower than entity creation, the queue grows unboundedly.
- The SleepWorker already has a max queue size of 50 (line 40 of worker.py). Enrichment jobs would share this pool.
- **Mitigation**: Enrichment jobs should be lower priority than inference/review. If the queue is full, drop enrichment jobs silently.

**7. Enrichment During Hygiene Race**
- Enrichment runs on entity X. Meanwhile, hygiene merges entity X into entity Y. The enrichment result arrives and tries to update a deleted entity.
- **Mitigation**: Check entity exists before writing enrichment results. Use `find_entity_by_name()` to re-resolve the entity ID.

**8. Cost Runaway**
- A conversation about 50 different companies triggers 50 enrichment LLM calls.
- **Mitigation**: Daily enrichment budget. Track enrichment spend per day and stop after a threshold (e.g., $1/day). At ~$0.002/call, that's 500 enrichments/day -- more than enough.

## Recommended v1 Scope

**Goal**: Get measurable retrieval improvement with minimal complexity.

### Phase 1: Category Tagging (sync, LLM-based)

Add a post-extraction step that sends newly created entities to a cheap LLM call for category tagging.

**Enrichment prompt** (single call for a batch of entities):
```
Given these entities extracted from a conversation, add category tags that would help with retrieval.

Entities:
- Russ & Daughters (type: place, description: "Restaurant Adam visited")
- Python (type: technology, description: "Programming language used in project")

For each entity, provide:
- categories: list of 1-3 relevant categories (e.g., "restaurant", "jewish food", "lower east side")
- short_description: one sentence if the current description is vague

Return JSON array.
```

**Implementation**:
1. After `store_extracted_entities()` returns, collect newly created entity IDs.
2. Batch them (up to 10 per call).
3. Single Haiku call to generate categories.
4. Store categories in entity metadata via `update_entity_metadata()` or direct SQL.
5. Re-embed entity with categories appended to name+description for improved vector search.

**Cost**: ~$0.002-0.005 per batch of 10 entities. Negligible.

**New config fields**:
```python
# In MaasvConfig
enrichment_enabled: bool = True
enrichment_model: str = "claude-haiku-4-5-20251001"
enrichment_llm: object = None  # Per-operation LLM override
enrichment_max_batch: int = 10
enrichment_daily_budget_cents: int = 100  # $1/day cap
```

### Phase 2: Async Enrichment via SleepWorker

Add `JobType.ENRICH` to the SleepWorker. Queue enrichment jobs after extraction.

**New enrichment types to add**:
1. **Neighborhood normalization** for place entities (local lookup table, sync-safe)
2. **Description enrichment** for entities with vague descriptions (LLM call)
3. **Cross-entity relationship inference** (LLM call over entity batch + existing graph)

**Implementation**:
1. Add `JobType.ENRICH` to `worker.py`.
2. Create `/Users/macmini/Projects/maasv/maasv/lifecycle/enrich.py` with `run_enrich_job()`.
3. The job processes a batch of entity IDs, running each enrichment type in sequence.
4. Track enrichment state in entity metadata: `{"enrichment_status": "pending|complete|failed", "enriched_at": "...", "enrichment_version": 1}`.

### What NOT to Build in v1

- No external API calls (Nominatim, Wikipedia, Wikidata). Add these in v2 if v1 shows retrieval improvement.
- No full subagent framework. Use simple function calls, not agent orchestration.
- No real-time enrichment at retrieval time.
- No image/media enrichment.
- No company enrichment (Clearbit-style). Not relevant for personal assistant use.

## First Use Case in Doris

**Scenario**: Adam asks Doris "What Jewish restaurants have I been to recently?"

**Today (without enrichment)**:
1. maasv searches for "Jewish restaurants" -- vector search finds nothing because "Jewish" isn't in any memory or entity name.
2. BM25 also finds nothing -- no keyword match.
3. Graph search finds Adam's `visited` relationships to restaurants, but there's no "Jewish food" categorization.
4. Doris says "I don't have any memories about Jewish restaurants."

**With v1 enrichment**:
1. When "Russ & Daughters" was stored, the enrichment step added `{categories: ["jewish food", "deli", "appetizing shop", "lower east side"]}` to entity metadata.
2. The entity description was enriched to "Russ & Daughters, a Jewish appetizing shop on East Houston Street, Lower East Side."
3. Vector search for "Jewish restaurants" now matches the enriched description.
4. Doris says "You visited Russ & Daughters recently -- it's a Jewish appetizing shop on the Lower East Side."

**Measurable improvement**: This query goes from 0 results to 1+ correct results. That's the kind of concrete, testable improvement that justifies enrichment.

## Dependencies in maasv Codebase

### Files That Need Changes

| File | Change |
|---|---|
| `maasv/config.py` | Add enrichment config fields (enabled, model, budget, batch size) |
| `maasv/extraction/entity_extraction.py` | Add post-extraction enrichment hook in `extract_and_store_entities()` |
| `maasv/lifecycle/worker.py` | Add `JobType.ENRICH` |
| `maasv/__init__.py` | Add `get_llm_for("enrichment")` support |

### New Files

| File | Purpose |
|---|---|
| `maasv/lifecycle/enrich.py` | Enrichment job runner + enrichment logic |
| `maasv/enrichment/` (optional) | If enrichment grows complex enough to warrant its own package |

### Existing Code That Supports Enrichment

- **`find_or_create_entity()`** (graph.py line 352): Returns entity ID needed for metadata updates.
- **`update_memory_metadata()`** (store.py line 305): Updates metadata dict for memories.
- **Entity metadata** (graph.py line 256): Free-form JSON dict -- can store enrichment data without schema migration.
- **SleepWorker** (worker.py): Ready to accept new job types with minimal changes.
- **`get_llm_for()`** (__init__.py line 153): Already supports per-operation LLM overrides. Adding "enrichment" is trivial.

### No Schema Migration Required

Enrichment data goes into the existing `metadata` JSON column on both `entities` and `memories` tables. No new columns, no new tables, no migration.

## Open Questions

1. **Should enrichment re-embed entities?** If we add categories to entity metadata, should we regenerate the entity's embedding to include the new categories? This improves vector search but adds cost (one embedding call per enriched entity). The answer is probably yes for v1, since the whole point is improving retrieval.

2. **Enrichment versioning?** If we re-enrich an entity (e.g., after a new conversation provides more context), should we version the enrichment metadata? Or just overwrite? Versioning adds complexity but prevents data loss.

3. **Entity type vs. category tags?** Currently entities have a single `entity_type` (person, place, etc.). Should category tags be a separate concept, or should we support multiple types per entity? Separate is cleaner -- `entity_type` is structural, categories are semantic.

4. **Budget tracking granularity?** Track enrichment spend per day, per entity, or per conversation? Per-day is simplest and sufficient for v1.

5. **When to re-enrich?** If an entity's description changes (e.g., through a new conversation), should it be re-enriched? Probably yes, but only if the description changed materially.

6. **Enrichment in the embedding?** Should enrichment metadata (categories, enriched description) be included in the entity's vector embedding, or only stored as metadata for post-retrieval filtering? Including it in the embedding helps vector search. Excluding it keeps embeddings pure. For v1, include it -- the whole point is improving retrieval.

7. **How does this interact with predicate normalization?** The recently-added predicate normalization (MAASV-related, graph.py lines 93-203) uses embedding similarity to map unknown predicates. Enrichment-generated relationships would go through the same validation. Should enrichment use a richer predicate set?

8. **Batching strategy?** LangMem's debouncing pattern (wait for conversation to settle, then process all entities at once) is appealing. Should enrichment wait for the conversation to end, or enrich as entities are created? Waiting is better -- it reduces LLM calls and provides better context for disambiguation.

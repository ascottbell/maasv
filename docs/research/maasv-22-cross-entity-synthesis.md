# MAASV-22: Cross-Entity Synthesis — Research

## Problem Statement

maasv's knowledge graph stores explicit relationships — Adam `lives_in` Upper West Side, Adam `has_property_in` Hudson Valley, Sarah `lives_in` Beacon — but cannot connect dots across entities. Today's graph traversal is limited to 1-hop expansion from a query-matched entity, and only in service of memory retrieval (finding memories that mention related entities). There is no inference layer.

### Questions maasv cannot answer today

These all require synthesizing information across multiple entities or inferring implicit relationships:

1. **"Who do I know that lives near my Hudson Valley house?"** — Requires: Adam → `has_property_in` → Hudson Valley, Person X → `lives_in` → Beacon/Cold Spring/etc., geographic proximity inference between Hudson Valley and nearby towns.

2. **"What technologies do my active projects have in common?"** — Requires: traversing Doris → `built_with` → Python, TerryAnn → `built_with` → Next.js, maasv → `built_with` → Python, then aggregating and finding commonalities.

3. **"Who at Company X might know about Y?"** — Requires: Person → `works_at` → Company X, Person → `interested_in` / `works_on` → Y-related topics, multi-hop inference.

4. **"What decisions led to the current Doris architecture?"** — The `get_causal_chain` function exists but is limited to explicit causal predicates. Many architectural decisions are implicit across conversation memories, not edges.

5. **"What have I been spending time on this month?"** — Requires temporal aggregation across entities with recent relationship activity.

6. **"Gabby and I should find a restaurant near the Hudson Valley house — what do we both like?"** — Requires: Adam preferences, Gabby preferences, geographic filtering, intersection.

The common thread: the data exists in the graph, but the system cannot traverse, aggregate, filter, or infer across entities to answer compound questions.

---

## Current State in maasv

### Entity Model
- **File:** `/Users/macmini/Projects/maasv/maasv/core/graph.py`
- Entities have: `id`, `name`, `entity_type`, `canonical_name`, `metadata`, `access_count`, `last_accessed_at`
- Entity types: person, place, project, organization, event, technology (defined in extraction prompt, `entity_extraction.py:186-192`)
- FTS5 search with tiered fallback: word-match → trigram → LIKE (`graph.py:494-603`)
- Entity dedup via normalized name matching (`graph.py:324-385`)

### Relationship Model
- **File:** `/Users/macmini/Projects/maasv/maasv/core/graph.py:628-748`
- Relationships are temporal: `valid_from`, `valid_to` (bitemporal with `ingested_at`)
- Two forms: entity-to-entity (`object_id`) or entity-to-value (`object_value`)
- ~88 valid predicates in the allowlist (`graph.py:28-88`) with normalization fallback via embedding similarity
- Confidence scores on all relationships, clamped to [0.0, 1.0]
- Dedup on active relationships via partial unique indexes (migration 4)

### Graph Queries Available Today
1. **`get_entity_relationships(entity_id)`** — All relationships for one entity, both directions (`graph.py:765-826`)
2. **`graph_query(subject_type, predicate, object_type)`** — Pattern matching with type/predicate filters (`graph.py:962-1006`)
3. **`get_causal_chain(entity_id, direction, max_hops)`** — BFS traversal of causal predicates only (`graph.py:835-909`)
4. **`get_entity_profile(entity_id)`** — All current relationships grouped by predicate (`graph.py:1009-1042`)
5. **`search_entities(query)`** — FTS search for entities by name (`graph.py:494-603`)

### Graph Signal in Retrieval
- **File:** `/Users/macmini/Projects/maasv/maasv/core/retrieval.py`
- **`_expand_query_from_graph(db, query)`** (line 176): Finds entities matching query terms via FTS, does 1-hop expansion, appends related entity names to BM25 query as OR terms.
- **`_find_memories_by_graph(db, query)`** (line 354): The dedicated graph signal. Finds entities via FTS → 1-hop expansion → searches memory content for entity names. Uses cardinality filtering (skips hub entities with >50 relationships).
- **Graph slot injection** (line 942): Optional feature to force a graph-sourced result into the last retrieval slot if graph results didn't make it through RRF.
- **Key limitation:** Everything is 1-hop. There is no multi-hop traversal, no path finding, no community detection, no aggregation across entity neighborhoods.

### Sleep-Time Inference
- **File:** `/Users/macmini/Projects/maasv/maasv/lifecycle/inference.py`
- Current inference is narrow: resolves vague references ("that place", "the restaurant") to specific entities using LLM
- Stores results as `has_reference` relationships
- Does NOT infer implicit relationships between entities
- Does NOT do any cross-entity synthesis

### What's Missing
- No multi-hop traversal beyond causal chains
- No path-finding between arbitrary entities
- No community detection / clustering
- No relationship inference (if A→B and B→C, infer A→C)
- No temporal pattern detection
- No aggregation queries (what entities share property X?)
- No geographic or semantic proximity between entity values
- The LLM is never asked to reason over graph structure

---

## Inference Approaches

### Embedding-Based Link Prediction (TransE, RotatE, ComplEx, DistMult)

**How they work:** These methods embed entities and relations into continuous vector spaces and score candidate triples (h, r, t) by geometric operations. TransE models relationships as translations: h + r ≈ t. RotatE uses complex-valued rotation. ComplEx uses complex-valued tensor decomposition. DistMult uses bilinear diagonal scoring.

**Training:** All require training on existing triples, typically with negative sampling. They optimize to rank true triples higher than corrupted ones, evaluated by MRR, Hits@k, Mean Rank.

**Suitability for maasv's scale:**

This is where honesty matters. These methods were designed for and evaluated on large benchmark datasets:

| Dataset | Entities | Relations | Triples |
|---------|----------|-----------|---------|
| FB15k-237 | 14,541 | 237 | 310,116 |
| WN18RR | 40,943 | 11 | 93,003 |
| YAGO3-10 | 123,182 | 37 | 1,089,040 |

maasv has hundreds of entities, ~88 predicates, and likely low thousands of triples. At this scale:

- **Overfitting is near-certain.** With a few hundred entities, embedding dimensions (typically 100-500) can trivially memorize the entire graph. There isn't enough data for the geometric constraints to learn meaningful patterns.
- **Cold start problem.** New entities have no embedding history. In a personal KG where new entities appear regularly (Adam meets someone, visits a new place), the model can't reason about them until retrained.
- **Training cost vs. value.** Even lightweight training requires meaningful compute. For a graph this small, an LLM reading the entire adjacency list in its context window is more practical than training embeddings.
- **No off-the-shelf gains.** Pre-trained KGE models trained on Wikidata/Freebase would need fine-tuning, and the domain mismatch (encyclopedic knowledge vs. personal knowledge) makes transfer learning unlikely to help.

**Verdict:** Not practical for maasv. The graph is too small for statistical embedding methods to learn meaningful patterns. The compute-to-value ratio is poor.

### Rule-Based Inference (AMIE, AnyBURL, RDFRules)

**How they work:** These systems mine logical rules from observed triples. AnyBURL samples random paths and generalizes them into Horn rules. AMIE does top-down search over rule space. Example: if many entities have both `works_at(X, Y)` and `located_in(X, Z)` where Y is also `located_in(Z)`, the system learns: `works_at(X, Y) ∧ located_in(Y, Z) → located_in(X, Z)`.

**Suitability for maasv:**

- **Minimum data requirements:** Rule mining needs statistical support — enough instances of a pattern to distinguish signal from noise. With hundreds of entities and sparse relationships, most rules will have support of 1-3 instances, indistinguishable from coincidence.
- **Path sampling issue:** AnyBURL works by sampling paths. With a small, sparse graph, the path space is so limited that sampling converges quickly to the full graph — at which point you might as well just enumerate paths directly.
- **Where it could work:** If maasv grew to thousands of entities with dense relationships (e.g., a corporate knowledge graph), rule mining becomes more viable. For a personal KG, the patterns are too idiosyncratic.
- **Practical alternative:** Instead of learning rules from data, you can hand-code domain rules relevant to personal knowledge. "If person X lives_in location near location Y where Adam has_property_in, then X is a Hudson Valley neighbor." This is what makes sense at maasv's scale.

**Verdict:** Formal rule mining is overkill for hundreds of entities. Hand-coded domain rules or LLM-based rule application are more practical.

### LLM-Based Reasoning

**How it works:** Feed the LLM relevant portions of the graph (entity profiles, neighborhood subgraphs, relationship lists) and ask it to reason about connections, infer implicit relationships, or answer multi-hop questions.

**This is the sweet spot for maasv.** Here's why:

1. **The graph fits in context.** With hundreds of entities and low thousands of relationships, the full graph (or relevant subgraphs) fits within a single LLM context window. A Claude Haiku 4.5 call with 10K tokens of graph context is ~$0.01.

2. **No training required.** The LLM already understands that "Beacon, NY" is near "Hudson Valley" and that "lives_in" implies "commutes through." This world knowledge is exactly what embedding-based methods lack at small scale.

3. **Flexible inference.** The same mechanism handles geographic proximity, professional connections, temporal reasoning, and preference matching — all without separate models.

4. **Already in the stack.** maasv already uses Claude Haiku for entity extraction and sleep-time inference. Adding graph reasoning is an extension of existing patterns.

**Approaches within LLM-based reasoning:**

- **Subgraph-to-prompt:** Extract a relevant subgraph, serialize it as structured text, ask the LLM to reason. This is essentially what GraphRAG's local search does.
- **Iterative traversal:** LLM decides which edges to follow, like an agent navigating the graph. More expensive (multiple LLM calls) but handles deep reasoning.
- **Batch inference during sleep-time:** Periodically scan the graph for inference opportunities (new entities near existing clusters, missing symmetric relationships, etc.) and pre-compute inferred edges.

**Risks:**
- **Hallucination.** The LLM might infer relationships that don't exist. Mitigation: all inferred relationships get lower confidence scores and an `source="llm_inference"` tag.
- **Cost at scale.** Each synthesis query requires an LLM call. At maasv's current usage patterns (personal assistant, not SaaS), this is manageable.

**Verdict:** Most practical approach for maasv. LLM world knowledge compensates for the graph's small size. No training data needed. Fits existing architecture.

### Path-Based Reasoning

**How it works:** Find paths between entities using graph algorithms (BFS, DFS, shortest path, all simple paths), then use path structure to answer questions or infer relationships.

**Relevant algorithms for maasv:**

1. **Shortest path / all paths between two entities.** Already partially implemented in `get_causal_chain` but limited to causal predicates. Generalizing to all predicates would enable "how is entity A connected to entity B?" queries.

2. **Personalized PageRank (PPR).** Used by HippoRAG to great effect. Given a seed set of entities (extracted from query), PPR computes a relevance score for every other entity in the graph. Entities reachable through many short paths from the seed get higher scores. This naturally handles multi-hop reasoning: entities 2-3 hops away still get scores proportional to their connectivity.

3. **Community detection (Louvain, Label Propagation).** Groups densely connected entities into clusters. Useful for: "What are the major themes in my knowledge graph?" and for pre-computing community summaries (a la GraphRAG).

4. **Subgraph extraction.** Given a query, extract the relevant neighborhood (k-hop from seed entities) and pass to LLM for reasoning. This is the bridge between path-based and LLM-based approaches.

**Suitability for maasv:**

- All these algorithms are simple to implement on top of SQLite adjacency lists. No external dependencies needed.
- PPR is particularly well-suited: it handles the "spreading activation" pattern where relevance flows through the graph from query entities.
- At hundreds of entities, even naive implementations (Python BFS, O(V+E)) are sub-millisecond.

**Verdict:** Highly practical. Path-based algorithms provide the traversal foundation, LLM provides the reasoning. This is the combination that works.

### Hybrid Approaches (GraphRAG, HippoRAG, LightRAG)

These systems combine graph structure with LLM reasoning in different ways. They represent the state of the art for graph-augmented AI and are the most relevant models for maasv.

**Covered in detail in the Prior Art section below.**

---

## Prior Art

### Microsoft GraphRAG
- **Paper:** [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130) (Edge et al., April 2024)
- **Code:** [github.com/microsoft/graphrag](https://microsoft.github.io/graphrag/)

**Architecture:** Extract entity knowledge graph from source documents → community detection (Leiden algorithm) → LLM-generated community summaries at multiple hierarchy levels → two search modes (local: entity neighborhood traversal, global: community summary aggregation).

**Local search:** Identifies entities semantically related to query → fans out to neighbors, relationships, covariates, community reports → assembles context window → generates answer. Three traversal strategies: breadth-first, shortest-path, confidence-weighted.

**Global search:** For "sensemaking" questions. Aggregates community summaries across the hierarchy. Prunes irrelevant communities early. New dynamic search (2025) searches entire graph instead of fixed community level.

**Relevance to maasv:** The local search pattern (entity identification → neighborhood extraction → LLM reasoning) maps directly to what maasv needs. Community detection and summarization are valuable for pre-computing "what is this cluster about?" answers. The community hierarchy is overkill for hundreds of entities but the concept of entity-seeded subgraph extraction is directly applicable.

**What to borrow:** Entity-seeded local search. Subgraph serialization patterns. The principle that graph structure determines WHAT to retrieve, LLM determines HOW to answer.

### HippoRAG
- **Paper:** [HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](https://arxiv.org/abs/2405.14831) (Gutiérrez et al., NeurIPS 2024)
- **Code:** [github.com/OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)

**Architecture:** Inspired by hippocampal indexing theory. LLM acts as "neocortex" for pattern recognition. Knowledge graph serves as "hippocampal index." Personalized PageRank acts as "pattern completion" — filling in gaps from partial cues.

**Key innovation:** When a query arrives, extract entities → find them in the knowledge graph → run PPR from those seed nodes → retrieve passages associated with high-PPR nodes. This naturally handles multi-hop: if the query mentions entity A, and A connects to B which connects to C, PPR assigns non-zero weight to C even though it's 2 hops away.

**Performance:** Outperforms standard RAG on multi-hop QA by up to 20%. Achieves comparable or better performance than iterative retrieval (IRCoT) while being 10-20x cheaper and 6-13x faster.

**Relevance to maasv:** PPR is the missing piece in maasv's graph signal. Currently, `_find_memories_by_graph` does 1-hop expansion. Replacing or augmenting this with PPR would naturally extend to multi-hop without multiple traversal passes. The algorithm is simple to implement on an adjacency list.

**What to borrow:** Personalized PageRank for relevance scoring from seed entities. The biologically-inspired framing is elegant but the practical win is PPR as a graph signal.

### Zep / Graphiti
- **Paper:** [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/abs/2501.13956) (Rasmussen et al., January 2025)
- **Code:** [github.com/getzep/graphiti](https://github.com/getzep/graphiti)

**Architecture:** Three-tier graph: episode subgraph (raw events), semantic entity subgraph (extracted entities/relationships), community subgraph (clustered themes). Bi-temporal model with explicit validity intervals on every edge. Uses Neo4j as backing store.

**Cross-entity queries:** Three search functions: cosine similarity on entity embeddings, BM25 full-text, and BFS from semantically relevant entities. Multi-hop retrieval via BFS respects temporal consistency and edge validity.

**Entity extraction:** Continuous ingestion pipeline. Extracts and resolves entities against existing graph in real-time. Supports hyper-edges (multi-entity facts).

**Relevance to maasv:** Very similar problem domain (agent memory, temporal knowledge). Zep's bi-temporal model mirrors what maasv already has (`valid_from`/`valid_to` + `ingested_at`). Their three-search-function approach (cosine + BM25 + graph traversal) parallels maasv's 3-signal retrieval. The main thing maasv lacks is the community subgraph layer.

**Key difference:** Zep uses Neo4j; maasv uses SQLite. At maasv's scale, SQLite is sufficient and avoids the Neo4j dependency. The algorithms (BFS, community detection) are implementable directly.

### Mem0
- **Paper:** [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413)
- **Docs:** [docs.mem0.ai/open-source/features/graph-memory](https://docs.mem0.ai/open-source/features/graph-memory)

**Architecture:** Optional graph memory layer on top of vector-based retrieval. Entity extraction → conflict detection → LLM-powered update resolver (add, merge, invalidate, skip). Graph enables subgraph retrieval and semantic triplet matching for multi-hop queries.

**Cross-entity approach:** Mem0's graph memory enables multi-hop reasoning by traversing relationship paths. Performance: graph memory adds ~2% overall accuracy improvement over base vector retrieval, with notable gains on multi-hop questions specifically.

**Relevance to maasv:** Mem0's architecture validates that graph memory is additive to vector retrieval (not a replacement). The 2% overall / larger multi-hop improvement matches what we'd expect — graph helps most on questions that require connecting dots across entities.

### LightRAG
- **Paper:** [LightRAG: Simple and Fast Retrieval-Augmented Generation](https://arxiv.org/abs/2410.05779) (EMNLP 2025)

**Architecture:** Builds a lightweight knowledge graph during ingestion. Dual-level retrieval: low-level (specific entities, direct relationships) and high-level (broader themes, conceptual relationships across entities). Incremental updates without full reconstruction.

**Relevance to maasv:** The dual-level retrieval concept maps to what maasv needs: low-level = current 1-hop expansion, high-level = the cross-entity synthesis layer we're adding. The incremental update design is important — maasv's graph grows continuously and can't afford batch recomputation.

### Neo4j Graph Data Science Library
- **Docs:** [neo4j.com/docs/graph-data-science/current/algorithms/](https://neo4j.com/docs/graph-data-science/current/algorithms/)

**Relevant algorithms:** Community detection (Louvain, Label Propagation, SLLPA for overlapping communities), centrality (PageRank, Betweenness, Closeness), path finding (shortest path, all shortest paths, spanning tree), similarity (node similarity, Jaccard, overlap).

**Relevance to maasv:** These are the graph primitives that maasv could implement natively in SQLite/Python. The most valuable for cross-entity synthesis: PPR (relevance scoring from seed entities), Louvain (community detection for clustering), shortest path (finding connections between arbitrary entities).

### SQLite-Based GraphRAG
- **Reference:** [How to Build Lightweight GraphRAG with SQLite](https://dev.to/stephenc222/how-to-build-lightweight-graphrag-with-sqlite-53le)

Demonstrates that GraphRAG patterns (entity extraction, relationship storage, centrality computation, graph-augmented retrieval) work well with SQLite for small-to-medium datasets. This validates maasv's approach of staying on SQLite rather than adding Neo4j.

---

## What's Practical vs. Academic

| Approach | Academic Interest | Practical for maasv | Why |
|----------|:--:|:--:|-----|
| TransE/RotatE/ComplEx | High | No | Graph too small for statistical embedding to learn meaningful patterns |
| DistMult | High | No | Same scale problem |
| GNN-based link prediction | High | No | Needs training data, GPU, overkill for hundreds of nodes |
| AMIE/AnyBURL rule mining | Medium | No | Insufficient statistical support at this scale |
| Hand-coded domain rules | Low | Yes | Simple, predictable, no training needed |
| LLM-based graph reasoning | High | **Yes** | Already in stack, handles world knowledge, no training |
| Personalized PageRank | Medium | **Yes** | Simple algorithm, powerful multi-hop signal, sub-millisecond |
| BFS/shortest path generalized | Low | **Yes** | Trivial to implement, directly answers "how are X and Y connected?" |
| Community detection (Louvain) | Medium | **Yes** | Pre-computes entity clusters for theme-based queries |
| Subgraph extraction + LLM | High | **Yes** | Core pattern from GraphRAG/HippoRAG, fits maasv architecture |
| Community summarization | High | Maybe v2 | Useful but not essential at hundreds of entities |
| Full GraphRAG pipeline | High | No (v1) | Too heavy for a personal KG; the local search pattern alone is valuable |
| Bi-temporal reasoning (Zep-style) | Medium | Already have it | maasv's valid_from/valid_to/ingested_at already supports this |

**The honest assessment:** Most KG completion research is solving a different problem than maasv has. They're trying to complete Wikidata where you have millions of triples and can learn statistical regularities. maasv's problem is that the data is already there but the system can't traverse or reason over it. This is a graph traversal + LLM reasoning problem, not a statistical inference problem.

---

## Recommended v1 Scope

The goal: let Doris answer multi-hop questions about Adam's life by traversing the knowledge graph and reasoning over the results. Minimum viable implementation that delivers real value.

### Core Components

**1. Multi-hop path finder** (`graph.py`)
Generalize `get_causal_chain` to work with all predicates. Given two entity IDs (or an entity ID and a type/predicate pattern), find all paths up to N hops.

```python
def find_paths(
    start_entity_id: str,
    end_entity_id: str | None = None,
    end_type: str | None = None,
    max_hops: int = 3,
    predicate_filter: set[str] | None = None,
) -> list[list[dict]]:
    """Find paths between entities or from entity to type."""
```

**2. Personalized PageRank** (`graph.py` or new `graph_algorithms.py`)
Implement PPR over the adjacency list. Given seed entity IDs (from query), compute relevance scores for all reachable entities. Use as a new signal in retrieval.

```python
def personalized_pagerank(
    seed_entity_ids: list[str],
    alpha: float = 0.15,  # restart probability
    max_iterations: int = 20,
    min_score: float = 1e-6,
) -> dict[str, float]:
    """Compute PPR scores from seed entities."""
```

This is a straightforward power-iteration algorithm. At maasv's scale (hundreds of nodes), convergence is near-instant.

**3. Subgraph extraction** (`graph.py`)
Extract a k-hop neighborhood around seed entities, formatted for LLM consumption.

```python
def extract_subgraph(
    seed_entity_ids: list[str],
    max_hops: int = 2,
    max_entities: int = 50,
    include_values: bool = True,
) -> dict:
    """Extract neighborhood subgraph for LLM reasoning."""
```

Returns structured data: entities with types, relationships with predicates and confidence, formatted as a compact text representation the LLM can reason over.

**4. LLM synthesis function** (new `core/synthesis.py` or extend `lifecycle/inference.py`)
Given a question and a subgraph, ask the LLM to synthesize an answer.

```python
def synthesize_answer(
    question: str,
    subgraph: dict,
    model: str | None = None,
) -> dict:
    """Use LLM to reason over graph context and answer a multi-hop question."""
```

The prompt template would include the serialized subgraph and ask the LLM to: identify relevant paths, apply world knowledge (e.g., geographic proximity), synthesize an answer, cite which entities/relationships support the answer.

**5. Enhanced graph signal in retrieval** (`retrieval.py`)
Replace the 1-hop expansion in `_find_memories_by_graph` with PPR-based expansion. Instead of expanding one hop and searching for entity names, compute PPR from seed entities and search for the top-K PPR entities.

### What's NOT in v1

- Automatic relationship inference (pre-computing inferred edges) — save for v2 sleep-time inference
- Community detection and summarization — useful but not essential for the primary use case
- Custom query language — queries come through natural language, decomposed by the LLM
- Entity embeddings in a shared space — the graph is small enough that PPR + LLM reasoning covers the same ground
- Rule learning from graph patterns — not enough data to learn rules statistically

---

## First Use Case in Doris

**"Who do I know that lives near my Hudson Valley house?"**

### How it would work with v1:

1. **Query decomposition** (existing or new): Parse "Hudson Valley house" → find entity "Hudson Valley" (place) and "Adam" (person) with `has_property_in` relationship.

2. **Subgraph extraction**: Extract 2-hop neighborhood from Hudson Valley entity. This picks up: all entities with `lives_in`, `located_in`, `visited` relationships to Hudson Valley or nearby places.

3. **PPR from seed entities**: Run PPR seeded from [Hudson Valley, Adam]. People connected to both Adam and Hudson Valley-area places get highest scores.

4. **LLM synthesis**: Pass the subgraph + question to Claude Haiku. The LLM applies world knowledge: "Beacon is in the Hudson Valley. Cold Spring is in the Hudson Valley. Peekskill is in the Hudson Valley." It identifies people connected to those places.

5. **Response**: "Based on your knowledge graph, Sarah lives in Beacon which is in the Hudson Valley. You also visited Cold Spring with the Martins last summer."

### Cost estimate:
- PPR computation: <1ms on hundreds of nodes
- Subgraph extraction: <5ms (SQL queries)
- LLM call (Claude Haiku 4.5, ~2K input + 500 output tokens): ~$0.005
- Total latency: ~500ms (dominated by LLM API call)

### Second use case: "What technologies do all my projects use?"

1. Entity search for "project" type → returns Doris, TerryAnn, maasv, etc.
2. Extract subgraph: all `built_with`, `uses_tech`, `runs_on`, `written_in` relationships from project entities
3. LLM aggregates: "Python is used by Doris and maasv. Next.js by TerryAnn. All use SQLite."
4. No PPR needed here — just type-filtered relationship aggregation + LLM synthesis.

---

## Dependencies in maasv Codebase

### Builds on existing code

| File | What it provides | What changes |
|------|-----------------|-------------|
| `core/graph.py` | Entity/relationship CRUD, `get_causal_chain` BFS pattern | Add `find_paths`, `personalized_pagerank`, `extract_subgraph` |
| `core/retrieval.py` | 3-signal pipeline, `_find_memories_by_graph` | Replace 1-hop expansion with PPR-based expansion |
| `core/db.py` | Database schema, connection management | No schema changes needed for v1 (graph tables already exist) |
| `extraction/entity_extraction.py` | Entity/relationship extraction from text | No changes needed |
| `lifecycle/inference.py` | Sleep-time inference pattern | Extend or create sibling for cross-entity synthesis |
| `lifecycle/worker.py` | Sleep worker, job types | Possibly add `SYNTHESIS` job type for pre-computation |
| `config.py` | Configuration dataclass | Add synthesis-related config (max_hops, ppr_alpha, synthesis_model) |

### New code needed

| Component | Location | Complexity |
|-----------|----------|-----------|
| `personalized_pagerank()` | `core/graph.py` or new `core/graph_algorithms.py` | Low — 30-50 lines of power iteration |
| `find_paths()` | `core/graph.py` | Low — generalize existing `get_causal_chain` |
| `extract_subgraph()` | `core/graph.py` | Low — k-hop BFS + serialization |
| `synthesize_answer()` | new `core/synthesis.py` | Medium — prompt engineering + response parsing |
| PPR integration in retrieval | `core/retrieval.py` | Medium — replace `_find_memories_by_graph` internals |
| MCP server endpoint | `mcp_server/server.py` | Low — expose synthesis via existing MCP pattern |
| Tests | `tests/test_synthesis.py` | Medium — need graph fixtures, mock LLM |

### No new dependencies

Everything can be implemented with Python standard library + existing SQLite schema. No Neo4j, no PyTorch, no sentence-transformers. PPR is pure Python math on an adjacency list. Graph algorithms are BFS/DFS on SQL results.

---

## Open Questions

1. **Should synthesis be query-time or pre-computed?** Query-time (call LLM per question) gives freshest answers but adds ~500ms latency. Pre-computed (sleep-time job that periodically infers relationships) gives instant answers but may be stale. Probably both: pre-compute common patterns, fall back to query-time for novel questions.

2. **How to serialize subgraphs for the LLM?** Options: JSON triples, natural language sentences ("Adam lives in Upper West Side. Adam has property in Hudson Valley."), or a structured format. Need to test which format gives best reasoning quality from Haiku.

3. **Confidence propagation.** If A→B has confidence 0.9 and B→C has confidence 0.8, what confidence should an inferred A→C relationship have? Simple: product (0.72). But should path length discount further? PPR naturally handles this via the damping factor.

4. **Where does synthesis live in the Doris call flow?** Options: (a) Doris explicitly calls a `graph_synthesize` MCP tool, (b) maasv's retrieval pipeline automatically includes synthesis when multi-hop results are detected, (c) both. Probably (c) — automatic for retrieval enrichment, explicit tool for direct graph questions.

5. **How to handle geographic reasoning?** "Near Hudson Valley" requires knowing that Beacon, Cold Spring, Peekskill are all in the Hudson Valley. Options: (a) rely entirely on LLM world knowledge during synthesis, (b) add `located_in` edges for geographic hierarchy (Beacon → Hudson Valley → New York), (c) add a geocoding step. For v1, (a) is sufficient. (b) would help if Adam's graph has these facts already.

6. **Should inferred relationships be persisted?** If the LLM infers "Adam probably commutes through the Hudson Valley on weekends," should that become a graph edge? Arguments for: avoids re-inference. Arguments against: stale inferences are worse than no inference, and the LLM might hallucinate relationships. Recommendation: persist with `source="llm_synthesis"`, low confidence, and a TTL for automatic expiry.

7. **PPR damping factor tuning.** Alpha=0.15 is the classic default (meaning 15% chance of restarting at a seed node per step). For a personal KG where most relevant entities are 1-2 hops away, a higher alpha (0.2-0.3) might be better to keep scores concentrated near seed entities rather than diffusing across the whole graph. Needs empirical testing.

8. **Cost budget for synthesis.** Each synthesis call costs ~$0.005 with Haiku. How many synthesis calls per Doris session is acceptable? At current usage patterns (dozens of queries per day), even $0.10/day for synthesis is well within budget. But should there be a per-query or per-session cap?

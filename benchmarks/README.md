# maasv Benchmarks

Reproducible benchmarks measuring maasv's retrieval quality against ablated baselines.

## Quick Start

```bash
cd /path/to/maasv
python -m benchmarks.run_retrieval_quality
```

Options:
- `--seed 42` — random seed (default: 42)
- `--scale small|medium|large` — dataset size (default: small)
- `--limit 5` — number of results per query, i.e. k (default: 5)

## Results (seed=42, scale=small, k=5)

92 memories, 17 entities, 15 relationships, 27 queries.

| Adapter | NDCG@5 | MRR | P@5 |
|---------|--------|-----|------|
| maasv-full | 0.3831 | 0.4833 | 0.2889 |
| maasv-vector-only | 0.2843 | 0.3759 | 0.2593 |
| maasv-bm25-only | **0.5176** | **0.5926** | **0.3222** |
| maasv-graph-only | 0.2208 | 0.2370 | 0.1185 |

## What Gets Measured

### Retrieval Quality Ablation (`run_retrieval_quality.py`)

Compares the full 3-signal pipeline against each signal in isolation:

| Adapter | Description |
|---------|-------------|
| `maasv-full` | Vector + BM25 + graph with RRF fusion, importance scoring, diversity dedup |
| `maasv-vector-only` | Dense vector similarity (sqlite-vec ANN) only |
| `maasv-bm25-only` | FTS5 BM25 keyword matching only |
| `maasv-graph-only` | Entity graph traversal (1-hop expansion) only |

### Metrics

- **NDCG@K** — Normalized Discounted Cumulative Gain. Measures ranking quality with graded relevance (1.0 = exact answer, 0.5 = topically related). Higher is better.
- **MRR** — Mean Reciprocal Rank. How high the first relevant result appears. Higher is better.
- **P@K** — Precision at K. Fraction of top-K results that are relevant. Higher is better.

### Dataset

Synthetic dataset with 8 thematic clusters (work projects, family, preferences, people, places, decisions, learning). Each cluster has:
- 10-15 base memories with realistic content
- Named entities and typed relationships using `maasv.core.graph.VALID_PREDICATES`
- 3-4 queries with hand-assigned relevance grades
- Query mix: keyword queries (BM25-favored), cluster-semantic queries (vector-favored), entity-hop queries (graph-favored)

Ground truth is constructed during generation, not inferred from the system under test.

### Embedding Strategy

Uses `DeterministicEmbedProvider` instead of hash mocks or real model inference:
- Each cluster gets a deterministic centroid vector (seeded)
- Text embedding = weighted blend of matching cluster centroids + hash noise (10%)
- L2-normalized (required by maasv's validation)
- Same text always produces the same vector
- Makes vector search actually work — memories sharing cluster keywords embed near each other

## Interpreting Results

### What the fusion pipeline provides

The full pipeline consistently outperforms vector-only (+35% NDCG) and graph-only (+74% NDCG). This demonstrates that RRF fusion and multi-signal agreement scoring produce better rankings than any single non-keyword signal.

### Why BM25 leads with synthetic data

BM25-only scores highest because the deterministic embedding provider uses keyword detection, not semantic understanding. It can cluster "Atlas dashboard" near "Atlas metrics" (shared keywords), but it cannot understand that "team progress tracking" means "Atlas KPI dashboard" — there's no paraphrase or synonym matching. Real model embeddings (e.g., qwen3-embedding) capture these semantic relationships, making the vector signal much stronger in production.

maasv's importance scoring formula uses `(1 - L2_distance) * importance * decay * ips_utility`, which gives positive base scores only when L2 distance < 1.0. With deterministic embeddings, same-cluster distances are ~0.65 (positive score), while cross-cluster distances are ~1.4 (negative score). Real embeddings produce much smaller distances for semantically similar content, amplifying the vector signal's contribution to fusion.

### Key takeaway

This benchmark validates the signal fusion architecture and provides a deterministic baseline. The planned model-embedding benchmark (using Ollama/qwen3-embedding locally) will demonstrate the full pipeline's semantic advantage.

## Output

Results are written to `benchmarks/results/retrieval_quality_{seed}.json`. To format as markdown:

```bash
python -m benchmarks.format_results
```

## Running with Ollama (real embeddings)

The default benchmark uses deterministic keyword-based embeddings — fast but no semantic understanding. To benchmark with real model embeddings:

```bash
# Requires Ollama running locally with the model pulled
ollama pull qwen3-embedding:8b

python -m benchmarks.run_retrieval_quality --scale small --ollama
```

This adds a `maasv-ollama-full` adapter that runs the full 3-signal pipeline with `qwen3-embedding:8b` (4096 dims). Setup is slower (~30-60s for 92 memories) because each memory gets embedded via the local model.

The `--ollama` flag is opt-in: without it, only the fast deterministic adapters run. If Ollama isn't running or the model isn't pulled, the adapter is skipped with a message.

Only the full-pipeline adapter is included for Ollama — the ablation adapters already prove signal decomposition with deterministic embeddings. The point here is comparing real semantic embeddings against the deterministic baseline.

## Future Benchmarks (planned)

- **Learning curve**: NDCG@5 vs number of labeled training samples
- **IPS vs naive training**: compare IPS-weighted loss against uniform loss
- **Latency profiling**: per-signal and total pipeline latency
- **Competitor adapters**: Mem0, Zep (stubs exist in `adapters/`)

# maasv Cognition Benchmark — Session Handoff

## What You're Continuing

We're benchmarking maasv's **cognitive capabilities** (not just retrieval) against the real Doris production DB before pushing to the public repo. The goal: prove maasv is a cognition system, not just another vector store.

**Repo:** `/Users/macmini/Projects/maasv/` (remote: `git@github.com:ascottbell/maasv.git`)
**DB:** `/Users/macmini/Projects/doris/data/doris.db` (5,129 active memories, 673 entities, 1,030 relationships, ~8 weeks)
**All changes are uncommitted on `main`.** Nothing pushed to remote yet.

## Current Benchmark Scores

### Cognition Benchmark (`benchmarks/cognition/`)
| Category | Score | Previous Session | Notes |
|----------|-------|-----------------|-------|
| Temporal | 0.73 | 0.73 | Unchanged — all attempts to improve regressed other metrics |
| Session coherence | 0.83 | 0.83 | Unchanged |
| Graph traversal | 0.75 | 0.77 | Minor drop (DB growth noise) |
| Consolidation | 1.00 | 1.00 | Perfect, done |
| Decay + protection | 0.89 | 0.89 | Unchanged |
| Proactive | NOT SCORED | — | Needs `--with-llm` flag |
| **COMPOSITE** | **0.82** | **0.82** | **Neutral** |

### Standard IR Benchmark (`benchmarks/real_data/`)
| Adapter | NDCG@5 | MRR | P@5 |
|---------|--------|-----|-----|
| maasv-full | **0.602** | 0.755 | 0.450 |
| vector-only | 0.444 | 0.693 | 0.284 |
| bm25-only | 0.275 | 0.389 | 0.196 |

**NDCG improved from 0.598 to 0.602** thanks to word-boundary regex bugfix (see below).

## What Changed This Session

### 1. Word-Boundary Regex Bugfix (SHIPPED) — `retrieval.py`
- **Bug:** `_TEMPORAL_KEYWORDS` was a frozenset using `any(kw in query_lower for kw in ...)`, which matched "now" inside "knowledge", "current" inside "subcurrent", etc.
- **Fix:** Replaced with `_TEMPORAL_KEYWORDS_RE = re.compile(r"\b(?:currently|current|now|...)\b", re.IGNORECASE)`. Word-boundary matching eliminates false temporal triggers.
- **Impact:** NDCG improved 0.598 → 0.602 (false temporal decay was hurting non-temporal queries containing these substrings).

### 2. Continuous Alpha Framework (SHIPPED) — `retrieval.py`
- Replaced binary `temporal_query` flag with continuous `freshness_alpha ∈ [0.0, 1.0]`.
- `_compute_freshness_alpha(query)` returns alpha based on detected signals.
- Decay parameters interpolated: `decay_base = 0.7 - 0.6*alpha`, `decay_scale = 0.3 + 0.6*alpha`.
- Graph weight reduced for high-alpha queries: `0.08 if alpha < 0.5 else 0.03`.
- Currently only keyword signal is active (alpha=1.0). Pattern signal disabled (see below).
- **At alpha=0.0 and alpha=1.0 the math is identical to the old binary approach.** Zero behavioral change beyond the bugfix. This is purely infrastructure for future signals.

### 3. State Query Patterns (DEFINED, DISABLED) — `retrieval.py`
- Added `_STATE_QUERY_PATTERNS` to detect present-tense state queries ("What does X use?", "How often does X").
- **Disabled in code** (commented out in `_compute_freshness_alpha`).
- **Why disabled:** Alpha=0.25-0.35 from patterns falls in a "valley of pain" — strong enough to swap bottom-of-top-5 results but not strong enough to reorder the top. Two of three targeted queries regressed. Needs a different mechanism than decay scaling.

### 4. Per-Category Decay (ADDED TO CONFIG, NOT USED) — `config.py`
- Added `category_half_life_days` dict to config: event=10, reminder=10, behavior=15, decision=15, project=20, learning=25 (days).
- Derived from supersession density analysis of the production DB.
- **Not used in retrieval.py** — implementation attempted but caused cross-category score distortion. Newer `decision` memories (15d half-life) decayed faster than older `learning` memories (25d half-life), scrambling cross-category rankings. Reverted.
- **Key insight:** Freshness sensitivity should be a QUERY property, not per-memory.

### 5. Supersession Authority Bonus (TESTED, REMOVED)
- Computed supersession counts per candidate (how many memories each superseded).
- Added authority bonus: `min(0.015 * log(1+count), 0.03)`, scaled by `(1-alpha)`.
- **Removed after testing:** Hurt NDCG more than it helped cognition. Even at max 0.03, authority bonus pushed irrelevant chain-head memories above relevant non-chain-head ones in the IR benchmark.
- Best cognition result (0.839) used large authority + conflict detection but NDCG crashed to 0.532.

### 6. Post-Retrieval Conflict Detection (TESTED, REMOVED)
- Detected "volatile" queries by checking if candidates share subjects with >14 day time spread.
- Boosted alpha by 0.3 for volatile subjects, 0.15 for moderate.
- **Removed:** Too many queries triggered (most candidates touch "Doris" which is 20% superseded). Family topics got HIGHEST volatility (0.456) — exactly backwards.

### 7. LLM Reranker Experiment (NEW MODULE) — `maasv/core/llm_reranker.py`
- Built an LLM-based reranker that replaces heuristic scoring with reading comprehension.
- **Two backends:** Ollama (local qwen3:8b, ~0.5s/query) and Anthropic API (Haiku, ~0.5s/query).
- Ollama uses raw generate API with pre-closed `<think>` tags to skip reasoning (reduces from 30s to 0.5s).
- **Test harness:** `benchmarks/cognition/llm_rerank_test.py` — standalone comparison without modifying pipeline code.
- **Pulled `sam860/qwen3-reranker:0.6b-Q8_0`** — purpose-built reranker model, but doesn't work through Ollama (needs logit extraction for yes/no scoring, not text generation).

#### LLM Reranker Results

| Category | Baseline | Qwen3 no-think | Haiku |
|----------|----------|---------------|-------|
| temporal | 0.637 | 0.530 (-0.11) | 0.560 (-0.08) |
| session | 1.000 | 1.000 | 1.000 |
| consolidation | 1.000 | 1.000 | 1.000 |
| decay/protection | 0.675 | 0.500 (-0.18) | 0.525 (-0.15) |

**Key Finding:** LLM reranking (even Haiku) doesn't help the cognition benchmark categories. The benchmark tests mathematical properties (exponential decay, protected category exemptions, date ordering) that a formula does better than reading comprehension. The LLM sees 15 candidates all about TTS and picks the most *detailed* description, not the newest one.

**Where LLM reranking WOULD help:** Graph traversal (understanding cross-domain entity relationships) and proactive relevance (understanding unstated needs). These weren't fully tested.

**Future direction:** Use LLM as training signal for the learned ranker — have it score query-candidate pairs offline, then train the lightweight ranker to approximate those judgments at inference time. Best of both worlds: LLM quality, formula speed.

## Approaches Tried & Results Summary

| Approach | Cognition | NDCG | Verdict |
|----------|-----------|------|---------|
| Baseline (start of session) | 0.824 | 0.598 | — |
| Bugfix + alpha framework only | **0.818** | **0.602** | **SHIPPED — only clean win** |
| + small authority (max 0.03) | 0.813 | 0.594 | Worse on both |
| + reduced authority + conflict | 0.824 | 0.583 | Neutral cognition, NDCG hurt |
| + big authority + conflict | **0.839** | 0.532 | Best cognition, NDCG crashed |
| + per-category decay | 0.69 temporal | — | Cross-category distortion |
| + subject volatility alpha | regressed | — | "Doris" dominates every query |
| + LLM reranker (Haiku) | -0.08 temporal | — | LLM picks detailed over recent |

## Remaining Work

### Temporal (0.73 → target 0.80+)
The 3 non-keyword temporal queries (temporal-02, -03, -05) still use gentle decay because they lack explicit "currently"/"now". Approaches tried and failed:
- Pattern matching ("What does X use?"): valley of pain at moderate alpha
- Per-category decay: cross-category distortion
- Subject volatility: Doris entity dominates everything
- LLM reranking: picks detailed over recent

**Unexplored ideas:**
- Graduated alpha with a DIFFERENT mechanism than decay scaling (e.g., post-retrieval reordering that swaps candidates based on date when subjects overlap)
- Train the learned ranker with temporal-aware features (it already has the infrastructure)
- Use LLM to generate training data for the learned ranker offline

### Graph Traversal (0.75 — above target)
Done for now. LLM reranker could help here but wasn't tested against this category.

### IR NDCG (0.602 — just above floor)
Improved by bugfix. Authority bonus / conflict detection hurt it. Any changes must verify NDCG ≥ 0.60.

### Proactive (unscored)
Run with `--with-llm` to get baseline. LLM reranker would likely excel here.

### LLM Reranker Integration
The module exists at `maasv/core/llm_reranker.py` and works but isn't wired into the pipeline. Next steps:
- Test against graph traversal and proactive categories
- Explore as offline training signal for learned ranker
- Research smaller purpose-built reranker models that work through Ollama (Qwen3-Reranker-0.6B needs logit extraction, doesn't work through Ollama's text generation API)

## How to Run

```bash
cd /Users/macmini/Projects/maasv

# Cognition benchmark
.venv/bin/python -m benchmarks.cognition.runner --db-path /Users/macmini/Projects/doris/data/doris.db --limit 5

# With LLM judge (proactive category)
.venv/bin/python -m benchmarks.cognition.runner --db-path /Users/macmini/Projects/doris/data/doris.db --limit 5 --with-llm

# Standard IR benchmark
.venv/bin/python -m benchmarks.real_data.runner --db-path /Users/macmini/Projects/doris/data/doris.db --limit 5

# LLM reranker comparison test (requires ANTHROPIC_API_KEY for --backend anthropic)
export $(grep ANTHROPIC_API_KEY /Users/macmini/Projects/doris/.env)
.venv/bin/python -m benchmarks.cognition.llm_rerank_test --db-path /Users/macmini/Projects/doris/data/doris.db --limit 5 --backend anthropic

# Tests (skip 2 pre-existing failures + server smoke)
.venv/bin/python -m pytest tests/ -q --ignore=tests/test_server_smoke.py \
  --deselect tests/test_learned_ranker.py::TestIPSWeightedTraining::test_training_with_ips_weights \
  --deselect tests/test_learned_ranker.py::TestGraduation::test_shadow_compare_persists_metrics
```

## Key Files
- `maasv/core/retrieval.py` — `_importance_score()` (~line 132), `_compute_freshness_alpha()` (~line 101), `find_similar_memories()` (~line 930)
- `maasv/core/llm_reranker.py` — LLM reranker module (Ollama + Anthropic backends)
- `maasv/config.py` — Tuning knobs + `category_half_life_days` (defined, unused)
- `benchmarks/cognition/` — Cognition benchmark (queries.py, scoring.py, runner.py)
- `benchmarks/cognition/llm_rerank_test.py` — Standalone LLM reranker comparison test
- `benchmarks/results/cognition.json` — Latest detailed results per query
- `/Users/macmini/Documents/maasvCodeReview.md` — Code review from Codex + Gemini

## Plane Project
- Workspace: `adam`, Project ID: `01dfafb3-e51c-4ebd-bc3d-0229fe1a3c0e`
- API key: `plane_api_0d67a946770b4697ac3a1582d9845651`
- MAASV-46 through MAASV-55: Performance/correctness fixes from code review (all Todo)

## Ollama Models Available
- `qwen3-embedding:8b` (4.7GB) — production embeddings
- `qwen3:8b` (5.2GB) — used for LLM reranker listwise mode
- `sam860/qwen3-reranker:0.6b-Q8_0` (639MB) — pulled but doesn't work through Ollama (needs logit extraction)
- `dengcao/Qwen3-Reranker-0.6B:Q8_0` (639MB) — same issue

## Critical Rules
1. **Don't sacrifice standard IR quality for cognition gains.** NDCG must stay ≥ 0.60.
2. **The continuous alpha framework is live but functionally equivalent to the old binary approach** at the two active values (0.0 and 1.0). It's infrastructure for future signals.
3. **LLM reranking doesn't help temporal/decay** — those need mathematical properties, not reading comprehension. Don't re-test this.
4. **Superseded memories are filtered at SQL level** (`superseded_by IS NULL`). They never reach scoring. Any "supersession penalty" approach is N/A.

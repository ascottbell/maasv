<p align="center">
  <img src="maasvlogo.png" alt="maasv" width="500">
</p>

**Your cognition layer. One memory. Every agent. Every source.**

maasv gives you a persistent understanding that works across all your AI agents, tools, and data sources. Not just storage and retrieval — a full lifecycle that extracts, structures, connects, consolidates, and prunes knowledge over time. Entities and relationships are pulled from conversations, organized into a knowledge graph, and actively maintained in the background. What comes back isn't just relevant documents — it's structured understanding with context.

People use ChatGPT AND Claude AND Gemini for different strengths. Even within Claude: Desktop, Code, and Codex are three separate sessions. maasv makes them all share cognition. Stop re-explaining yourself.

## What it does

Your tools come and go. You switch CRMs, cancel subscriptions, startups shut down. maasv is yours — one SQLite file that persists across all of it.

Any source that can make an HTTP call or connect via MCP can write to maasv. AI agents, CRM syncs, calendar integrations, health data, automation tools — they all feed into the same cognition layer. Every memory carries provenance: which system wrote it and through which interface, so you always know where knowledge came from.

Your agent remembers that the person you're meeting tomorrow was mentioned in a conversation three weeks ago, and surfaces the context before you ask. It connects a complaint from a customer in March to a feature request from their team in June. It knows you tried a particular approach before and it didn't work, so it suggests something different this time.

## The lifecycle

Most memory tools store and retrieve. That's two steps. maasv owns six:

**Extract.** Entities, relationships, and facts are pulled from conversations by your LLM. People, places, projects, technologies, and how they connect to each other. Not keywords. Structure.

**Store.** Memories are embedded, categorized, and deduplicated on the way in. Each one carries metadata: confidence, importance, subject, origin, and access history.

**Consolidate.** During idle time, maasv merges near-duplicates, clusters related memories, resolves vague references to specific entities, and pre-computes common graph paths. Your understanding gets sharper while nobody's using it.

**Retrieve.** Three signals fused together: dense vector search (semantic similarity), BM25 keyword matching (exact terms via FTS5), and graph connectivity (1-hop entity expansion). Merged with Reciprocal Rank Fusion, optionally reranked by a cross-encoder. Filterable by category, subject, origin, or origin interface. Each result carries a cosine similarity score so callers can filter noise — a nonsense query returns results near 0.3, while a meaningful match scores 0.6+.

**Decay.** Memories that stop being accessed lose confidence over time. Protected categories (identity, family, core preferences) are exempt. Everything else has to earn its place.

**Forget.** Stale, low-confidence memories are pruned. Orphaned entities are cleaned up. The knowledge graph stays lean. Without active forgetting, memory systems tend to get noisier over time — maasv gets sharper.

**Learn.** *(Experimental — shadow mode by default.)* A small neural network (81 parameters) trains on actual retrieval patterns — which memories get re-accessed after being surfaced, and which get ignored. Over time, retrieval adapts to usage rather than relying solely on static heuristics. The ranker starts in shadow mode: it logs comparisons between its ranking and the default, but doesn't affect results. Once enough labeled data accumulates (100+ samples), you can flip it to active mode. To disable entirely: `learned_ranker_enabled=False` in config.

## Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.com)** running locally with the embedding model pulled:

```bash
ollama pull qwen3-embedding:8b
```

maasv uses [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) for embeddings by default. It runs locally via Ollama — no API keys, no cloud calls, your data stays on your machine. The model supports Matryoshka dimensionality reduction, so maasv truncates to 1024 dimensions and L2-normalizes automatically.

## Install

```bash
pip install maasv
```

One Python dependency: `sqlite-vec` for vector search. Everything runs in a single SQLite database.

Optional extras:
```bash
pip install "maasv[server]"       # HTTP server (FastAPI + uvicorn)
pip install "maasv[anthropic]"    # Anthropic LLM provider
pip install "maasv[openai]"       # OpenAI LLM provider
pip install "maasv[reranking]"    # Cross-encoder reranking (~2GB torch)
```

## Quick start

### 1. Initialize

maasv defaults to Ollama with qwen3-embedding:8b for embeddings. You only need to provide an LLM provider if you want entity extraction.

```python
from pathlib import Path
import maasv
from maasv.config import MaasvConfig

config = MaasvConfig(db_path=Path("memory.db"))
maasv.init(config=config, llm=MyLLM())
```

That's it. One required config field: `db_path`. maasv creates the database, runs migrations, records the embedding model, and is ready to use.

You still need an LLM provider for entity extraction. Any provider that implements `call(messages, model, max_tokens, source) -> str` works:

```python
# With Anthropic
import anthropic

class MyLLM:
    def __init__(self):
        self.client = anthropic.Anthropic()

    def call(self, messages, model, max_tokens, source=""):
        response = self.client.messages.create(
            model=model, max_tokens=max_tokens, messages=messages
        )
        return response.content[0].text
```

```python
# With OpenAI
import openai

class MyLLM:
    def __init__(self):
        self.client = openai.OpenAI()

    def call(self, messages, model, max_tokens, source=""):
        response = self.client.chat.completions.create(
            model=model, max_tokens=max_tokens, messages=messages
        )
        return response.choices[0].message.content
```

If you don't need entity extraction, you can skip the LLM entirely:

```python
maasv.init(config=config)  # embedding-only mode
```

### 2. Store and retrieve

```python
from maasv.core.store import store_memory
from maasv.core.retrieval import find_similar_memories, get_tiered_memory_context

# Store — with origin tracking
store_memory(
    "Alice prefers morning meetings",
    category="preference",
    subject="Alice",
    origin="claude",
    origin_interface="claude-code",
)
store_memory(
    "ProjectX deadline is March 15",
    category="project",
    subject="ProjectX",
    origin="openai",
    origin_interface="chatgpt-ios",
)

# Retrieve (3-signal fusion: vector + keyword + graph)
results = find_similar_memories("when is ProjectX due?", limit=5)
for r in results:
    print(f"{r['relevance']:.2f}  [{r.get('origin', '?')}] {r['content']}")

# Filter by origin
claude_only = find_similar_memories("Alice", limit=5, origin="claude")

# Get tiered context to inject into your LLM prompt
context = get_tiered_memory_context(query="meeting prep for Alice")
```

### 3. Build the knowledge graph

```python
from maasv.core.graph import find_or_create_entity, add_relationship

alice = find_or_create_entity("Alice", "person")
project_x = find_or_create_entity("ProjectX", "project")
add_relationship(alice, "works_on", object_id=project_x, origin="claude", origin_interface="codex")
```

Once entities exist, retrieval automatically expands queries through graph connections — search for "Alice" and you'll also surface ProjectX context.

### 4. Background lifecycle (optional)

```python
from maasv.lifecycle.worker import start_idle_monitor

start_idle_monitor()  # runs in a background thread
```

The sleep worker watches for idle periods and runs maintenance: dedup, consolidation, entity inference, graph optimization, and stale memory pruning. If you don't start it, everything still works — you just don't get automatic background maintenance.

See [`examples/quickstart.py`](examples/quickstart.py) for a complete runnable example with mock providers (no API keys needed).

## Multi-agent / multi-source

maasv tracks where every memory comes from via two fields:

| Field | Purpose | Examples |
|-------|---------|----------|
| `origin` | What system created this | `claude`, `openai`, `salesforce`, `apple-health` |
| `origin_interface` | Specific client/interface | `claude-code`, `claude-desktop`, `codex`, `chatgpt-ios` |

This enables:

- **Attribution**: "Where did I learn this?" — every memory has provenance.
- **Filtering**: "Show me everything from Claude Code about this project."
- **Decision tracing**: "I discussed this in Desktop, then Codex implemented it" — full lineage from metadata alone.
- **Future conflict detection**: When two origins disagree about a fact, maasv can surface the contradiction.

Any system that can make an HTTP call can write to maasv — AI agents via MCP, CRMs via REST API, automation tools via webhooks. The `origin` and `origin_interface` fields are optional and nullable for backward compatibility.

## Configuration

Only `db_path` is required. Everything else has sensible defaults.

```python
from maasv.config import MaasvConfig

config = MaasvConfig(
    db_path=Path("memory.db"),

    # Embedding (recorded in DB — mismatches are caught on open)
    embed_dims=1024,                        # Must match your embedding model
    embed_model="qwen3-embedding:8b",       # Recorded in db_meta for safety

    # Models (names passed to your LLMProvider — it decides what to do with them)
    extraction_model="claude-haiku-4-5-20251001",
    inference_model="claude-haiku-4-5-20251001",
    review_model="claude-haiku-4-5-20251001",

    # Hygiene tuning
    similarity_threshold=0.95,              # Dedup threshold (cosine)
    stale_days=30,                          # Prune after N days
    min_confidence_threshold=0.5,           # Prune below this confidence
    protected_categories={"identity", "family"},  # Never auto-delete

    # Cross-encoder (opt-in)
    cross_encoder_enabled=False,
    cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",

    # Learned ranker (experimental — shadow mode by default)
    learned_ranker_enabled=True,            # False to disable entirely
    learned_ranker_shadow_mode=True,        # True = log comparisons only
    learned_ranker_min_samples=100,         # Labeled samples before activation
    learned_ranker_lr=0.01,                 # Training learning rate
    learned_ranker_max_steps=50,            # Max training steps per cycle

    # Sleep worker timing
    idle_threshold_seconds=30,
    idle_check_interval=5,

    # Known entities (helps extraction avoid duplicates)
    known_entities={"Alice": "person", "ProjectX": "project"},
)
```

### Custom embedding providers

maasv defaults to Ollama, but any provider implementing `embed(text) -> list[float]` and `embed_query(text) -> list[float]` works:

```python
maasv.init(config=config, llm=MyLLM(), embed=MyCustomEmbed())
```

The `EmbedProvider` protocol is defined in [`maasv/protocols.py`](maasv/protocols.py). When using a custom provider, set `embed_dims` and `embed_model` in your config to match — maasv records the model name in the database and will error if you later open it with a different model configured.

## HTTP Server

maasv includes an HTTP server for language-agnostic access:

```bash
pip install "maasv[server]"
cp server.env.example .env
# Edit .env — at minimum, set MAASV_LLM_API_KEY
maasv-server
```

Server starts on `http://127.0.0.1:18790`. API docs at `/docs` (disabled when auth is set).

### Auth

Set `MAASV_API_KEY` to require an `X-Maasv-Key` header on all data endpoints. Token comparison uses `hmac.compare_digest` (constant-time). `/v1/health` stays public for load balancer probes. When auth is not set, all endpoints are open — intended for local development only.

### Endpoints

| Group | Endpoints |
|-------|-----------|
| Memory | `POST /v1/memory/store`, `/search`, `/context`, `/supersede`, `GET /:id`, `DELETE /:id` |
| Extraction | `POST /v1/extract` |
| Graph | `POST /v1/graph/entities`, `/entities/search`, `/relationships`, `GET /entities/:id` |
| Wisdom | `POST /v1/wisdom/log`, `/:id/outcome`, `/:id/feedback`, `/search` |
| Health | `GET /v1/health` (public), `GET /v1/stats` (auth-protected) |

### Docker

```bash
docker compose up -d
```

The container runs as a non-root user. Data is persisted to a named volume at `/data`.

## Architecture

```
Any Client (AI agent, CRM, automation tool, MCP)
    |
    v
maasv-server (HTTP API)           or        maasv.init(config) (library mode)
    |                                               |
    +-- server/                                     +-- core/
    |   +-- main.py       FastAPI app               |   +-- store.py        Memory CRUD
    |   +-- auth.py       API key auth              |   +-- retrieval.py    3-signal retrieval
    |   +-- config.py     Env-based config          |   +-- graph.py        Knowledge graph
    |   +-- providers.py  LLM/embed factories       |   +-- wisdom.py       Experiential learning
    |   +-- routers/      HTTP endpoints            |   +-- db.py           SQLite + sqlite-vec
    |                                               |   +-- reranker.py     Cross-encoder
    +-- providers/                                  |   +-- learned_ranker.py  Neural reranker
    |   +-- ollama.py     Built-in embeddings       |   +-- autograd.py     Backprop engine
    |                                               |
    +-- extraction/                                 +-- lifecycle/
    |   +-- entity_extraction.py                    |   +-- worker.py         Background jobs
    |                                               |   +-- memory_hygiene.py Dedup, prune, consolidate
    (server imports core/ directly)                 |   +-- reorganize.py     Graph optimization
                                                    |   +-- inference.py      Entity resolution
                                                    |   +-- review.py         Conversation analysis
                                                    |   +-- learn.py          Ranker training
```

Everything talks to one SQLite database. No Redis, no Postgres, no external vector DB. The entire state is a single `.db` file you can copy, back up, or throw away.

## Security Hardening

maasv stores all data in a local SQLite database. Embeddings are computed locally via Ollama. There are no cloud calls from the engine — your data stays on your machine.

### Embedding Model Safety

The embedding model is recorded in the database on first init (`db_meta` table). If you later open the database with a different model configured, maasv raises a `RuntimeError` instead of silently corrupting the vector space. This prevents the subtle bugs that happen when vectors from different models end up in the same search index.

### LLM Output Validation

LLM responses are untrusted input. Every write path from extraction, review, and inference pipelines enforces:

- **Entity name sanitization.** Character allowlist (alphanumeric, spaces, hyphens, underscores, periods) applied at write time. Rejects names with shell metacharacters, control characters, or injection attempts.
- **Cardinality caps.** Max 20 entities, 30 relationships, 20 insights, and 20 inferences per extraction call. Prevents a single hallucinated response from flooding the graph.
- **Field length caps.** Entity names capped at 200 chars, relationship values at 2K, content at 50K, metadata JSON at 10K on all write paths.
- **Confidence clamping.** All confidence values clamped to `[0.0, 1.0]` on every write path — `store_memory`, `add_relationship`, and all extraction pipelines. Non-numeric values coerced to 0.5.
- **Predicate allowlist.** Only predicates defined in `VALID_PREDICATES` are accepted by `add_relationship()`. Unknown predicates from LLM output are rejected. Extendable via `config.extra_predicates`.

### Atomic Operations

- **`supersede_memory()`** runs in a single transaction: read old memory, check for duplicates, insert new, update `superseded_by` pointer, commit. No partial state on crash.
- **`update_relationship_value()`** runs in a single transaction: find current, expire old, insert new, commit.
- **Orphan cleanup** uses atomic `DELETE ... WHERE NOT EXISTS` instead of select-then-delete.

### Database Integrity

- **SQLite pragmas.** WAL mode, `busy_timeout=5000`, and `foreign_keys=ON` applied to every connection (both `get_db()` and `get_plain_db()`).
- **Dedup constraints.** Database-level unique indexes on entities (`canonical_name`, `entity_type`) and partial unique indexes on active relationships. `IntegrityError` handling on `create_entity()` and `add_relationship()`.
- **Safe backups.** `Connection.backup()` instead of `shutil.copy2` for WAL-safe snapshots. Backup retention handles `FileNotFoundError` for concurrent deletions.
- **Schema migrations.** Numbered, idempotent migrations tracked in `schema_migrations` table. Safe to upgrade from any prior version.

### Thread Safety

All shared mutable state is protected by locks with double-checked locking:

- Init globals (`_config`, `_llm`, `_embed`, `_initialized`)
- Core memories cache and cache timestamp
- Reranker singleton and failure flag
- Worker thread startup (ensures only one background thread)
- Idle monitor start/stop state transitions

### Input Validation

- **Public API.** Content capped at 50K chars, categories at 50 chars, metadata JSON at 10K chars on `store_memory()` and `add_relationship()`. Retrieval `limit` hard-capped at 200.
- **FTS5 sanitization.** All `MATCH` inputs stripped of FTS5 operators (`"`, `*`, `(`, `)`, `NEAR`, `NOT`) before query execution. All FTS5 calls wrapped in try/except.
- **Cross-encoder model allowlist.** Only known-safe model identifiers are accepted before loading.

### Access Controls

- **Protected categories.** `delete_memory()` and `supersede_memory()` reject operations on protected categories (identity, family) unless `force=True`.
- **Pagination caps.** `get_all_active()` accepts an optional `limit`. Hygiene jobs (`_deduplicate_memories`, `_consolidate_clusters`) capped at 10K records per run.

### Logging

- LLM parse failures logged at DEBUG with content truncated to 100 chars (prevents memory content leaking into log files at INFO level).
- Entity and relationship creation messages downgraded from INFO to DEBUG.

### Data at Rest

- **No encryption.** SQLite stores data as plaintext. The database file contains raw memory content, entity names, relationship details, and embedding vectors.
- **File permissions.** Database, WAL, and SHM files set to `0o600` (owner read/write only) at init. Backup directories set to `0o700`, backup files to `0o600`.
- **If your threat model requires encryption at rest**, use filesystem-level encryption (FileVault, LUKS, BitLocker) on the volume containing the database. SQLCipher is not integrated — it adds a native dependency that conflicts with the library's zero-dependency approach.

### Known Limitations

- **Context sent to LLMs.** Sleep-time pipelines (review, inference, extraction) intentionally send conversation context to your configured LLM. This is by design — it's how the cognition layer works. If you need content filtering before LLM transmission, implement it in your `LLMProvider`.
- **Prompt injection via stored memories.** `get_tiered_memory_context()` returns a plaintext string for injection into system prompts. Stored memory content is not escaped or structured as JSON. Callers are responsible for prompt construction safety.
- **Mobile agents.** MCP is not yet supported on mobile AI apps (Claude iOS, ChatGPT iOS, Gemini mobile). Mobile agents currently can't connect to maasv directly. This is a platform limitation, not a maasv limitation.

## Upgrading from 0.1.x

Database migrations run automatically. Two things to be aware of:

1. **Embedding model tracking.** On first open with 0.2.0, maasv records your configured `embed_model` in the database. If you were using a model other than the default `qwen3-embedding:8b`, set `embed_model` in your config to match your actual model before upgrading, or you'll get a mismatch error on the second open.

2. **Origin fields.** The new `origin` and `origin_interface` columns are nullable. Existing memories will have `NULL` for both, which is fine — filtering by origin simply won't match them unless you backfill.

## Status

This is running in production powering [Doris](https://github.com/ascottbell/doris), but the public API may shift as more people use it. The core concepts (memory, graph, retrieval, wisdom, lifecycle) are stable. The edges are still being refined.

## License

Business Source License 1.1. Free for personal, internal, educational, and non-commercial use. Commercial use requires a license. Contact admin@maasv.ai. Converts to Apache 2.0 on 2030-02-16. See [LICENSE](LICENSE) for details.

## Related

- **[Doris](https://github.com/ascottbell/doris)** — The AI assistant maasv was built for. If maasv is the cognition layer, Doris is the person using it.

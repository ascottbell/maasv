<p align="center">
  <img src="maasvlogo.png" alt="maasv" width="500">
</p>

**A cognition layer for AI agents.**

maasv gives your agent a real memory — not just storage and retrieval, but a full lifecycle that extracts, structures, connects, consolidates, and prunes knowledge over time. Entities and relationships are pulled from conversations, organized into a knowledge graph, and actively maintained in the background. What comes back out when you query isn't just relevant documents — it's structured understanding with context.

## What it does

Your agent remembers that the person you're meeting tomorrow was mentioned in a conversation three weeks ago, and surfaces the context before you ask. It connects a complaint from a customer in March to a feature request from their team in June. It knows you tried a particular approach before and it didn't work, so it suggests something different this time.

The knowledge graph grows, consolidates, and prunes itself over time. Data comes in from disparate sources, gets structured into entities and relationships, and the connections between them become queryable. Your agent builds perspective across conversations, not just within them.

## The lifecycle

Most memory tools store and retrieve. That's two steps. maasv owns six:

**Extract.** Entities, relationships, and facts are pulled from conversations by your LLM. People, places, projects, technologies, and how they connect to each other. Not keywords. Structure.

**Store.** Memories are embedded, categorized, and deduplicated on the way in. Each one carries metadata: confidence, importance, subject, and access history.

**Consolidate.** During idle time, maasv merges near-duplicates, clusters related memories, resolves vague references to specific entities, and pre-computes common graph paths. Your agent's understanding gets sharper while nobody's using it.

**Retrieve.** Three signals fused together: dense vector search (semantic similarity), BM25 keyword matching (exact terms via FTS5), and graph connectivity (1-hop entity expansion). Merged with Reciprocal Rank Fusion, optionally reranked by a cross-encoder. This is how your agent finds the thing it didn't know it was looking for.

**Decay.** Memories that stop being accessed lose confidence over time. Protected categories (identity, family, core preferences) are exempt. Everything else has to earn its place.

**Forget.** Stale, low-confidence memories are pruned. Orphaned entities are cleaned up. The knowledge graph stays lean. Without active forgetting, memory systems tend to get noisier over time — maasv gets sharper.

**Learn.** *(Experimental — shadow mode by default.)* A small neural network (81 parameters) trains on your agent's actual retrieval patterns — which memories get re-accessed after being surfaced, and which get ignored. Over time, retrieval adapts to your agent's usage rather than relying solely on static heuristics. The ranker starts in shadow mode: it logs comparisons between its ranking and the default, but doesn't affect results. Once enough labeled data accumulates (100+ samples), you can flip it to active mode. To disable entirely: `learned_ranker_enabled=False` in config.

## Install

```bash
pip install maasv
```

One dependency: `sqlite-vec` for vector search. Everything runs locally in a single SQLite database. No external services, no API keys for the engine itself.

Optional extras:
```bash
pip install "maasv[server]"              # HTTP server (FastAPI)
pip install "maasv[server,anthropic,voyage]"  # Server + cloud providers
pip install "maasv[reranking]"           # Cross-encoder reranking (~2GB torch)
```

## Quick start

maasv requires an LLM provider (for entity extraction) and an embedding provider (for vector search). It ships with a built-in Ollama provider for fully local embeddings, or you can bring your own.

### 1. Set up providers

**Fastest path — local embeddings with Ollama:**

maasv includes a built-in embedding provider backed by [Ollama](https://ollama.com) and [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-8B). Qwen3-Embedding supports Matryoshka dimensionality reduction, so maasv can truncate to any dimension (default 1024) and L2-normalize automatically. No API keys needed for embeddings.

```python
import maasv
from maasv.config import MaasvConfig

config = MaasvConfig(db_path=Path("memory.db"), embed_dims=1024)
maasv.init(config=config, llm=MyLLM(), embed="ollama")
```

Pull the model first: `ollama pull qwen3-embedding:8b`

You still need an LLM provider for entity extraction — see below.

**With OpenAI:**

```python
import openai

class MyLLM:
    def __init__(self):
        self.client = openai.OpenAI()

    def call(self, messages, model, max_tokens, source=""):
        response = self.client.chat.completions.create(
            model=model, max_tokens=max_tokens, messages=messages
        )
        return response.choices[0].message.content

class MyEmbed:
    def __init__(self):
        self.client = openai.OpenAI()

    def embed(self, text):
        response = self.client.embeddings.create(
            model="text-embedding-3-large", input=text
        )
        return response.data[0].embedding

    def embed_query(self, text):
        return self.embed(text)
```

**With Anthropic:**

```python
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

Anthropic doesn't have a native embeddings API — pair it with any embedding provider (OpenAI, Voyage AI, Ollama, or a local model like `sentence-transformers`). The LLM and embedding providers don't need to come from the same vendor.

Any provider works. The protocol is just `call()` for LLM and `embed()`/`embed_query()` for embeddings. See [`maasv/protocols.py`](maasv/protocols.py) for the full type signatures.

### 2. Initialize

```python
from pathlib import Path
import maasv
from maasv.config import MaasvConfig

config = MaasvConfig(db_path=Path("memory.db"), embed_dims=3072)  # 3072 for text-embedding-3-large
maasv.init(config=config, llm=MyLLM(), embed=MyEmbed())
```

That's it. maasv creates the database, runs migrations, and is ready to use. The only required config is `db_path` and `embed_dims` (must match your embedding model's output dimensions).

### 3. Store and retrieve

```python
from maasv.core.store import store_memory
from maasv.core.retrieval import find_similar_memories, get_tiered_memory_context

# Store
store_memory("Alice prefers morning meetings", category="preferences", subject="Alice")
store_memory("ProjectX deadline is March 15", category="project", subject="ProjectX")

# Retrieve (3-signal fusion: vector + keyword + graph)
results = find_similar_memories("when is ProjectX due?", limit=5)

# Or get tiered context to inject into your LLM prompt
context = get_tiered_memory_context(query="meeting prep for Alice")
```

### 4. Build the knowledge graph

```python
from maasv.core.graph import find_or_create_entity, add_relationship

alice = find_or_create_entity("Alice", "person")
project_x = find_or_create_entity("ProjectX", "project")
add_relationship(alice, "works_on", object_id=project_x)
```

Once entities exist, retrieval automatically expands queries through graph connections — search for "Alice" and you'll also surface ProjectX context.

### 5. Background lifecycle (optional)

```python
from maasv.lifecycle.worker import start_idle_monitor

start_idle_monitor()  # runs in a background thread
```

The sleep worker watches for idle periods and runs maintenance: dedup, consolidation, entity inference, graph optimization, and stale memory pruning. If you don't start it, everything still works — you just don't get automatic background maintenance.

See [`examples/quickstart.py`](examples/quickstart.py) for a complete runnable example with mock providers (no API keys needed).

## Configuration

Only `db_path` and `embed_dims` are required. Everything else has sensible defaults.

```python
from maasv.config import MaasvConfig

config = MaasvConfig(
    db_path=Path("memory.db"),
    embed_dims=1024,                    # Must match your embedding model

    # Models (names passed to your LLMProvider -- it decides what to do with them)
    extraction_model="claude-haiku-4-5-20251001",
    inference_model="claude-haiku-4-5-20251001",
    review_model="claude-haiku-4-5-20251001",

    # Hygiene tuning
    similarity_threshold=0.95,          # Dedup threshold (cosine)
    stale_days=30,                      # Prune after N days
    min_confidence_threshold=0.5,       # Prune below this confidence
    protected_categories={"identity", "family"},  # Never auto-delete

    # Cross-encoder (opt-in)
    cross_encoder_enabled=False,
    cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",

    # Learned ranker (experimental — shadow mode by default)
    learned_ranker_enabled=True,           # False to disable entirely
    learned_ranker_shadow_mode=True,       # True = log comparisons only, don't affect results
    learned_ranker_min_samples=100,        # Labeled samples needed before model activates
    learned_ranker_lr=0.01,                # Training learning rate
    learned_ranker_max_steps=50,           # Max training steps per cycle

    # Sleep worker timing
    idle_threshold_seconds=30,
    idle_check_interval=5,

    # Known entities (helps extraction avoid duplicates)
    known_entities={"Alice": "person", "ProjectX": "project"},
)
```

## HTTP Server

maasv includes an optional HTTP server for language-agnostic access. Install with the `server` extra:

```bash
pip install "maasv[server,anthropic,voyage]"
cp server.env.example .env
# Fill in API keys
maasv-server
```

Server starts on `http://127.0.0.1:18790`. API docs at `/docs` (disabled when auth is set).

### Auth

Set `MAASV_API_KEY` to require an `X-Maasv-Key` header on all data endpoints. `/v1/health` stays public for load balancer probes. When auth is not set, all endpoints are open — intended for local development.

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
Your Agent / HTTP Client
    |
    v
maasv.init(config, llm, embed)       or       maasv-server (HTTP API)
    |                                               |
    +-- core/                                       +-- server/
    |   +-- store.py        Memory CRUD             |   +-- main.py       FastAPI app
    |   +-- retrieval.py    3-signal retrieval       |   +-- auth.py       API key auth
    |   +-- graph.py        Knowledge graph          |   +-- config.py     Env-based config
    |   +-- wisdom.py       Experiential learning    |   +-- providers.py  LLM/embed factories
    |   +-- db.py           SQLite + sqlite-vec      |   +-- routers/      HTTP endpoints
    |   +-- reranker.py     Cross-encoder            |
    |   +-- learned_ranker.py  Neural reranker (exp) |
    |   +-- autograd.py     Backprop engine          |
    |                                               |
    +-- extraction/                                 (server imports core/ directly)
    |   +-- entity_extraction.py
    |
    +-- lifecycle/
    |   +-- worker.py         Background jobs
    |   +-- memory_hygiene.py Dedup, prune, consolidate
    |   +-- reorganize.py     Graph optimization
    |   +-- inference.py      Entity resolution
    |   +-- review.py         Conversation analysis
    |   +-- learn.py          Ranker training (exp)
    |
    +-- providers/
        +-- ollama.py         Built-in Ollama embeddings
```

Everything talks to one SQLite database. No Redis, no Postgres, no external services. The entire state of an agent's memory is a single `.db` file you can copy, back up, or throw away.

## Security Hardening

maasv stores all data in a local SQLite database. There are no external network calls from the engine itself — your data stays on your machine. The following hardening measures are implemented across the codebase.

### LLM Output Validation

LLM responses are untrusted input. Every write path from extraction, review, and inference pipelines enforces:

- **Entity name sanitization.** Character allowlist (alphanumeric, spaces, hyphens, underscores, periods) applied at write time. Rejects names with shell metacharacters, control characters, or injection attempts.
- **Cardinality caps.** Max 20 entities, 30 relationships, 20 insights, and 20 inferences per extraction call. Prevents a single hallucinated response from flooding the graph.
- **Field length caps.** Entity names capped at 200 chars, relationship values at 2K, content at 10K on all extraction paths.
- **Confidence clamping.** All confidence values clamped to `[0.0, 1.0]` on every write path — `store_memory`, `add_relationship`, and all extraction pipelines. Non-numeric values coerced to 0.5.
- **Predicate allowlist.** Only predicates defined in `PREDICATE_OBJECT_TYPE` are accepted by `add_relationship()`. Unknown predicates from LLM output are rejected.

### Atomic Operations

- **`supersede_memory()`** runs in a single transaction: read old memory, check for duplicates, insert new, update `superseded_by` pointer, commit. No partial state on crash.
- **`update_relationship_value()`** runs in a single transaction: find current, expire old, insert new, commit.
- **Orphan cleanup** uses atomic `DELETE ... WHERE NOT EXISTS` instead of select-then-delete.

### Database Integrity

- **SQLite pragmas.** WAL mode, `busy_timeout=5000`, and `foreign_keys=ON` applied to every connection (both `get_db()` and `get_plain_db()`).
- **Dedup constraints.** Database-level unique indexes on entities (`canonical_name`, `entity_type`) and partial unique indexes on active relationships. `IntegrityError` handling on `create_entity()` and `add_relationship()`.
- **Safe backups.** `Connection.backup()` instead of `shutil.copy2` for WAL-safe snapshots. Backup retention handles `FileNotFoundError` for concurrent deletions.

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

## Status

This is running in production powering [Doris](https://github.com/ascottbell/doris), but the public API may shift as more people use it. The core concepts (memory, graph, retrieval, wisdom, lifecycle) are stable. The edges are still being refined.

## License

Business Source License 1.1. Free for personal, internal, educational, and non-commercial use. Commercial use requires a license. Contact admin@maasv.ai. Converts to Apache 2.0 on 2030-02-16. See [LICENSE](LICENSE) for details.

## Related

- **[Doris](https://github.com/ascottbell/doris)** The AI assistant maasv was built for. If maasv is the cognition layer, Doris is the person using it.

"""maasv adapters: full pipeline + individual signal ablations."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import urllib.request
from pathlib import Path

import maasv
from maasv.config import MaasvConfig
from maasv.core.db import _db, get_query_embedding, serialize_embedding
from maasv.core.graph import add_relationship, find_or_create_entity
from maasv.core.learned_ranker import reload_model
from maasv.core.retrieval import (
    _find_memories_by_bm25,
    _find_memories_by_graph,
    find_similar_memories,
)
from maasv.core.store import store_memory

from benchmarks.adapters.base import MemorySystemAdapter
from benchmarks.dataset.schemas import BenchmarkDataset
from benchmarks.embed.deterministic import DeterministicEmbedProvider

logger = logging.getLogger(__name__)


class _NoOpLLM:
    """Stub LLM that returns empty JSON (no entity extraction)."""

    def call(self, messages, model, max_tokens, source=""):
        return "[]"


def _setup_maasv(
    adapter: MemorySystemAdapter,
    dataset: BenchmarkDataset,
    extra_config: dict | None = None,
    embed_provider: object | None = None,
    embed_dims: int | None = None,
) -> tuple[Path, dict[int, str]]:
    """Shared setup: init maasv with a fresh temp DB, ingest dataset.

    Args:
        embed_provider: If provided, use this instead of DeterministicEmbedProvider.
        embed_dims: Embedding dimensions. Required when embed_provider is given.
                    Defaults to 64 for deterministic provider.

    Returns (tmp_dir, index_to_id mapping).
    """
    dims = embed_dims if embed_dims is not None else 64
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"maasv_bench_{adapter.name}_"))
    db_path = tmp_dir / "bench.db"

    config_kwargs = dict(
        db_path=db_path,
        embed_dims=dims,
        embed_model="benchmark-deterministic" if embed_provider is None else "benchmark-ollama",
        cross_encoder_enabled=False,
        learned_ranker_enabled=False,
    )
    if extra_config:
        config_kwargs.update(extra_config)

    config = MaasvConfig(**config_kwargs)

    if embed_provider is None:
        embed_provider = DeterministicEmbedProvider(
            dims=64,
            cluster_keywords=dataset.cluster_keywords,
            seed=dataset.seed,
        )

    # Fresh init — clears singleton state
    maasv.init(config=config, llm=_NoOpLLM(), embed=embed_provider)
    reload_model()

    # Ingest memories
    index_to_id: dict[int, str] = {}
    for i, mem in enumerate(dataset.memories):
        mid = store_memory(
            content=mem.content,
            category=mem.category,
            subject=mem.subject,
            source="benchmark",
            confidence=1.0,
            metadata=mem.metadata,
            dedup_threshold=0.0,  # No dedup — store everything
        )
        index_to_id[i] = mid

    # Create entities and build name→id map
    entity_ids: dict[str, str] = {}
    for ent in dataset.entities:
        eid = find_or_create_entity(
            name=ent.name,
            entity_type=ent.entity_type,
        )
        entity_ids[ent.name] = eid

    # Create relationships
    for rel in dataset.relationships:
        subj_id = entity_ids.get(rel.subject_name)
        obj_id = entity_ids.get(rel.object_name)
        if subj_id and obj_id:
            add_relationship(
                subject_id=subj_id,
                predicate=rel.predicate,
                object_id=obj_id,
                source="benchmark",
            )

    return tmp_dir, index_to_id


class MaasvFullAdapter(MemorySystemAdapter):
    """Full 3-signal pipeline: vector + BM25 + graph with RRF fusion."""

    name = "maasv-full"

    def __init__(self):
        self._tmp_dir: Path | None = None
        self._index_to_id: dict[int, str] = {}

    def setup(self, dataset: BenchmarkDataset, config: dict | None = None) -> None:
        self._tmp_dir, self._index_to_id = _setup_maasv(self, dataset, config)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        return find_similar_memories(query, limit=limit)

    def teardown(self) -> None:
        if self._tmp_dir and self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def get_memory_id_for_index(self, dataset_index: int) -> str | None:
        return self._index_to_id.get(dataset_index)


class MaasvVectorOnlyAdapter(MemorySystemAdapter):
    """Vector similarity only (no BM25, no graph)."""

    name = "maasv-vector-only"

    def __init__(self):
        self._tmp_dir: Path | None = None
        self._index_to_id: dict[int, str] = {}

    def setup(self, dataset: BenchmarkDataset, config: dict | None = None) -> None:
        self._tmp_dir, self._index_to_id = _setup_maasv(self, dataset, config)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        query_embedding = get_query_embedding(query)
        with _db() as db:
            rows = db.execute(
                """
                SELECT
                    v.id, m.content, m.category, m.subject, m.confidence,
                    m.created_at, m.metadata, m.importance, m.access_count,
                    m.surfacing_count, m.origin, m.origin_interface,
                    distance
                FROM memory_vectors v
                JOIN memories m ON v.id = m.id
                WHERE m.superseded_by IS NULL
                AND v.embedding MATCH ?
                AND k = ?
                ORDER BY distance
                """,
                (serialize_embedding(query_embedding), limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def teardown(self) -> None:
        if self._tmp_dir and self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def get_memory_id_for_index(self, dataset_index: int) -> str | None:
        return self._index_to_id.get(dataset_index)


class MaasvBM25OnlyAdapter(MemorySystemAdapter):
    """BM25 keyword matching only."""

    name = "maasv-bm25-only"

    def __init__(self):
        self._tmp_dir: Path | None = None
        self._index_to_id: dict[int, str] = {}

    def setup(self, dataset: BenchmarkDataset, config: dict | None = None) -> None:
        self._tmp_dir, self._index_to_id = _setup_maasv(self, dataset, config)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        with _db() as db:
            results = _find_memories_by_bm25(db, query, limit=limit)
        return results

    def teardown(self) -> None:
        if self._tmp_dir and self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def get_memory_id_for_index(self, dataset_index: int) -> str | None:
        return self._index_to_id.get(dataset_index)


class MaasvGraphOnlyAdapter(MemorySystemAdapter):
    """Graph traversal only."""

    name = "maasv-graph-only"

    def __init__(self):
        self._tmp_dir: Path | None = None
        self._index_to_id: dict[int, str] = {}

    def setup(self, dataset: BenchmarkDataset, config: dict | None = None) -> None:
        self._tmp_dir, self._index_to_id = _setup_maasv(self, dataset, config)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        with _db() as db:
            results = _find_memories_by_graph(db, query, limit=limit)
        return results

    def teardown(self) -> None:
        if self._tmp_dir and self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def get_memory_id_for_index(self, dataset_index: int) -> str | None:
        return self._index_to_id.get(dataset_index)


# ---------------------------------------------------------------------------
# Ollama adapter — real model embeddings
# ---------------------------------------------------------------------------

_OLLAMA_MODEL = "qwen3-embedding:8b"
_OLLAMA_DIMS = 4096
_OLLAMA_BASE_URL = "http://localhost:11434"


class MaasvOllamaFullAdapter(MemorySystemAdapter):
    """Full 3-signal pipeline with real Ollama embeddings (qwen3-embedding:8b)."""

    name = "maasv-ollama-full"

    def __init__(self):
        self._tmp_dir: Path | None = None
        self._index_to_id: dict[int, str] = {}

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        """Check if Ollama is running and the model is pulled.

        Returns (available, message).
        """
        try:
            req = urllib.request.Request(
                f"{_OLLAMA_BASE_URL}/api/tags",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
        except Exception:
            return False, "Ollama is not running at localhost:11434"

        model_names = [m.get("name", "") for m in data.get("models", [])]
        # Ollama model names may include :latest suffix
        if not any(_OLLAMA_MODEL in name for name in model_names):
            return False, (
                f"Model {_OLLAMA_MODEL} not found. "
                f"Pull it with: ollama pull {_OLLAMA_MODEL}"
            )

        return True, "ok"

    def setup(self, dataset: BenchmarkDataset, config: dict | None = None) -> None:
        from maasv.providers.ollama import OllamaEmbed

        ollama_embed = OllamaEmbed(
            model=_OLLAMA_MODEL,
            base_url=_OLLAMA_BASE_URL,
            dims=_OLLAMA_DIMS,
        )
        self._tmp_dir, self._index_to_id = _setup_maasv(
            self,
            dataset,
            config,
            embed_provider=ollama_embed,
            embed_dims=_OLLAMA_DIMS,
        )

    def search(self, query: str, limit: int = 5) -> list[dict]:
        return find_similar_memories(query, limit=limit)

    def teardown(self) -> None:
        if self._tmp_dir and self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def get_memory_id_for_index(self, dataset_index: int) -> str | None:
        return self._index_to_id.get(dataset_index)

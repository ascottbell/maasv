"""
maasv quickstart: store memories, build a knowledge graph, retrieve with context.

This example uses mock providers so you can run it without any API keys or
embedding models. In production, you'd swap these for real providers
(see the EmbedProvider and LLMProvider protocols in maasv/protocols.py).

    python examples/quickstart.py
"""

import hashlib
import tempfile
from pathlib import Path

import maasv
from maasv.config import MaasvConfig


# -- Step 0: Implement the two provider protocols ---------------------------
# maasv doesn't bundle an LLM or embedding model. You bring your own.
# These mocks let you run the example without any external dependencies.

class LocalEmbedProvider:
    """Hash-based embeddings for demo purposes. Not useful for real retrieval."""

    def __init__(self, dims: int = 64):
        self.dims = dims

    def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        vec = [b / 255.0 for b in h]
        while len(vec) < self.dims:
            vec.extend(vec)
        return vec[: self.dims]

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)


class LocalLLMProvider:
    """Stub LLM that returns empty JSON. Entity extraction won't produce
    results with this, but everything else works fine."""

    def call(self, messages, model, max_tokens, source=""):
        return "[]"


# -- Step 1: Initialize maasv ----------------------------------------------

with tempfile.TemporaryDirectory() as tmp:
    db_path = Path(tmp) / "demo.db"

    config = MaasvConfig(
        db_path=db_path,
        embed_dims=64,
        cross_encoder_enabled=False,
    )

    maasv.init(
        config=config,
        llm=LocalLLMProvider(),
        embed=LocalEmbedProvider(dims=64),
    )

    # -- Step 2: Store some memories ----------------------------------------

    from maasv.core.store import store_memory

    store_memory("Morgan is the tech lead on the Atlas project", category="project", subject="Morgan")
    store_memory("The team prefers Rust for backend services", category="preference", subject="Tech stack")
    store_memory("Atlas is a geospatial mapping tool for field researchers", category="project", subject="Atlas")
    store_memory("I prefer Python over JavaScript", category="preference")
    store_memory("Started building Atlas in March 2024", category="project", subject="Atlas")
    store_memory("Atlas uses a vector store for fast retrieval", category="project", subject="Atlas")

    print("Stored 6 memories.\n")

    # -- Step 3: Build the knowledge graph ----------------------------------

    from maasv.core.graph import find_or_create_entity, add_relationship

    morgan = find_or_create_entity("Morgan", "person")
    taylor = find_or_create_entity("Taylor", "person")
    atlas = find_or_create_entity("Atlas", "project")
    rust = find_or_create_entity("Rust", "technology")
    portland = find_or_create_entity("Portland", "place")

    add_relationship(morgan, "works_with", object_id=taylor)
    add_relationship(morgan, "works_on", object_id=atlas)
    add_relationship(morgan, "lives_in", object_id=portland)
    add_relationship(atlas, "uses_tech", object_id=rust)

    print("Built knowledge graph: 5 entities, 4 relationships.\n")

    # -- Step 4: Retrieve with 3-signal fusion ------------------------------

    from maasv.core.retrieval import find_similar_memories

    results = find_similar_memories("Tell me about Atlas", limit=3)
    print("Query: 'Tell me about Atlas'")
    for mem in results:
        print(f"  [{mem['category']}] {mem['content']}")

    print()

    # -- Step 5: Tiered context (what you'd inject into an LLM prompt) ------

    from maasv.core.retrieval import get_tiered_memory_context

    context = get_tiered_memory_context(query="Atlas project status")
    print("Tiered context for 'Atlas project status':")
    print(context)
    print()

    # -- Step 6: Log a decision to the wisdom system ------------------------

    from maasv.core.wisdom import log_reasoning, record_outcome, add_feedback

    wisdom_id = log_reasoning(
        action_type="framework_recommendation",
        reasoning="Team needed async support and low memory footprint. "
                  "Chose Tokio because Atlas processes large geospatial datasets.",
        context="Backend framework decision, Atlas project",
    )

    record_outcome(wisdom_id, "success", "Handles 10K concurrent connections without issues")
    add_feedback(wisdom_id, score=5, notes="Great choice for our workload")

    print("Logged wisdom entry with outcome and feedback.")
    print("\nDone. In production, the sleep worker would now run entity extraction,")
    print("inference, review, and hygiene jobs in the background.")

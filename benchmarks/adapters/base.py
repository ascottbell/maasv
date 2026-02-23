"""Base adapter interface for memory system benchmarks."""

from __future__ import annotations

from abc import ABC, abstractmethod

from benchmarks.dataset.schemas import BenchmarkDataset


class MemorySystemAdapter(ABC):
    """Interface that all memory system adapters must implement."""

    name: str

    @abstractmethod
    def setup(self, dataset: BenchmarkDataset, config: dict | None = None) -> None:
        """Ingest dataset into the memory system.

        Must populate the system with memories, entities, and relationships
        from the dataset. Must build an index mapping dataset memory indices
        to system-internal IDs (for judgment evaluation).
        """

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Search the memory system. Each result must have an 'id' key."""

    @abstractmethod
    def teardown(self) -> None:
        """Clean up resources (temp dirs, DB connections, etc.)."""

    def get_memory_id_for_index(self, dataset_index: int) -> str | None:
        """Map a dataset memory index to the system's internal ID.

        Must be callable after setup(). Returns None if the index wasn't stored.
        """
        return self._index_to_id.get(dataset_index)

    @classmethod
    def is_available(cls) -> bool:
        """Whether this adapter's dependencies are installed."""
        return True

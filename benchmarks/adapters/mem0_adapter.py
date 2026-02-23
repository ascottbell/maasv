"""Mem0 adapter stub — not yet implemented."""

from __future__ import annotations

from benchmarks.adapters.base import MemorySystemAdapter
from benchmarks.dataset.schemas import BenchmarkDataset


class Mem0Adapter(MemorySystemAdapter):
    name = "mem0"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import mem0  # noqa: F401
            return True
        except ImportError:
            return False

    def setup(self, dataset: BenchmarkDataset, config: dict | None = None) -> None:
        raise NotImplementedError(
            "Mem0 adapter not yet implemented. Install: pip install mem0ai"
        )

    def search(self, query: str, limit: int = 5) -> list[dict]:
        raise NotImplementedError

    def teardown(self) -> None:
        pass

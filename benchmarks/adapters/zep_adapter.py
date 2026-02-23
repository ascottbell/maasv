"""Zep adapter stub — not yet implemented."""

from __future__ import annotations

from benchmarks.adapters.base import MemorySystemAdapter
from benchmarks.dataset.schemas import BenchmarkDataset


class ZepAdapter(MemorySystemAdapter):
    name = "zep"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import zep_cloud  # noqa: F401
            return True
        except ImportError:
            return False

    def setup(self, dataset: BenchmarkDataset, config: dict | None = None) -> None:
        raise NotImplementedError(
            "Zep adapter not yet implemented. Install: pip install zep-cloud"
        )

    def search(self, query: str, limit: int = 5) -> list[dict]:
        raise NotImplementedError

    def teardown(self) -> None:
        pass

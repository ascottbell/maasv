"""Data schemas for benchmark datasets."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Memory:
    content: str
    category: str
    subject: str | None = None
    importance: float = 0.5
    created_days_ago: int = 0
    metadata: dict | None = None


@dataclass
class Entity:
    name: str
    entity_type: str


@dataclass
class Relationship:
    subject_name: str
    predicate: str
    object_name: str


@dataclass
class QueryJudgment:
    """A query with ground-truth relevance judgments.

    relevant_memory_indices: indices into BenchmarkDataset.memories
    relevance_grades: parallel list, 1.0 = exact answer, 0.5 = topically related
    """

    query: str
    relevant_memory_indices: list[int]
    relevance_grades: list[float]


@dataclass
class BenchmarkDataset:
    memories: list[Memory]
    entities: list[Entity]
    relationships: list[Relationship]
    judgments: list[QueryJudgment]
    cluster_keywords: dict[str, list[str]]
    seed: int = 42

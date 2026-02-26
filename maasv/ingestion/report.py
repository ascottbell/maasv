"""Ingestion statistics tracking."""

from dataclasses import dataclass, field


@dataclass
class ChunkError:
    """Record of a failed chunk processing attempt."""

    chunk_index: int
    source_file: str
    error: str


@dataclass
class IngestReport:
    """Tracks statistics for an ingestion run."""

    files_scanned: int = 0
    files_skipped: int = 0
    total_chunks: int = 0
    memories_stored: int = 0
    memories_deduped: int = 0
    entities_created: int = 0
    entities_skipped: int = 0
    relationships_created: int = 0
    chunk_errors: list[ChunkError] = field(default_factory=list)

    def merge(self, other: "IngestReport") -> None:
        """Merge another report into this one (for directory aggregation)."""
        self.files_scanned += other.files_scanned
        self.files_skipped += other.files_skipped
        self.total_chunks += other.total_chunks
        self.memories_stored += other.memories_stored
        self.memories_deduped += other.memories_deduped
        self.entities_created += other.entities_created
        self.entities_skipped += other.entities_skipped
        self.relationships_created += other.relationships_created
        self.chunk_errors.extend(other.chunk_errors)

    def summary(self) -> str:
        """Human-readable summary for CLI output."""
        lines = [
            f"Files scanned:        {self.files_scanned}",
            f"Files skipped:        {self.files_skipped}",
            f"Total chunks:         {self.total_chunks}",
            f"Memories stored:      {self.memories_stored}",
            f"Memories deduped:     {self.memories_deduped}",
            f"Entities created:     {self.entities_created}",
            f"Entities skipped:     {self.entities_skipped}",
            f"Relationships created:{self.relationships_created}",
        ]
        if self.chunk_errors:
            lines.append(f"Chunk errors:         {len(self.chunk_errors)}")
            for err in self.chunk_errors[:5]:
                lines.append(f"  - chunk {err.chunk_index} ({err.source_file}): {err.error}")
            if len(self.chunk_errors) > 5:
                lines.append(f"  ... and {len(self.chunk_errors) - 5} more")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serializable dict for REST responses."""
        return {
            "files_scanned": self.files_scanned,
            "files_skipped": self.files_skipped,
            "total_chunks": self.total_chunks,
            "memories_stored": self.memories_stored,
            "memories_deduped": self.memories_deduped,
            "entities_created": self.entities_created,
            "entities_skipped": self.entities_skipped,
            "relationships_created": self.relationships_created,
            "chunk_errors": [
                {"chunk_index": e.chunk_index, "source_file": e.source_file, "error": e.error}
                for e in self.chunk_errors
            ],
        }

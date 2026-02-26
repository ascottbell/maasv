"""Core ingestion orchestration: text → chunks → memories + entities."""

import json
import logging
from pathlib import Path
from typing import Callable, Optional

from maasv.ingestion.chunking import Chunk, chunk_file, chunk_text as _chunk_text
from maasv.ingestion.formats import is_supported
from maasv.ingestion.report import ChunkError, IngestReport

logger = logging.getLogger(__name__)


def _process_chunk(
    chunk: Chunk,
    category: str,
    source: str,
    origin: Optional[str],
    origin_interface: Optional[str],
    report: IngestReport,
    dry_run: bool,
    run_extraction: bool,
) -> None:
    """Process a single chunk: store memory, optionally extract entities.

    Updates report in-place. Never raises.
    """
    report.total_chunks += 1

    if dry_run:
        return

    # Build provenance metadata
    metadata = {
        "source_file": chunk.source_file,
        "format": chunk.format.value,
        "chunk_index": chunk.index,
    }
    if chunk.metadata:
        metadata.update(chunk.metadata)

    # Store memory
    try:
        from maasv.core.store import store_memory

        memory_id = store_memory(
            content=chunk.content,
            category=category,
            source=source,
            metadata=metadata,
            origin=origin,
            origin_interface=origin_interface,
        )

        # store_memory returns existing ID on dedup — check if it's a new store
        # by looking at whether the ID was just created (starts with mem_)
        # Dedup detection: if store_memory found a duplicate, it logged it.
        # We count it as stored either way, but track dedup via the store_memory
        # internal logging. For accurate counting, we check if it was a new insert
        # by querying created_at, but that's expensive. Instead, just count all as stored.
        report.memories_stored += 1

    except Exception as e:
        report.chunk_errors.append(
            ChunkError(
                chunk_index=chunk.index,
                source_file=chunk.source_file,
                error=str(e),
            )
        )
        logger.error("Failed to store chunk %d from %s: %s", chunk.index, chunk.source_file, e)
        return  # Skip extraction if storage failed

    # Extract entities
    if run_extraction:
        try:
            from maasv.extraction.entity_extraction import extract_and_store_entities

            # Use heading metadata as topic if available
            topic = chunk.metadata.get("heading", "")
            if topic.startswith("#"):
                topic = topic.lstrip("#").strip()

            result = extract_and_store_entities(summary=chunk.content, topic=topic)

            storage = result.get("storage", {})
            report.entities_created += storage.get("entities_created", 0)
            report.entities_skipped += storage.get("entities_skipped", 0)
            report.relationships_created += storage.get("relationships_created", 0)

        except Exception as e:
            # Memory already stored — extraction failure is non-fatal
            logger.warning(
                "Entity extraction failed for chunk %d from %s: %s",
                chunk.index,
                chunk.source_file,
                e,
            )


def ingest_text(
    text: str,
    category: str = "imported",
    source: str = "ingestion",
    origin: Optional[str] = None,
    origin_interface: Optional[str] = None,
    dry_run: bool = False,
    run_extraction: bool = True,
    chunk_size: int = 3000,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> IngestReport:
    """Ingest raw text: chunk and store as memories.

    Args:
        text: Raw text content to ingest.
        category: Memory category for all chunks.
        source: Source label (e.g., "chatgpt-export", "notes").
        origin: System origin (e.g., "chatgpt", "notion").
        origin_interface: Specific interface (e.g., "api", "cli").
        dry_run: If True, parse and chunk but don't store anything.
        run_extraction: If True, run LLM entity extraction on each chunk.
        chunk_size: Target chunk size in characters.
        progress_callback: Called with (chunk_index, status_message) for each chunk.

    Returns:
        IngestReport with statistics.
    """
    report = IngestReport(files_scanned=1)

    for chunk in _chunk_text(text, source_file="<text>", chunk_size=chunk_size):
        if progress_callback:
            progress_callback(chunk.index, f"Processing chunk {chunk.index}")

        _process_chunk(
            chunk=chunk,
            category=category,
            source=source,
            origin=origin,
            origin_interface=origin_interface,
            report=report,
            dry_run=dry_run,
            run_extraction=run_extraction,
        )

    return report


def ingest_file(
    path: Path,
    category: str = "imported",
    source: Optional[str] = None,
    origin: Optional[str] = None,
    origin_interface: Optional[str] = None,
    dry_run: bool = False,
    run_extraction: bool = True,
    chunk_size: int = 3000,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> IngestReport:
    """Ingest a single file: detect format, chunk, store memories.

    Args:
        path: Path to the file.
        category: Memory category for all chunks.
        source: Source label (defaults to filename).
        origin: System origin.
        origin_interface: Specific interface.
        dry_run: Parse and chunk without storing.
        run_extraction: Run LLM entity extraction.
        chunk_size: Target chunk size in characters.
        progress_callback: Called with (chunk_index, status_message).

    Returns:
        IngestReport with statistics.
    """
    path = Path(path).resolve()
    report = IngestReport(files_scanned=1)

    if source is None:
        source = path.name

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    if not is_supported(path):
        raise ValueError(f"Unsupported file format: {path.suffix}")

    try:
        for chunk in chunk_file(path, chunk_size=chunk_size):
            if progress_callback:
                progress_callback(chunk.index, f"Processing {path.name} chunk {chunk.index}")

            _process_chunk(
                chunk=chunk,
                category=category,
                source=source,
                origin=origin,
                origin_interface=origin_interface,
                report=report,
                dry_run=dry_run,
                run_extraction=run_extraction,
            )
    except UnicodeDecodeError:
        logger.warning("Skipping non-UTF8 file: %s", path)
        report.files_skipped += 1
    except (ValueError, json.JSONDecodeError) as e:
        logger.warning("Skipping file %s: %s", path, e)
        report.files_skipped += 1

    return report


def ingest_directory(
    path: Path,
    category: str = "imported",
    origin: Optional[str] = None,
    origin_interface: Optional[str] = None,
    recursive: bool = True,
    dry_run: bool = False,
    run_extraction: bool = True,
    chunk_size: int = 3000,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> IngestReport:
    """Ingest all supported files in a directory.

    Args:
        path: Directory path.
        category: Memory category for all chunks.
        origin: System origin.
        origin_interface: Specific interface.
        recursive: Walk subdirectories.
        dry_run: Parse and chunk without storing.
        run_extraction: Run LLM entity extraction.
        chunk_size: Target chunk size in characters.
        progress_callback: Called with (file_index, status_message).

    Returns:
        Merged IngestReport with statistics from all files.
    """
    path = Path(path).resolve()
    report = IngestReport()

    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    # Collect supported files
    if recursive:
        files = sorted(f for f in path.rglob("*") if f.is_file() and is_supported(f))
    else:
        files = sorted(f for f in path.iterdir() if f.is_file() and is_supported(f))

    for i, file_path in enumerate(files):
        if progress_callback:
            progress_callback(i, f"[{i + 1}/{len(files)}] {file_path.name}")

        file_report = ingest_file(
            path=file_path,
            category=category,
            source=file_path.name,
            origin=origin,
            origin_interface=origin_interface,
            dry_run=dry_run,
            run_extraction=run_extraction,
            chunk_size=chunk_size,
        )
        report.merge(file_report)

    return report

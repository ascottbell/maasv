"""Format-specific chunking strategies for ingestion.

All chunkers are generators yielding Chunk instances — one chunk live at a time,
no OOM risk regardless of file size.
"""

import csv
import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

from maasv.ingestion.formats import FileFormat, detect_format

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 3000  # chars
MIN_CHUNK_SIZE = 50  # extract_from_summary minimum


@dataclass
class Chunk:
    """A single chunk of content ready for memory storage."""

    content: str
    index: int
    source_file: str
    format: FileFormat
    metadata: dict = field(default_factory=dict)


# --- Text chunking ---


def chunk_text(
    text: str,
    source_file: str = "<text>",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Generator[Chunk, None, None]:
    """Split plain text on paragraph boundaries (double newlines)."""
    paragraphs = text.split("\n\n")
    buffer = ""
    index = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        candidate = f"{buffer}\n\n{para}" if buffer else para

        if len(candidate) > chunk_size and buffer:
            if len(buffer.strip()) >= MIN_CHUNK_SIZE:
                yield Chunk(
                    content=buffer.strip(),
                    index=index,
                    source_file=source_file,
                    format=FileFormat.PLAIN_TEXT,
                )
                index += 1
                buffer = para
            else:
                # Buffer too small to yield alone — keep accumulating
                buffer = candidate
        else:
            buffer = candidate

    if buffer.strip() and len(buffer.strip()) >= MIN_CHUNK_SIZE:
        yield Chunk(
            content=buffer.strip(),
            index=index,
            source_file=source_file,
            format=FileFormat.PLAIN_TEXT,
        )


# --- Markdown chunking ---


def chunk_markdown(
    text: str,
    source_file: str = "<markdown>",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Generator[Chunk, None, None]:
    """Split markdown on heading boundaries, merging small sections."""
    lines = text.split("\n")
    sections: list[tuple[str, str]] = []  # (heading, body)
    current_heading = ""
    current_body: list[str] = []

    for line in lines:
        if line.startswith("#"):
            if current_body or current_heading:
                sections.append((current_heading, "\n".join(current_body).strip()))
            current_heading = line
            current_body = []
        else:
            current_body.append(line)

    if current_body or current_heading:
        sections.append((current_heading, "\n".join(current_body).strip()))

    buffer_heading = ""
    buffer_body = ""
    index = 0

    for heading, body in sections:
        section_text = f"{heading}\n{body}".strip() if heading else body

        if not section_text:
            continue

        candidate = f"{buffer_body}\n\n{section_text}" if buffer_body else section_text

        if len(candidate) > chunk_size and buffer_body:
            if len(buffer_body.strip()) >= MIN_CHUNK_SIZE:
                yield Chunk(
                    content=buffer_body.strip(),
                    index=index,
                    source_file=source_file,
                    format=FileFormat.MARKDOWN,
                    metadata={"heading": buffer_heading} if buffer_heading else {},
                )
                index += 1
                buffer_heading = heading
                buffer_body = section_text
            else:
                # Buffer too small to yield alone — keep accumulating
                if not buffer_heading and heading:
                    buffer_heading = heading
                buffer_body = candidate
        else:
            if not buffer_heading and heading:
                buffer_heading = heading
            buffer_body = candidate

    if buffer_body.strip() and len(buffer_body.strip()) >= MIN_CHUNK_SIZE:
        yield Chunk(
            content=buffer_body.strip(),
            index=index,
            source_file=source_file,
            format=FileFormat.MARKDOWN,
            metadata={"heading": buffer_heading} if buffer_heading else {},
        )


# --- CSV chunking ---

CSV_ROWS_PER_CHUNK = 20


def chunk_csv(
    text: str,
    source_file: str = "<csv>",
    delimiter: Optional[str] = None,
) -> Generator[Chunk, None, None]:
    """Chunk CSV/TSV into groups of rows, rendered as 'Field: value' text."""
    if delimiter is None:
        delimiter = "\t" if source_file.endswith((".tsv", ".TSV")) else ","

    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if not reader.fieldnames:
        return

    buffer_rows: list[str] = []
    index = 0
    row_count = 0

    for row in reader:
        entry = "\n".join(f"{k}: {v}" for k, v in row.items() if v)
        buffer_rows.append(entry)
        row_count += 1

        if row_count >= CSV_ROWS_PER_CHUNK:
            content = "\n---\n".join(buffer_rows)
            if len(content) >= MIN_CHUNK_SIZE:
                yield Chunk(
                    content=content,
                    index=index,
                    source_file=source_file,
                    format=FileFormat.CSV,
                    metadata={"rows": row_count, "fields": list(reader.fieldnames)},
                )
                index += 1
            buffer_rows = []
            row_count = 0

    if buffer_rows:
        content = "\n---\n".join(buffer_rows)
        if len(content) >= MIN_CHUNK_SIZE:
            yield Chunk(
                content=content,
                index=index,
                source_file=source_file,
                format=FileFormat.CSV,
                metadata={"rows": row_count, "fields": list(reader.fieldnames)},
            )


# --- JSON chunking ---


def chunk_json(
    text: str,
    source_file: str = "<json>",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Generator[Chunk, None, None]:
    """Chunk JSON: arrays batch by element, objects by top-level key, JSONL auto-detected."""
    text = text.strip()

    # JSONL detection: multiple JSON objects/values separated by newlines
    if not text.startswith(("{", "[")):
        yield from _chunk_jsonl(text, source_file, chunk_size)
        return

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try JSONL as fallback
        yield from _chunk_jsonl(text, source_file, chunk_size)
        return

    if isinstance(data, list):
        yield from _chunk_json_array(data, source_file, chunk_size)
    elif isinstance(data, dict):
        yield from _chunk_json_object(data, source_file, chunk_size)
    else:
        # Scalar — single chunk
        content = json.dumps(data, indent=2, ensure_ascii=False)
        if len(content) >= MIN_CHUNK_SIZE:
            yield Chunk(
                content=content,
                index=0,
                source_file=source_file,
                format=FileFormat.JSON,
            )


def _chunk_jsonl(
    text: str, source_file: str, chunk_size: int
) -> Generator[Chunk, None, None]:
    """Chunk JSONL (newline-delimited JSON)."""
    buffer: list[str] = []
    buffer_len = 0
    index = 0

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
            rendered = json.dumps(obj, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            rendered = line

        if buffer_len + len(rendered) > chunk_size and buffer:
            content = "\n\n".join(buffer)
            if len(content) >= MIN_CHUNK_SIZE:
                yield Chunk(
                    content=content,
                    index=index,
                    source_file=source_file,
                    format=FileFormat.JSON,
                    metadata={"json_type": "jsonl"},
                )
                index += 1
            buffer = []
            buffer_len = 0

        buffer.append(rendered)
        buffer_len += len(rendered)

    if buffer:
        content = "\n\n".join(buffer)
        if len(content) >= MIN_CHUNK_SIZE:
            yield Chunk(
                content=content,
                index=index,
                source_file=source_file,
                format=FileFormat.JSON,
                metadata={"json_type": "jsonl"},
            )


def _chunk_json_array(
    data: list, source_file: str, chunk_size: int
) -> Generator[Chunk, None, None]:
    """Chunk a JSON array by batching elements."""
    buffer: list[str] = []
    buffer_len = 0
    index = 0

    for item in data:
        rendered = json.dumps(item, indent=2, ensure_ascii=False)

        if buffer_len + len(rendered) > chunk_size and buffer:
            content = "\n\n".join(buffer)
            if len(content) >= MIN_CHUNK_SIZE:
                yield Chunk(
                    content=content,
                    index=index,
                    source_file=source_file,
                    format=FileFormat.JSON,
                    metadata={"json_type": "array"},
                )
                index += 1
            buffer = []
            buffer_len = 0

        buffer.append(rendered)
        buffer_len += len(rendered)

    if buffer:
        content = "\n\n".join(buffer)
        if len(content) >= MIN_CHUNK_SIZE:
            yield Chunk(
                content=content,
                index=index,
                source_file=source_file,
                format=FileFormat.JSON,
                metadata={"json_type": "array"},
            )


def _chunk_json_object(
    data: dict, source_file: str, chunk_size: int
) -> Generator[Chunk, None, None]:
    """Chunk a JSON object by top-level keys."""
    buffer: list[str] = []
    buffer_len = 0
    index = 0

    for key, value in data.items():
        rendered = json.dumps({key: value}, indent=2, ensure_ascii=False)

        if buffer_len + len(rendered) > chunk_size and buffer:
            content = "\n\n".join(buffer)
            if len(content) >= MIN_CHUNK_SIZE:
                yield Chunk(
                    content=content,
                    index=index,
                    source_file=source_file,
                    format=FileFormat.JSON,
                    metadata={"json_type": "object"},
                )
                index += 1
            buffer = []
            buffer_len = 0

        buffer.append(rendered)
        buffer_len += len(rendered)

    if buffer:
        content = "\n\n".join(buffer)
        if len(content) >= MIN_CHUNK_SIZE:
            yield Chunk(
                content=content,
                index=index,
                source_file=source_file,
                format=FileFormat.JSON,
                metadata={"json_type": "object"},
            )


# --- PDF chunking ---


def chunk_pdf(
    path: Path,
    source_file: str = "<pdf>",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Generator[Chunk, None, None]:
    """Chunk PDF by pages, merging small pages up to chunk_size."""
    try:
        import pymupdf
    except ImportError:
        raise ImportError("pymupdf is required for PDF ingestion: pip install pymupdf")

    doc = pymupdf.open(str(path))
    buffer = ""
    buffer_start_page = 1
    index = 0

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text().strip()

            if not page_text:
                continue

            candidate = f"{buffer}\n\n{page_text}" if buffer else page_text

            if len(candidate) > chunk_size and buffer:
                if len(buffer.strip()) >= MIN_CHUNK_SIZE:
                    yield Chunk(
                        content=buffer.strip(),
                        index=index,
                        source_file=source_file,
                        format=FileFormat.PDF,
                        metadata={"start_page": buffer_start_page, "end_page": page_num},
                    )
                    index += 1
                buffer = page_text
                buffer_start_page = page_num + 1
            else:
                if not buffer:
                    buffer_start_page = page_num + 1
                buffer = candidate

        if buffer.strip() and len(buffer.strip()) >= MIN_CHUNK_SIZE:
            yield Chunk(
                content=buffer.strip(),
                index=index,
                source_file=source_file,
                format=FileFormat.PDF,
                metadata={"start_page": buffer_start_page, "end_page": len(doc)},
            )
    finally:
        doc.close()


# --- Dispatcher ---


def chunk_file(
    path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Generator[Chunk, None, None]:
    """Detect format and dispatch to the appropriate chunker."""
    fmt = detect_format(path)
    source_file = str(path)

    if fmt == FileFormat.UNSUPPORTED:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # PDF is binary — handled directly by pymupdf
    if fmt == FileFormat.PDF:
        yield from chunk_pdf(path, source_file=source_file, chunk_size=chunk_size)
        return

    # All other formats are text-based
    text = path.read_text(encoding="utf-8")

    if fmt == FileFormat.PLAIN_TEXT:
        yield from chunk_text(text, source_file=source_file, chunk_size=chunk_size)
    elif fmt == FileFormat.MARKDOWN:
        yield from chunk_markdown(text, source_file=source_file, chunk_size=chunk_size)
    elif fmt == FileFormat.CSV:
        yield from chunk_csv(text, source_file=source_file)
    elif fmt == FileFormat.JSON:
        yield from chunk_json(text, source_file=source_file, chunk_size=chunk_size)

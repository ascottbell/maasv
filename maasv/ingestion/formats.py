"""File format detection for ingestion."""

from enum import Enum
from pathlib import Path


class FileFormat(Enum):
    PLAIN_TEXT = "text"
    MARKDOWN = "markdown"
    CSV = "csv"
    JSON = "json"
    PDF = "pdf"
    UNSUPPORTED = "unsupported"


# Extension → format mapping
_EXTENSION_MAP: dict[str, FileFormat] = {
    ".txt": FileFormat.PLAIN_TEXT,
    ".text": FileFormat.PLAIN_TEXT,
    ".log": FileFormat.PLAIN_TEXT,
    ".md": FileFormat.MARKDOWN,
    ".markdown": FileFormat.MARKDOWN,
    ".csv": FileFormat.CSV,
    ".tsv": FileFormat.CSV,
    ".json": FileFormat.JSON,
    ".jsonl": FileFormat.JSON,
    ".pdf": FileFormat.PDF,
}


def detect_format(path: Path) -> FileFormat:
    """Detect file format from extension."""
    return _EXTENSION_MAP.get(path.suffix.lower(), FileFormat.UNSUPPORTED)


def is_supported(path: Path) -> bool:
    """Check if a file's format is supported for ingestion."""
    return detect_format(path) != FileFormat.UNSUPPORTED

"""General-purpose data ingestion for maasv."""

from maasv.ingestion.pipeline import ingest_directory, ingest_file, ingest_text
from maasv.ingestion.report import IngestReport

__all__ = ["ingest_text", "ingest_file", "ingest_directory", "IngestReport"]

"""Ingestion endpoint: accept text or file path, chunk, store memories + entities."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

router = APIRouter()
logger = logging.getLogger(__name__)


# --- Request/Response models ---


class IngestRequest(BaseModel):
    # Input — exactly one required
    text: Optional[str] = Field(None, max_length=500000, description="Raw text to ingest")
    file_path: Optional[str] = Field(None, max_length=1000, description="Path to file or directory to ingest")

    # Metadata
    category: str = Field("imported", max_length=100, description="Memory category for stored chunks")
    source: Optional[str] = Field(None, max_length=200, description="Source label (defaults to filename)")
    origin: Optional[str] = Field(None, max_length=100, description="System origin (e.g., chatgpt, notion)")
    origin_interface: Optional[str] = Field(None, max_length=100, description="Interface (e.g., api, cli)")

    # Behavior
    dry_run: bool = Field(False, description="Parse and chunk without storing")
    run_extraction: bool = Field(True, description="Run LLM entity extraction on each chunk")
    chunk_size: int = Field(3000, ge=100, le=50000, description="Target chunk size in characters")
    recursive: bool = Field(True, description="Recurse into subdirectories (if file_path is a directory)")

    @model_validator(mode="after")
    def exactly_one_input(self):
        has_text = self.text is not None and self.text.strip()
        has_path = self.file_path is not None and self.file_path.strip()
        if not has_text and not has_path:
            raise ValueError("Either 'text' or 'file_path' must be provided")
        if has_text and has_path:
            raise ValueError("Provide 'text' or 'file_path', not both")
        return self


# --- Endpoint ---


@router.post("/ingest")
def ingest(req: IngestRequest):
    """Ingest text or a file/directory into maasv memory."""
    from pathlib import Path

    if req.text:
        from maasv.ingestion.pipeline import ingest_text

        report = ingest_text(
            text=req.text,
            category=req.category,
            source=req.source or "api",
            origin=req.origin,
            origin_interface=req.origin_interface or "api",
            dry_run=req.dry_run,
            run_extraction=req.run_extraction,
            chunk_size=req.chunk_size,
        )

    else:
        path = Path(req.file_path).resolve()

        if path.is_dir():
            from maasv.ingestion.pipeline import ingest_directory

            report = ingest_directory(
                path=path,
                category=req.category,
                origin=req.origin,
                origin_interface=req.origin_interface or "api",
                recursive=req.recursive,
                dry_run=req.dry_run,
                run_extraction=req.run_extraction,
                chunk_size=req.chunk_size,
            )

        elif path.is_file():
            from maasv.ingestion.pipeline import ingest_file

            try:
                report = ingest_file(
                    path=path,
                    category=req.category,
                    source=req.source,
                    origin=req.origin,
                    origin_interface=req.origin_interface or "api",
                    dry_run=req.dry_run,
                    run_extraction=req.run_extraction,
                    chunk_size=req.chunk_size,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        else:
            raise HTTPException(status_code=404, detail=f"Path not found: {req.file_path}")

    return report.to_dict()

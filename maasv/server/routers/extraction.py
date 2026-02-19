"""Extraction endpoint: entity/relationship extraction from text."""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()
logger = logging.getLogger(__name__)


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100000, description="Text to extract entities from")
    topic: str = Field("", max_length=200, description="Optional topic hint for extraction")


@router.post("/extract")
def extract(req: ExtractRequest):
    """Extract entities and relationships from text and store them in the knowledge graph."""
    from maasv.extraction.entity_extraction import extract_and_store_entities

    try:
        result = extract_and_store_entities(summary=req.text, topic=req.topic)
    except Exception:
        logger.exception("Entity extraction failed")
        raise HTTPException(status_code=500, detail="Extraction failed")

    return result

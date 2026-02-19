"""Wisdom endpoints: experiential learning (log, outcome, feedback, search)."""

import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_METADATA_BYTES = 10_000
MAX_TAGS = 20


# --- Request models ---

class LogReasoningRequest(BaseModel):
    action_type: str = Field(..., min_length=1, max_length=200)
    reasoning: str = Field(..., min_length=1, max_length=10000)
    action_data: Optional[dict] = None
    trigger: Optional[str] = Field(None, max_length=500)
    context: Optional[str] = Field(None, max_length=10000)
    tags: Optional[list[str]] = None

    @field_validator("action_data")
    @classmethod
    def check_action_data_size(cls, v: dict | None) -> dict | None:
        if v is not None and len(json.dumps(v, default=str)) > MAX_METADATA_BYTES:
            raise ValueError(f"action_data exceeds {MAX_METADATA_BYTES} byte limit")
        return v

    @field_validator("tags")
    @classmethod
    def check_tags(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            if len(v) > MAX_TAGS:
                raise ValueError(f"tags list exceeds {MAX_TAGS} items")
            for tag in v:
                if len(tag) > 100:
                    raise ValueError("individual tag exceeds 100 characters")
        return v


class RecordOutcomeRequest(BaseModel):
    outcome: str = Field(..., min_length=1, max_length=200, description="success, failed, partial, etc.")
    details: Optional[str] = Field(None, max_length=10000)


class AddFeedbackRequest(BaseModel):
    score: int = Field(..., ge=1, le=5, description="1-5 rating")
    notes: Optional[str] = Field(None, max_length=2000)


class SearchWisdomRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    limit: int = Field(10, ge=1, le=50)


# --- Endpoints ---

@router.post("/log")
def log_reasoning(req: LogReasoningRequest):
    """Log reasoning before taking an action. Returns wisdom ID for later feedback."""
    from maasv.core.wisdom import log_reasoning

    wisdom_id = log_reasoning(
        action_type=req.action_type,
        reasoning=req.reasoning,
        action_data=req.action_data,
        trigger=req.trigger,
        context=req.context,
        tags=req.tags,
    )
    return {"wisdom_id": wisdom_id}


@router.post("/{wisdom_id}/outcome")
def record_outcome(wisdom_id: str, req: RecordOutcomeRequest):
    """Record the outcome of a previously logged action."""
    from maasv.core.wisdom import record_outcome

    updated = record_outcome(
        wisdom_id=wisdom_id,
        outcome=req.outcome,
        details=req.details,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Wisdom {wisdom_id} not found")

    return {"updated": True, "wisdom_id": wisdom_id}


@router.post("/{wisdom_id}/feedback")
def add_feedback(wisdom_id: str, req: AddFeedbackRequest):
    """Attach feedback (1-5 score) to a wisdom entry."""
    from maasv.core.wisdom import add_feedback

    try:
        updated = add_feedback(
            wisdom_id=wisdom_id,
            score=req.score,
            notes=req.notes,
        )
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid feedback parameters")

    if not updated:
        raise HTTPException(status_code=404, detail=f"Wisdom {wisdom_id} not found")

    return {"updated": True, "wisdom_id": wisdom_id}


@router.post("/search")
def search_wisdom(req: SearchWisdomRequest):
    """Full-text search across reasoning, context, and feedback."""
    from maasv.core.wisdom import search_wisdom

    results = search_wisdom(query=req.query, limit=req.limit)
    return {"results": results, "count": len(results)}

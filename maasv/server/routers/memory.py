"""Memory endpoints: store, search, context, get, delete, supersede."""

import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_METADATA_BYTES = 10_000  # 10KB cap on serialized metadata


def _check_metadata_size(v: dict | None) -> dict | None:
    if v is not None and len(json.dumps(v, default=str)) > MAX_METADATA_BYTES:
        raise ValueError(f"metadata exceeds {MAX_METADATA_BYTES} byte limit")
    return v


# --- Request/Response models ---

class StoreRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=50000, description="The fact or memory to store")
    category: str = Field(..., min_length=1, max_length=100, description="Memory category (e.g. family, preference, project)")
    subject: Optional[str] = Field(None, max_length=200, description="Who/what this is about")
    source: str = Field("manual", max_length=100, description="Where this came from")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence score")
    metadata: Optional[dict] = Field(None, description="Additional structured data")

    _validate_metadata = field_validator("metadata")(_check_metadata_size)


class StoreResponse(BaseModel):
    memory_id: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    limit: int = Field(5, ge=1, le=50, description="Max results")
    category: Optional[str] = Field(None, max_length=100, description="Filter by category")
    subject: Optional[str] = Field(None, max_length=200, description="Filter by subject")


class ContextRequest(BaseModel):
    query: Optional[str] = Field(None, max_length=2000, description="Query for relevance filtering")
    core_limit: int = Field(10, ge=1, le=50)
    relevant_limit: int = Field(5, ge=1, le=50)
    use_semantic: bool = Field(False, description="Enable slower semantic search")


class SupersedeRequest(BaseModel):
    old_id: str = Field(..., max_length=100, description="Memory ID to supersede")
    new_content: str = Field(..., min_length=1, max_length=50000, description="Replacement content")


# --- Endpoints ---
# Routes use `def` (not `async def`) because they call synchronous maasv
# functions. FastAPI runs `def` routes in a threadpool, keeping the event
# loop free for other requests.

@router.post("/store", response_model=StoreResponse)
def store(req: StoreRequest):
    """Store a new memory with dedup check."""
    from maasv.core.store import store_memory

    memory_id = store_memory(
        content=req.content,
        category=req.category,
        subject=req.subject,
        source=req.source,
        confidence=req.confidence,
        metadata=req.metadata,
    )
    return StoreResponse(memory_id=memory_id)


@router.post("/search")
def search(req: SearchRequest):
    """Search memories using 3-signal retrieval (vector + BM25 + graph)."""
    from maasv.core.retrieval import find_similar_memories

    results = find_similar_memories(
        query=req.query,
        limit=req.limit,
        category=req.category,
        subject=req.subject,
    )
    return {"results": results, "count": len(results)}


@router.post("/context")
def context(req: ContextRequest):
    """Get tiered memory context (identity > family > preference > relevant)."""
    from maasv.core.retrieval import get_tiered_memory_context

    text = get_tiered_memory_context(
        query=req.query,
        core_limit=req.core_limit,
        relevant_limit=req.relevant_limit,
        use_semantic=req.use_semantic,
    )
    return {"context": text}


@router.get("/{memory_id}")
def get_memory(memory_id: str):
    """Get a specific memory by ID."""
    from maasv.core.db import _db

    with _db() as db:
        row = db.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")

    return dict(row)


@router.delete("/{memory_id}")
def delete(memory_id: str):
    """Permanently delete a memory."""
    from maasv.core.store import delete_memory

    try:
        deleted = delete_memory(memory_id)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")

    return {"deleted": True, "memory_id": memory_id}


@router.post("/supersede", response_model=StoreResponse)
def supersede(req: SupersedeRequest):
    """Mark a memory as superseded and create a replacement."""
    from maasv.core.store import supersede_memory

    try:
        new_id = supersede_memory(old_id=req.old_id, new_content=req.new_content)
    except ValueError:
        raise HTTPException(status_code=404, detail="Original memory not found")

    return StoreResponse(memory_id=new_id)

"""Graph endpoints: entity and relationship CRUD."""

import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_METADATA_BYTES = 10_000


def _check_metadata_size(v: dict | None) -> dict | None:
    if v is not None and len(json.dumps(v, default=str)) > MAX_METADATA_BYTES:
        raise ValueError(f"metadata exceeds {MAX_METADATA_BYTES} byte limit")
    return v


# --- Request models ---

class CreateEntityRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=500)
    entity_type: str = Field(..., min_length=1, max_length=50, description="person, place, project, org, event, technology")
    canonical_name: Optional[str] = Field(None, max_length=500)
    metadata: Optional[dict] = None

    _validate_metadata = field_validator("metadata")(_check_metadata_size)


class AddRelationshipRequest(BaseModel):
    subject_id: str = Field(..., max_length=100)
    predicate: str = Field(..., min_length=1, max_length=200)
    object_id: Optional[str] = Field(None, max_length=100)
    object_value: Optional[str] = Field(None, max_length=2000)
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    source: Optional[str] = Field(None, max_length=100)
    metadata: Optional[dict] = None

    _validate_metadata = field_validator("metadata")(_check_metadata_size)


class SearchEntitiesRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    entity_type: Optional[str] = Field(None, max_length=50)
    limit: int = Field(10, ge=1, le=100)


# --- Endpoints ---

@router.post("/entities")
def create_or_find_entity(req: CreateEntityRequest):
    """Create a new entity or find existing by name."""
    from maasv.core.graph import find_or_create_entity, get_entity

    entity_id = find_or_create_entity(
        name=req.name,
        entity_type=req.entity_type,
        metadata=req.metadata,
    )
    entity = get_entity(entity_id)
    return entity


@router.post("/entities/search")
def search_entities(req: SearchEntitiesRequest):
    """Search entities by name using FTS."""
    from maasv.core.graph import search_entities

    results = search_entities(
        query=req.query,
        entity_type=req.entity_type,
        limit=req.limit,
    )
    return {"results": results, "count": len(results)}


@router.get("/entities/{entity_id}")
def get_entity(entity_id: str):
    """Get entity with all active relationships."""
    from maasv.core.graph import get_entity_profile

    profile = get_entity_profile(entity_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")

    # Normalize relationship entries to use consistent field names.
    # maasv returns entity_name/value internally; we expose object_name/object_value
    # to match the relationship table schema and TypeScript types.
    for pred, rels in profile.get("relationships", {}).items():
        for rel in rels:
            if "entity_name" in rel:
                rel["object_name"] = rel.pop("entity_name")
            if "entity_id" in rel:
                rel["object_id"] = rel.pop("entity_id")
            if "entity_type" in rel:
                rel["object_type"] = rel.pop("entity_type")
            if "value" in rel:
                rel["object_value"] = rel.pop("value")

    return profile


@router.post("/relationships")
def add_relationship(req: AddRelationshipRequest):
    """Add a temporal relationship between entities."""
    from maasv.core.graph import add_relationship

    if req.object_id is None and req.object_value is None:
        raise HTTPException(status_code=422, detail="Must provide either object_id or object_value")

    rel_id = add_relationship(
        subject_id=req.subject_id,
        predicate=req.predicate,
        object_id=req.object_id,
        object_value=req.object_value,
        confidence=req.confidence,
        source=req.source,
        metadata=req.metadata,
    )
    return {"relationship_id": rel_id}

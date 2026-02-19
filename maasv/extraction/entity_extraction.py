"""
Entity Extraction for Knowledge Graph Population

Extracts entities and relationships from conversation summaries using LLM.
Uses the injected LLMProvider — no direct API calls.

Entity Types: person, place, project, organization, event, technology
"""

import logging
from typing import Optional

logger = logging.getLogger("maasv.extraction.entity_extraction")

# Words that should never be entities
BLOCKED_ENTITY_NAMES = {
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    "the", "a", "an", "this", "that", "these", "those",
    "someone", "something", "everyone", "everything",
    "anyone", "anything", "nobody", "nothing",
}

# Infer entity type from relationship predicate
PREDICATE_OBJECT_TYPE = {
    "located_in": "place", "lives_in": "place", "visited": "place", "located_at": "place",
    "works_on": "project", "manages": "project", "created": "project", "owns": "project",
    "works_at": "organization",
    "uses_tech": "technology", "built_with": "technology", "runs_on": "technology",
    "hosted_on": "technology", "depends_on": "technology", "written_in": "technology",
    "parent_of": "person", "child_of": "person", "married_to": "person",
    "sibling_of": "person", "friend_of": "person", "works_with": "person", "colleague_of": "person",
    # Causal predicates — any entity type
    "caused_by": None, "led_to": None, "resulted_in": None,
    "motivated_by": None, "enabled_by": None, "blocked_by": None, "chose_over": None,
}

# Causal predicates require higher confidence to avoid hallucination
CAUSAL_PREDICATES = {"caused_by", "led_to", "resulted_in", "motivated_by", "enabled_by", "blocked_by", "chose_over"}
CAUSAL_MIN_CONFIDENCE = 0.8

# Cardinality caps (Task 2)
MAX_ENTITIES_PER_EXTRACTION = 20
MAX_RELATIONSHIPS_PER_EXTRACTION = 30
# Field length caps (Task 3)
MAX_CONTENT_LENGTH = 10_000

PREDICATE_SUBJECT_TYPE = {
    "located_in": "place",
    "has_email": "person", "has_phone": "person",
    "has_birthday": "person", "has_age": "person",
}


def _is_garbage_entity(name: str) -> bool:
    """Check if an entity name is a pronoun, vague reference, or common word."""
    normalized = name.lower().strip()
    if normalized in BLOCKED_ENTITY_NAMES:
        return True
    if normalized.startswith(("that ", "the ", "this ", "my ", "a ", "an ", "some ")):
        return True
    if len(normalized) <= 1:
        return True
    return False


def _infer_entity_type(name: str, role: str, predicate: str) -> str:
    """Infer entity type from relationship context."""
    if role == "object":
        inferred = PREDICATE_OBJECT_TYPE.get(predicate)
    else:
        inferred = PREDICATE_SUBJECT_TYPE.get(predicate)
    return inferred or "unknown"


def _build_extraction_prompt(known_entities: dict[str, str]) -> str:
    """Build the extraction prompt with known entities from config."""
    entities_section = ""
    if known_entities:
        lines = [f"- {name} ({etype})" for name, etype in known_entities.items()]
        entities_section = "Known entities (avoid duplicates):\n" + "\n".join(lines)

    return """Analyze this conversation summary and extract entities and relationships for a knowledge graph.

Extract:
1. **People** - Anyone mentioned by name (family, friends, colleagues)
2. **Places** - Locations, restaurants, venues, addresses
3. **Projects** - Work or personal projects mentioned
4. **Organizations** - Companies, schools, teams
5. **Events** - Specific events or occasions
6. **Technologies** - Programming languages, frameworks, databases, tools, services, platforms

For each entity, also identify relationships to other entities.

""" + entities_section + """

Return JSON:
```json
{
    "entities": [
        {
            "name": "Display Name",
            "type": "person|place|project|organization|event|technology",
            "description": "Brief context",
            "confidence": 0.0-1.0
        }
    ],
    "relationships": [
        {
            "subject": "Entity Name",
            "predicate": "relationship_type",
            "object": "Other Entity or Value",
            "object_is_entity": true,
            "confidence": 0.0-1.0
        }
    ]
}
```

Relationship predicates:
- parent_of, child_of, married_to, sibling_of (family)
- friend_of, works_with, colleague_of (social)
- works_at, works_on, manages (professional)
- located_in, visited, lives_in (location)
- uses_tech, built_with, written_in (technology — project uses a technology)
- runs_on, hosted_on, depends_on (infrastructure — project runs on a platform/service)
- has_email, has_phone, has_birthday (attributes - object_is_entity: false)
- caused_by, led_to, resulted_in (causal — only when clearly stated, confidence 0.9+)
- motivated_by, enabled_by, blocked_by (causal reasoning)
- chose_over (decisions — subject chose X over object)

IMPORTANT:
- Only extract entities with PROPER NAMES
- NEVER extract pronouns or vague references
- Confidence 0.9+ for explicitly stated facts, 0.6-0.8 for inferred
- Skip known entities unless you have new information about them
- Return empty arrays if nothing notable to extract

CONVERSATION SUMMARY:
__SUMMARY__

TOPIC: __TOPIC__
"""


class EntityExtractor:
    """Extracts entities and relationships from conversation summaries."""

    def __init__(self, model: str = None):
        import maasv
        self.model = model or maasv.get_config().extraction_model

    def extract_from_summary(
        self,
        summary: str,
        topic: str = "",
        existing_entities: Optional[list[str]] = None
    ) -> dict:
        """Extract entities and relationships from a conversation summary."""
        if not summary or len(summary) < 50:
            return {"entities": [], "relationships": [], "status": "empty"}

        import maasv
        config = maasv.get_config()
        llm = maasv.get_llm()

        prompt_template = _build_extraction_prompt(config.known_entities)
        prompt = prompt_template.replace(
            "__SUMMARY__", summary
        ).replace(
            "__TOPIC__", topic or "General conversation"
        )

        try:
            content = llm.call(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=1500,
                source="entity-extract",
            )

            # Parse JSON from response
            from maasv.utils import parse_llm_json
            data = parse_llm_json(content)

            if data is None:
                logger.warning("Unparseable extraction response")
                return {"entities": [], "relationships": [], "status": "error", "error": "No data parsed"}

            entities = data.get("entities", [])
            relationships = data.get("relationships", [])

            # Task 2: Cardinality caps
            if len(entities) > MAX_ENTITIES_PER_EXTRACTION:
                logger.warning(f"Capping entities from {len(entities)} to {MAX_ENTITIES_PER_EXTRACTION}")
                entities = entities[:MAX_ENTITIES_PER_EXTRACTION]
            if len(relationships) > MAX_RELATIONSHIPS_PER_EXTRACTION:
                logger.warning(f"Capping relationships from {len(relationships)} to {MAX_RELATIONSHIPS_PER_EXTRACTION}")
                relationships = relationships[:MAX_RELATIONSHIPS_PER_EXTRACTION]

            logger.info(f"Extracted {len(entities)} entities, {len(relationships)} relationships")

            return {"entities": entities, "relationships": relationships, "status": "success"}

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return {"entities": [], "relationships": [], "status": "error", "error": str(e)}

    def store_extracted_entities(self, extraction_result: dict) -> dict:
        """Store extracted entities and relationships in the knowledge graph."""
        from maasv.core.graph import (
            find_or_create_entity,
            find_entity_by_name,
            add_relationship,
        )

        stats = {"entities_created": 0, "relationships_created": 0, "entities_skipped": 0}

        if extraction_result.get("status") != "success":
            return stats

        entity_id_map = {}

        from maasv.core.graph import _clamp_confidence, MAX_ENTITY_NAME_LENGTH, VALID_PREDICATES

        for entity in extraction_result.get("entities", []):
            name = entity.get("name", "").strip()[:MAX_ENTITY_NAME_LENGTH]
            entity_type = entity.get("type", "concept")
            confidence = _clamp_confidence(entity.get("confidence", 0.7))
            description = entity.get("description", "")
            if isinstance(description, str):
                description = description[:MAX_CONTENT_LENGTH]

            if not name or _is_garbage_entity(name):
                stats["entities_skipped"] += 1
                continue
            if confidence < 0.5:
                stats["entities_skipped"] += 1
                continue

            try:
                existing = find_entity_by_name(name, entity_type)
                if existing:
                    entity_id_map[name] = existing["id"]
                    stats["entities_skipped"] += 1
                    continue

                entity_id = find_or_create_entity(
                    name=name,
                    entity_type=entity_type,
                    metadata={
                        "description": description,
                        "source": "extraction",
                        "confidence": confidence,
                    }
                )
                entity_id_map[name] = entity_id
                stats["entities_created"] += 1
                logger.debug(f"Created entity: {name} ({entity_type})")

            except Exception as e:
                logger.warning(f"Failed to create entity {name}: {e}")

        for rel in extraction_result.get("relationships", []):
            subject_name = rel.get("subject", "").strip()[:MAX_ENTITY_NAME_LENGTH]
            predicate = rel.get("predicate", "").strip()
            object_name = rel.get("object", "").strip()[:MAX_ENTITY_NAME_LENGTH]
            object_is_entity = rel.get("object_is_entity", True)
            confidence = _clamp_confidence(rel.get("confidence", 0.7))

            if not subject_name or not predicate or not object_name:
                continue
            # Task 5: Reject unknown predicates
            if predicate not in VALID_PREDICATES:
                logger.warning(f"Skipping unknown predicate from extraction: {predicate!r}")
                continue
            # Causal predicates require higher confidence to avoid hallucination
            min_confidence = CAUSAL_MIN_CONFIDENCE if predicate in CAUSAL_PREDICATES else 0.5
            if confidence < min_confidence:
                continue

            try:
                if _is_garbage_entity(subject_name) or _is_garbage_entity(object_name):
                    continue

                subject_id = entity_id_map.get(subject_name)
                if not subject_id:
                    subject_entity = find_entity_by_name(subject_name)
                    if subject_entity:
                        subject_id = subject_entity["id"]
                    else:
                        subject_id = find_or_create_entity(
                            name=subject_name,
                            entity_type=_infer_entity_type(subject_name, "subject", predicate),
                            metadata={"source": "extraction_relationship"}
                        )

                if object_is_entity:
                    object_id = entity_id_map.get(object_name)
                    if not object_id:
                        object_entity = find_entity_by_name(object_name)
                        if object_entity:
                            object_id = object_entity["id"]
                        else:
                            object_id = find_or_create_entity(
                                name=object_name,
                                entity_type=_infer_entity_type(object_name, "object", predicate),
                                metadata={"source": "extraction_relationship"}
                            )

                    add_relationship(
                        subject_id=subject_id, predicate=predicate,
                        object_id=object_id, confidence=confidence, source="extraction"
                    )
                else:
                    add_relationship(
                        subject_id=subject_id, predicate=predicate,
                        object_value=object_name, confidence=confidence, source="extraction"
                    )

                stats["relationships_created"] += 1
                logger.debug(f"Created relationship: {subject_name} -{predicate}-> {object_name}")

            except Exception as e:
                logger.warning(f"Failed to create relationship: {e}")

        return stats


# Singleton
_extractor: Optional[EntityExtractor] = None


def get_entity_extractor() -> EntityExtractor:
    """Get the global entity extractor."""
    global _extractor
    if _extractor is None:
        _extractor = EntityExtractor()
    return _extractor


def extract_and_store_entities(summary: str, topic: str = "") -> dict:
    """Convenience function to extract and store entities from a summary."""
    extractor = get_entity_extractor()
    extraction_result = extractor.extract_from_summary(summary, topic)

    storage_result = {}
    if extraction_result.get("status") == "success":
        storage_result = extractor.store_extracted_entities(extraction_result)

    return {"extraction": extraction_result, "storage": storage_result}

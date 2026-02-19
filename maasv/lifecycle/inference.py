"""
Inference Generation - Sleep-Time Compute

Resolves vague references in recent conversations to specific entities.
Uses the injected LLMProvider — no direct API calls.
"""

import logging
import json
from typing import Callable
from datetime import datetime, timezone

logger = logging.getLogger("maasv.lifecycle.inference")


def run_inference_job(data: dict, cancel_check: Callable[[], bool]) -> dict:
    """
    Run an inference job on recent conversation.

    Args:
        data: {"messages": [...], "entities": [...]}
        cancel_check: Function to check if job should stop
    """
    messages = data.get("messages", [])
    known_entities = data.get("entities", [])

    if not messages:
        return {"inferences": [], "stored": 0}
    if cancel_check():
        return {"inferences": [], "stored": 0, "cancelled": True}

    conversation_text = _format_messages(messages)
    entities_text = _format_entities(known_entities)

    if cancel_check():
        return {"inferences": [], "stored": 0, "cancelled": True}

    inferences = _extract_inferences(conversation_text, entities_text)

    if cancel_check():
        return {"inferences": inferences, "stored": 0, "cancelled": True}

    stored = _store_inferences(inferences)

    return {"inferences": inferences, "stored": stored}


def _format_messages(messages: list) -> str:
    """Format messages for analysis."""
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
    return "\n".join(lines)


def _format_entities(entities: list) -> str:
    """Format entities for context."""
    if not entities:
        return "No known entities."
    lines = []
    for ent in entities:
        name = ent.get("name", "Unknown")
        etype = ent.get("entity_type", "unknown")
        lines.append(f"- {name} ({etype})")
    return "\n".join(lines)


def _extract_inferences(conversation: str, entities: str) -> list[dict]:
    """Use LLM to extract inferences from conversation."""
    import maasv

    prompt = f"""Analyze this conversation for vague references that could be linked to specific entities.

Known entities:
{entities}

Conversation:
{conversation}

Find phrases like:
- "that place", "the restaurant", "that guy"
- Pronouns referring to specific people/things
- Implied references ("we should go back" - go back where?)

For each reference found, provide:
1. The exact reference phrase
2. What entity it likely refers to
3. Entity type (person, place, restaurant, project, event)
4. Confidence (0.0-1.0)
5. Evidence (why you think this)

Return as JSON array. If no references found, return empty array.

JSON output:"""

    try:
        config = maasv.get_config()
        llm = maasv.get_llm()

        text = llm.call(
            messages=[{"role": "user", "content": prompt}],
            model=config.inference_model,
            max_tokens=2000,
            source="sleep-inference",
        )

        from maasv.utils import parse_llm_json
        inferences = parse_llm_json(text)

        if inferences is None:
            logger.warning("[Inference] Failed to parse response")
            return []

        # Task 2: Cardinality cap
        MAX_INFERENCES_PER_JOB = 20
        if len(inferences) > MAX_INFERENCES_PER_JOB:
            logger.warning(f"[Inference] Capping inferences from {len(inferences)} to {MAX_INFERENCES_PER_JOB}")
            inferences = inferences[:MAX_INFERENCES_PER_JOB]

        logger.info(f"[Inference] Extracted {len(inferences)} inferences")
        return inferences
    except Exception as e:
        logger.error(f"[Inference] API call failed: {e}")
        return []


def _store_inferences(inferences: list[dict]) -> int:
    """Store inferences as relationships on resolved entities.

    Instead of creating polluting type="reference" entities for vague phrases
    like "that place", we store the inference as a has_reference relationship
    on the resolved entity with the reference phrase in metadata.
    """
    if not inferences:
        return 0

    from maasv.core.graph import find_entity_by_name, find_or_create_entity, add_relationship, _clamp_confidence

    stored = 0
    for inf in inferences:
        try:
            resolved_name = inf.get("resolved_to")
            entity_type = inf.get("entity_type", "thing")
            confidence = _clamp_confidence(inf.get("confidence", 0.5))
            reference = inf.get("reference", "")
            evidence = inf.get("evidence", "")

            if not resolved_name:
                continue

            if isinstance(resolved_name, list):
                resolved_name = resolved_name[0] if resolved_name else None
            if isinstance(reference, list):
                reference = reference[0] if reference else ""
            if isinstance(entity_type, list):
                entity_type = entity_type[0] if entity_type else "thing"

            if not resolved_name or not isinstance(resolved_name, str):
                continue

            entity = find_entity_by_name(resolved_name)
            if not entity:
                entity_id = find_or_create_entity(
                    name=resolved_name,
                    entity_type=entity_type,
                    metadata={"source": "inferred", "confidence": confidence}
                )
            else:
                entity_id = entity["id"]

            # Store inference as a value-relationship on the resolved entity
            # instead of creating a separate "reference" entity.
            add_relationship(
                subject_id=entity_id,
                predicate="has_reference",
                object_value=reference,
                confidence=confidence * 0.8,
                source="sleep_inference",
                metadata={
                    "evidence": evidence,
                    "inferred_at": datetime.now(timezone.utc).isoformat(),
                    "is_vague_reference": True,
                }
            )

            stored += 1
            logger.debug(f"[Inference] Stored: '{reference}' -> {resolved_name}")

        except Exception as e:
            logger.warning(f"[Inference] Failed to store inference: {e}")

    return stored

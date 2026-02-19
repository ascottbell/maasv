"""Shared utilities for maasv."""

import json
import logging

logger = logging.getLogger(__name__)


def parse_llm_json(content: str) -> dict | list | None:
    """
    Parse JSON from an LLM response, handling markdown code blocks.

    Tries:
    1. Direct JSON parse
    2. Extract from ```json ... ``` blocks
    3. Extract from ``` ... ``` blocks

    Returns parsed data or None if unparseable.
    """
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    try:
        if "```json" in content:
            stripped = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            stripped = content.split("```")[1].split("```")[0]
        else:
            return None
        return json.loads(stripped.strip())
    except (json.JSONDecodeError, IndexError):
        pass

    logger.debug("Failed to parse LLM JSON: %s", content[:100])
    return None

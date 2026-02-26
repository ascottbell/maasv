"""
LLM-based reranker for maasv retrieval.

Uses an LLM to rerank candidates by semantic relevance in a single listwise
call. Supports two backends:
- Ollama (local): qwen3:8b with no-think mode (~0.5s per query)
- Anthropic API: claude-haiku-4-5 (~1s per query, much higher quality)

Non-destructive: integrates as an optional stage in the pipeline.
When disabled or on failure, falls back to existing heuristic scoring.
"""

import json
import logging
import os
import urllib.request
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)

# Backend constants
BACKEND_OLLAMA = "ollama"
BACKEND_ANTHROPIC = "anthropic"

_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_OLLAMA_MODEL = "qwen3:8b"
_DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


def _build_candidate_text(mem: dict, index: int) -> str:
    """Format a single candidate memory for the LLM prompt."""
    parts = [f"[{index}]"]

    created = mem.get("created_at", "")
    if created:
        parts.append(f"({created[:10]})")

    cat = mem.get("category", "")
    subj = mem.get("subject", "")
    if cat:
        parts.append(f"[{cat}]")
    if subj:
        parts.append(f"({subj})")

    content = mem.get("content", "").strip()
    if len(content) > 300:
        content = content[:297] + "..."
    parts.append(content)

    return " ".join(parts)


def _build_user_prompt(query: str, candidates: list[dict], top_n: int) -> str:
    """Build the reranking prompt (shared across backends)."""
    candidate_lines = [_build_candidate_text(mem, i) for i, mem in enumerate(candidates)]
    candidates_block = "\n".join(candidate_lines)
    today = date.today().isoformat()

    return f"""You are a personal memory retrieval system. Today is {today}.

Given the query and memory candidates, return the {top_n} most relevant candidate indices as a JSON array, ordered by relevance.

Rules:
- Semantic relevance: Does the memory directly answer or relate to the query?
- Temporal awareness: If the query asks about "current", "latest", "now", or present state, the NEWEST memory on that topic is the correct answer. Older memories on the same topic are outdated.
- Protected categories: Memories about identity, family, and preferences remain relevant regardless of age.
- Specificity over breadth: Prefer memories that directly address the query over tangential mentions.
- Dates matter: Each memory shows its creation date. Use this to judge recency.

Query: {query}

Candidates:
{candidates_block}

Return ONLY a JSON array of the top {top_n} indices. Example: [3, 7, 1, 12, 0]"""


# ── Ollama backend ──────────────────────────────────────────────────

def _rerank_ollama(
    query: str,
    candidates: list[dict],
    top_n: int,
    model: str = _DEFAULT_OLLAMA_MODEL,
    base_url: str = _DEFAULT_OLLAMA_URL,
) -> Optional[list[int]]:
    """Rerank via local Ollama with pre-closed thinking (no-think mode)."""
    user_msg = _build_user_prompt(query, candidates, top_n)

    # Qwen3 chat template with pre-closed thinking for instant response
    raw_prompt = (
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )

    payload = json.dumps({
        "model": model,
        "stream": False,
        "raw": True,
        "prompt": raw_prompt,
        "options": {"temperature": 0, "num_predict": 100},
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception:
        logger.warning("LLM reranker (ollama): call failed", exc_info=True)
        return None

    raw_response = data.get("response", "")
    if not raw_response:
        logger.debug("LLM reranker (ollama): empty response")
        return None

    return _parse_indices(raw_response, len(candidates))


# ── Anthropic backend ───────────────────────────────────────────────

def _rerank_anthropic(
    query: str,
    candidates: list[dict],
    top_n: int,
    model: str = _DEFAULT_ANTHROPIC_MODEL,
    api_key: Optional[str] = None,
) -> Optional[list[int]]:
    """Rerank via Anthropic API (Haiku)."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        logger.warning("LLM reranker (anthropic): no API key")
        return None

    user_msg = _build_user_prompt(query, candidates, top_n)

    payload = json.dumps({
        "model": model,
        "max_tokens": 200,
        "messages": [{"role": "user", "content": user_msg}],
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception:
        logger.warning("LLM reranker (anthropic): API call failed", exc_info=True)
        return None

    # Extract text from response
    content_blocks = data.get("content", [])
    if not content_blocks:
        logger.debug("LLM reranker (anthropic): empty content")
        return None

    raw_response = content_blocks[0].get("text", "")
    if not raw_response:
        return None

    return _parse_indices(raw_response, len(candidates))


# ── Shared parsing ──────────────────────────────────────────────────

def _parse_indices(response: str, max_index: int) -> Optional[list[int]]:
    """Extract a list of integer indices from the LLM response."""
    text = response.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Find FIRST complete JSON array (not the last ']' which may be in explanation)
    start = text.find("[")
    end = text.find("]", start + 1) if start != -1 else -1
    if start == -1 or end == -1:
        logger.warning("LLM reranker: no JSON array in response: %s", text[:200])
        return None

    try:
        indices = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        logger.warning("LLM reranker: bad JSON: %s", text[start:end + 1][:200])
        return None

    if not isinstance(indices, list):
        return None

    valid = []
    seen = set()
    for idx in indices:
        if isinstance(idx, int) and 0 <= idx < max_index and idx not in seen:
            valid.append(idx)
            seen.add(idx)

    if not valid:
        logger.warning("LLM reranker: no valid indices")
        return None

    return valid


# ── Public API ──────────────────────────────────────────────────────

def rerank(
    query: str,
    candidates: list[dict],
    top_n: int = 10,
    backend: str = BACKEND_OLLAMA,
    **kwargs,
) -> Optional[list[int]]:
    """
    Rerank candidates using an LLM.

    Args:
        backend: "ollama" or "anthropic"
        **kwargs: Passed to the backend (model, base_url, api_key, etc.)
    """
    if not candidates:
        return []

    if backend == BACKEND_ANTHROPIC:
        return _rerank_anthropic(query, candidates, top_n, **kwargs)
    else:
        return _rerank_ollama(query, candidates, top_n, **kwargs)


def rerank_candidates(
    query: str,
    candidates: list[dict],
    limit: int = 5,
    backend: str = BACKEND_OLLAMA,
    **kwargs,
) -> Optional[list[dict]]:
    """
    High-level reranking: takes candidates, returns reranked memory list.

    LLM picks go first, then unselected candidates in original order.
    Returns None on failure (caller should fall back to heuristic scoring).
    """
    if not candidates:
        return []

    ask_n = min(limit * 2, len(candidates))
    indices = rerank(query, candidates, top_n=ask_n, backend=backend, **kwargs)
    if indices is None:
        return None

    reranked = []
    selected_ids = set()

    for idx in indices:
        reranked.append(candidates[idx])
        selected_ids.add(idx)

    for i, mem in enumerate(candidates):
        if i not in selected_ids:
            reranked.append(mem)

    return reranked

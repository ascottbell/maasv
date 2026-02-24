"""LLM-as-judge with pooled judging for retrieval evaluation.

Pools results from all adapters, then uses Claude Haiku to assign relevance
grades per (query, memory) pair. Caches judgments incrementally to support
resume after partial runs.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path

import anthropic

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """\
You are an information retrieval relevance judge. You will be given a user's \
search query and a set of memory entries from a personal knowledge base. \
For each memory, rate how relevant it is to the query.

Relevance scale:
0 = Not relevant — no meaningful connection to the query
1 = Marginally relevant — tangentially related topic but doesn't answer the query
2 = Relevant — addresses the query topic with useful information
3 = Highly relevant — directly answers or is exactly what the user is looking for

Be strict: only give 3 for memories that clearly and directly answer the query."""

JUDGE_USER_TEMPLATE = """\
Query: {query}

Rate the relevance of each memory below. Reply with ONLY a JSON object mapping \
memory_id to relevance grade (0-3). Example: {{"mem_abc": 2, "mem_xyz": 0}}

Memories:
{memories}"""


def _build_memory_text(memories: list[dict]) -> str:
    """Format memories for the judge prompt."""
    parts = []
    for m in memories:
        subject = m.get("subject") or "no subject"
        category = m.get("category") or "unknown"
        content = m.get("content", "")[:600]
        parts.append(f'[{m["id"]}] ({category}/{subject}) {content}')
    return "\n\n".join(parts)


def _judge_batch(
    client: anthropic.Anthropic,
    query: str,
    memories: list[dict],
    max_retries: int = 3,
) -> dict[str, int]:
    """Judge all candidate memories for a single query in one Haiku call.

    Returns:
        Dict mapping memory_id -> relevance grade (0-3).
    """
    memories_text = _build_memory_text(memories)

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2048,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": JUDGE_USER_TEMPLATE.format(
                            query=query, memories=memories_text
                        ),
                    }
                ],
            )

            text = response.content[0].text.strip()
            # Handle markdown code blocks
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0].strip()

            judgments = json.loads(text)
            # Validate: all values should be ints 0-3
            result = {}
            for mid, grade in judgments.items():
                grade = int(grade)
                result[mid] = max(0, min(3, grade))
            return result

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "Judge parse error (attempt %d/%d) for query %r: %s",
                attempt + 1,
                max_retries,
                query[:50],
                e,
            )
            if attempt < max_retries - 1:
                time.sleep(1)
        except anthropic.RateLimitError:
            wait = 2 ** (attempt + 1)
            logger.warning("Rate limited, waiting %ds", wait)
            time.sleep(wait)

    logger.error("Failed to judge query after %d attempts: %r", max_retries, query[:50])
    return {}


def pool_and_judge(
    db_path: Path,
    queries: list[dict],
    run_adapters_fn,
    judgments_path: Path,
) -> dict[str, dict[str, int]]:
    """Run pooled judging: retrieve from all adapters, union results, judge.

    Args:
        db_path: Path to the (copied) production database.
        queries: List of query dicts (each has 'query' key).
        run_adapters_fn: Callable(query_str, db_path) -> dict[adapter_name, list[dict]]
            Each result dict must have 'id' and 'content' keys.
        judgments_path: Path to save/load cached judgments.

    Returns:
        Dict mapping query_string -> {memory_id: relevance_grade}.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY env var for LLM judging")

    client = anthropic.Anthropic(api_key=api_key)

    # Load existing judgments (supports resume)
    cached: dict[str, dict[str, int]] = {}
    if judgments_path.exists():
        with open(judgments_path) as f:
            cached = json.load(f)
        logger.info("Loaded %d cached query judgments", len(cached))

    # Fetch memory content for building judge prompts
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row

    def _get_memory(memory_id: str) -> dict | None:
        row = db.execute(
            "SELECT id, content, category, subject FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        return dict(row) if row else None

    total = len(queries)
    judged_count = 0
    skipped_count = 0

    for i, qobj in enumerate(queries):
        query_str = qobj["query"]

        if query_str in cached:
            skipped_count += 1
            continue

        # Pool: run query through all adapters, union unique memory IDs
        adapter_results = run_adapters_fn(query_str, db_path)
        pooled_ids: set[str] = set()
        for results in adapter_results.values():
            for r in results:
                pooled_ids.add(r["id"])

        if not pooled_ids:
            cached[query_str] = {}
            judged_count += 1
            continue

        # Fetch full memory content for judge
        memories = []
        for mid in pooled_ids:
            mem = _get_memory(mid)
            if mem:
                memories.append(mem)

        if not memories:
            cached[query_str] = {}
            judged_count += 1
            continue

        # Judge this batch
        logger.info(
            "Judging query %d/%d (%d candidates): %s",
            i + 1,
            total,
            len(memories),
            query_str[:60],
        )
        judgments = _judge_batch(client, query_str, memories)
        cached[query_str] = judgments
        judged_count += 1

        # Save incrementally after each query
        judgments_path.parent.mkdir(parents=True, exist_ok=True)
        with open(judgments_path, "w") as f:
            json.dump(cached, f, indent=2)

    db.close()

    logger.info(
        "Judging complete: %d new, %d cached, %d total",
        judged_count,
        skipped_count,
        len(cached),
    )
    return cached

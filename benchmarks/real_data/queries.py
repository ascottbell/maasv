"""Query generation from a production maasv database.

Samples memories per category, uses Claude Haiku to generate natural queries,
includes legitimate queries from retrieval_log, and saves to queries.json.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sqlite3
from pathlib import Path

import anthropic

logger = logging.getLogger(__name__)

CATEGORIES = [
    "project",
    "learning",
    "decision",
    "preference",
    "behavior",
    "relationship",
    "event",
    "family",
    "person",
    "identity",
]

QUERY_GENERATION_PROMPT = """\
You are generating search queries that a user might type to find specific memories \
in their personal knowledge base.

Below are {count} real memories from the "{category}" category. Generate {num_queries} \
diverse, natural search queries that someone might use to find these memories or \
similar ones. Mix these query types:
- Specific factual (e.g., "what database does project X use?")
- Project status (e.g., "current state of the auth system")
- Preference recall (e.g., "how do I like my coffee?")
- Decision context (e.g., "why did we choose FastAPI?")
- Cross-category (e.g., "what have I learned about deployment?")
- Entity-based (e.g., "everything related to Ollama")
- Exploratory (e.g., "recent decisions about the API")

Rules:
- Queries should be natural language, not keyword dumps
- Vary length: some short (2-4 words), some longer (6-12 words)
- Each query should plausibly match at least one of the provided memories
- Do NOT quote or copy memory content verbatim — paraphrase
- Return ONLY a JSON array of query strings, nothing else

Memories:
{memories}"""


# Queries to exclude from retrieval_log (test/debug queries)
_EXCLUDED_LOG_PATTERNS = {"MAASV_TEST", "test", "elephant", "benchmark"}


def _get_category_samples(
    db: sqlite3.Connection,
    category: str,
    n: int = 10,
    seed: int = 42,
) -> list[dict]:
    """Sample n memories from a given category."""
    rows = db.execute(
        """
        SELECT id, content, category, subject, importance
        FROM memories
        WHERE category = ? AND superseded_by IS NULL
        ORDER BY importance DESC, access_count DESC
        LIMIT 50
        """,
        (category,),
    ).fetchall()

    items = [dict(r) for r in rows]
    rng = random.Random(seed)
    # Take top 5 by importance + 5 random from the rest for diversity
    top = items[:5]
    rest = items[5:]
    if rest:
        sampled = rng.sample(rest, min(n - len(top), len(rest)))
    else:
        sampled = []
    return (top + sampled)[:n]


def _get_retrieval_log_queries(db: sqlite3.Connection, limit: int = 10) -> list[str]:
    """Pull legitimate queries from the retrieval_log table."""
    try:
        rows = db.execute(
            """
            SELECT DISTINCT query
            FROM retrieval_log
            ORDER BY timestamp DESC
            LIMIT 100
            """,
        ).fetchall()
    except sqlite3.OperationalError:
        logger.debug("retrieval_log table not found")
        return []

    queries = []
    for row in rows:
        q = row[0] if isinstance(row, (tuple, list)) else row["query"]
        # Skip test/debug queries
        if any(pat.lower() in q.lower() for pat in _EXCLUDED_LOG_PATTERNS):
            continue
        queries.append(q)
        if len(queries) >= limit:
            break
    return queries


def _generate_queries_for_category(
    client: anthropic.Anthropic,
    category: str,
    memories: list[dict],
    num_queries: int = 10,
) -> list[dict]:
    """Use Claude Haiku to generate natural queries from sampled memories."""
    memories_text = "\n\n".join(
        f"[{i+1}] ({m['subject'] or 'no subject'}) {m['content'][:500]}"
        for i, m in enumerate(memories)
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": QUERY_GENERATION_PROMPT.format(
                    count=len(memories),
                    category=category,
                    num_queries=num_queries,
                    memories=memories_text,
                ),
            }
        ],
    )

    text = response.content[0].text.strip()
    # Parse JSON array from response
    # Handle possible markdown code blocks
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    try:
        query_strings = json.loads(text)
    except json.JSONDecodeError:
        logger.error("Failed to parse Haiku response for %s: %s", category, text[:200])
        return []

    return [
        {"query": q, "source": f"generated/{category}", "category": category}
        for q in query_strings
        if isinstance(q, str)
    ]


def generate_all_queries(
    db_path: Path,
    output_path: Path,
    seed: int = 42,
    queries_per_category: int = 10,
) -> list[dict]:
    """Generate queries from production DB and save to JSON.

    Args:
        db_path: Path to the (copied) production database.
        output_path: Where to save queries.json.
        seed: Random seed for reproducibility.
        queries_per_category: How many queries to generate per category.

    Returns:
        List of query dicts with 'query', 'source', 'category' keys.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set ANTHROPIC_API_KEY env var for LLM query generation"
        )

    client = anthropic.Anthropic(api_key=api_key)
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row

    all_queries: list[dict] = []

    # Generate queries per category
    for category in CATEGORIES:
        samples = _get_category_samples(db, category, n=10, seed=seed)
        if not samples:
            logger.info("No memories found for category: %s", category)
            continue

        logger.info(
            "Generating %d queries for %s (%d samples)",
            queries_per_category,
            category,
            len(samples),
        )
        queries = _generate_queries_for_category(
            client, category, samples, num_queries=queries_per_category
        )
        all_queries.extend(queries)

    # Add retrieval_log queries
    log_queries = _get_retrieval_log_queries(db, limit=10)
    for q in log_queries:
        all_queries.append({"query": q, "source": "retrieval_log", "category": None})

    db.close()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_queries, f, indent=2)

    logger.info("Generated %d total queries -> %s", len(all_queries), output_path)
    return all_queries

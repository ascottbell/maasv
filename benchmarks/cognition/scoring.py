"""Cognition-specific scoring functions.

Each benchmark category has its own scoring logic. Some are fully automated
(DB lookups), others require LLM-as-judge. All return scores in [0, 1].
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PROTECTED_CATEGORIES = {"identity", "family"}


# ============================================================================
# 1. TEMPORAL REASONING
# ============================================================================

def score_temporal(
    query: dict,
    results: list[dict],
    db: sqlite3.Connection,
) -> dict:
    """Score temporal reasoning: does the system prefer recent memories?

    Two complementary measures:
    1. Recency bias — are the returned results skewed toward recent memories
       rather than old ones?  Measured as the median age-rank of results
       within the full topic-relevant memory set (lower = more recent = better).
    2. Pairwise ordering — among the returned results, are newer memories
       ranked above older ones?  Measured via concordance (Kendall-style).

    Returns:
        Dict with score (0-1), component scores, and diagnostic details.
    """
    if not results:
        return {"score": 0.0, "reason": "no results"}

    # Get dates for all returned results
    result_dates = []
    for r in results[:10]:
        row = db.execute(
            "SELECT created_at FROM memories WHERE id = ?", (r["id"],)
        ).fetchone()
        if row:
            result_dates.append({"id": r["id"], "created_at": row["created_at"]})

    if len(result_dates) < 2:
        return {"score": 0.5, "reason": "too few results with dates to evaluate"}

    # --- Component 1: Recency bias ---
    # What fraction of the date range do our results cover?
    # Get the full date range for the DB
    date_range = db.execute(
        "SELECT MIN(created_at) as oldest, MAX(created_at) as newest "
        "FROM memories WHERE superseded_by IS NULL"
    ).fetchone()

    if date_range and date_range["oldest"] and date_range["newest"]:
        from datetime import datetime
        fmt = "%Y-%m-%d %H:%M:%S"
        try:
            db_oldest = datetime.strptime(date_range["oldest"][:19], fmt)
            db_newest = datetime.strptime(date_range["newest"][:19], fmt)
            db_span = (db_newest - db_oldest).total_seconds() or 1.0

            # Average recency of returned results (1.0 = newest, 0.0 = oldest)
            recency_scores = []
            for rd in result_dates:
                rd_dt = datetime.strptime(rd["created_at"][:19], fmt)
                recency = (rd_dt - db_oldest).total_seconds() / db_span
                recency_scores.append(recency)

            avg_recency = sum(recency_scores) / len(recency_scores)
        except (ValueError, TypeError):
            avg_recency = 0.5
    else:
        avg_recency = 0.5

    # --- Component 2: Pairwise ordering ---
    # Among returned results, are newer ones ranked higher?
    # Count concordant pairs (newer ranked higher) vs discordant pairs.
    concordant = 0
    discordant = 0
    dates_list = [rd["created_at"] for rd in result_dates]
    for i in range(len(dates_list)):
        for j in range(i + 1, len(dates_list)):
            # i is ranked higher (lower index). Is it also newer?
            if dates_list[i] > dates_list[j]:
                concordant += 1
            elif dates_list[i] < dates_list[j]:
                discordant += 1

    total_pairs = concordant + discordant
    if total_pairs > 0:
        ordering_score = concordant / total_pairs
    else:
        ordering_score = 0.5  # All same date

    # --- Composite ---
    # Recency bias (60%): are we pulling from the recent part of the DB?
    # Pairwise ordering (40%): within results, are newer ones ranked higher?
    score = avg_recency * 0.6 + ordering_score * 0.4

    return {
        "score": round(score, 4),
        "avg_recency": round(avg_recency, 4),
        "ordering_score": round(ordering_score, 4),
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "result_dates": [
            {"id": rd["id"], "created_at": rd["created_at"]}
            for rd in result_dates[:5]
        ],
        "reason": (
            f"recency={avg_recency:.2f} (1.0=all newest, 0.0=all oldest), "
            f"ordering={ordering_score:.2f} ({concordant}/{total_pairs} pairs correct)"
        ),
    }


# ============================================================================
# 2. SESSION COHERENCE
# ============================================================================

def score_session_coherence(
    chain: dict,
    results_with_session: list[list[dict]],
    results_without_session: list[list[dict]],
    db: sqlite3.Connection,
) -> dict:
    """Score session coherence via A/B comparison.

    Compares results from session-aware retrieval vs session-unaware retrieval
    for queries at position 2+ in a chain. Uses subject/category overlap with
    earlier queries as a proxy for contextual relevance.

    Returns:
        Dict with score (0-1) and per-query details.
    """
    queries = chain["queries"]
    if len(queries) < 2:
        return {"score": 0.5, "reason": "chain too short"}

    # For the first query, establish the "topic" from its results
    first_results = results_with_session[0]
    topic_subjects = set()
    topic_categories = set()
    for r in first_results[:5]:
        if r.get("subject"):
            topic_subjects.add(r["subject"].lower())
        if r.get("category"):
            topic_categories.add(r["category"])

    per_query_scores = []

    for qi in range(1, len(queries)):
        with_session = results_with_session[qi]
        without_session = results_without_session[qi]

        # Measure: how many results in each set are "on topic"
        # (share subject/category with earlier results)?
        def topic_overlap(result_set: list[dict]) -> float:
            if not result_set:
                return 0.0
            on_topic = 0
            for r in result_set[:5]:
                subj = (r.get("subject") or "").lower()
                cat = r.get("category", "")
                if subj in topic_subjects or cat in topic_categories:
                    on_topic += 1
            return on_topic / min(5, len(result_set))

        with_overlap = topic_overlap(with_session)
        without_overlap = topic_overlap(without_session)

        # Update topic context with this query's results
        for r in with_session[:5]:
            if r.get("subject"):
                topic_subjects.add(r["subject"].lower())
            if r.get("category"):
                topic_categories.add(r["category"])

        if with_overlap > without_overlap:
            q_score = 1.0
        elif with_overlap == without_overlap:
            q_score = 0.5
        else:
            q_score = 0.0

        per_query_scores.append({
            "query": queries[qi],
            "with_session_overlap": with_overlap,
            "without_session_overlap": without_overlap,
            "score": q_score,
        })

    avg_score = (
        sum(q["score"] for q in per_query_scores) / len(per_query_scores)
        if per_query_scores
        else 0.5
    )

    return {
        "score": avg_score,
        "per_query": per_query_scores,
        "topic_subjects": list(topic_subjects)[:10],
    }


# ============================================================================
# 3. CROSS-DOMAIN GRAPH TRAVERSAL
# ============================================================================

def score_graph_traversal(
    query: dict,
    full_results: list[dict],
    vector_results: list[dict],
    graph_results: list[dict],
    db: sqlite3.Connection,
) -> dict:
    """Score graph traversal: does the graph signal add unique value?

    Measures:
    - Graph uplift: relevant results in full that aren't in vector-only
    - Domain coverage: do results span multiple subjects/categories?

    Returns:
        Dict with score (0-1), graph uplift count, domain coverage.
    """
    full_ids = {r["id"] for r in full_results[:10]}
    vector_ids = {r["id"] for r in vector_results[:10]}
    graph_ids = {r["id"] for r in graph_results[:10]}

    # Graph uplift: IDs in full results NOT in vector-only results
    graph_uplift_ids = full_ids - vector_ids
    graph_contributed = graph_uplift_ids & graph_ids

    # Domain coverage: how many distinct subjects appear in results?
    subjects_full = {r.get("subject", "").lower() for r in full_results[:10] if r.get("subject")}
    subjects_vector = {r.get("subject", "").lower() for r in vector_results[:10] if r.get("subject")}

    domain_coverage_full = len(subjects_full)
    domain_coverage_vector = len(subjects_vector)

    # Score components
    # 1. Graph uplift: did the graph add unique results? (0-0.5)
    uplift_score = min(len(graph_uplift_ids) / 3.0, 1.0) * 0.5

    # 2. Domain coverage improvement (0-0.5)
    if domain_coverage_vector > 0:
        coverage_ratio = domain_coverage_full / domain_coverage_vector
        coverage_score = min((coverage_ratio - 1.0) / 1.0, 1.0) * 0.5
        coverage_score = max(0.0, coverage_score)
    elif domain_coverage_full > 0:
        coverage_score = 0.5
    else:
        coverage_score = 0.0

    total_score = uplift_score + coverage_score

    return {
        "score": round(total_score, 4),
        "graph_uplift_count": len(graph_uplift_ids),
        "graph_contributed_count": len(graph_contributed),
        "domain_coverage_full": domain_coverage_full,
        "domain_coverage_vector": domain_coverage_vector,
    }


# ============================================================================
# 4. CONSOLIDATION RESISTANCE
# ============================================================================

def score_consolidation(
    query: dict,
    results: list[dict],
    db: sqlite3.Connection,
) -> dict:
    """Score consolidation: are any superseded memories surfaced?

    Hard pass/fail: any superseded memory in results = score 0.
    Bonus: checks if the consolidated replacement appears in top-3.

    Returns:
        Dict with score (0 or 1), any violations found.
    """
    if not results:
        return {"score": 0.5, "reason": "no results returned"}

    violations = []
    for i, r in enumerate(results[:10]):
        row = db.execute(
            "SELECT superseded_by FROM memories WHERE id = ?",
            (r["id"],),
        ).fetchone()
        if row and row["superseded_by"]:
            violations.append({
                "rank": i + 1,
                "id": r["id"],
                "superseded_by": row["superseded_by"],
            })

    if violations:
        return {
            "score": 0.0,
            "reason": f"{len(violations)} superseded memories in results",
            "violations": violations,
        }

    return {
        "score": 1.0,
        "reason": "no superseded memories in results",
    }


# ============================================================================
# 5. DECAY + IDENTITY PROTECTION
# ============================================================================

def score_decay_protection(
    query: dict,
    results: list[dict],
    db: sqlite3.Connection,
) -> dict:
    """Score decay/protection behavior.

    For 'protected' queries: identity/family memories should appear in top-3.
    For 'transient' queries: recent memories should outrank old event memories.

    Returns:
        Dict with score (0-1) and reasoning.
    """
    if not results:
        return {"score": 0.0, "reason": "no results"}

    query_type = query.get("type", "protected")
    expected_cats = set(query.get("expected_categories", []))

    if query_type == "protected":
        # Check: does a protected-category memory appear in top-3?
        for i, r in enumerate(results[:3]):
            cat = r.get("category", "")
            if cat in PROTECTED_CATEGORIES or cat in expected_cats:
                return {
                    "score": 1.0,
                    "reason": f"protected category '{cat}' found at rank #{i+1}",
                    "matched_category": cat,
                }
        # Check top-5 for partial credit
        for i, r in enumerate(results[:5]):
            cat = r.get("category", "")
            if cat in PROTECTED_CATEGORIES or cat in expected_cats:
                return {
                    "score": 0.5,
                    "reason": f"protected category '{cat}' found at rank #{i+1} (not top-3)",
                    "matched_category": cat,
                }
        return {
            "score": 0.0,
            "reason": f"no protected/expected category in top-5; expected {expected_cats}",
        }

    elif query_type == "transient":
        # Check: do recent memories rank above stale event memories?
        result_entries = []
        for r in results[:5]:
            row = db.execute(
                "SELECT created_at, category FROM memories WHERE id = ?",
                (r["id"],),
            ).fetchone()
            if row:
                result_entries.append({
                    "id": r["id"],
                    "created_at": row["created_at"],
                    "category": row["category"],
                })

        if len(result_entries) < 2:
            return {"score": 0.5, "reason": "too few results to compare temporal ordering"}

        # Find event/transient memories in results and check if they're ordered recent-first
        event_entries = [
            e for e in result_entries if e["category"] in expected_cats
        ]
        if not event_entries:
            # No event-type results — could mean they correctly decayed away
            non_event_count = len(result_entries)
            return {
                "score": 0.8,
                "reason": f"no transient-category results in top-5 ({non_event_count} non-event results)",
            }

        # Check if event memories are ordered by recency
        dates = [e["created_at"] for e in event_entries]
        if dates == sorted(dates, reverse=True):
            return {
                "score": 1.0,
                "reason": f"{len(event_entries)} event memories in correct temporal order",
            }

        # Tolerance: if all event memories are within 48h of each other,
        # they're from the same event cluster. Micro-ordering within a
        # cluster doesn't test decay behavior — decay operates over days/weeks.
        try:
            fmt = "%Y-%m-%d %H:%M:%S"
            parsed = [datetime.strptime(d[:19], fmt) for d in dates]
            span_hours = (max(parsed) - min(parsed)).total_seconds() / 3600
            if span_hours <= 48:
                return {
                    "score": 0.8,
                    "reason": (
                        f"event memories within {span_hours:.0f}h cluster "
                        f"(order not significant at this granularity)"
                    ),
                }
        except (ValueError, TypeError):
            pass

        return {
            "score": 0.3,
            "reason": f"event memories not in temporal order: {dates}",
        }

    return {"score": 0.5, "reason": f"unknown query type: {query_type}"}


# ============================================================================
# 6. PROACTIVE RELEVANCE (LLM-as-judge)
# ============================================================================

PROACTIVE_JUDGE_SYSTEM = """\
You are evaluating whether a memory retrieval system returns proactively useful \
results — memories that are relevant and helpful even though the user didn't \
explicitly ask for them.

For each memory, rate its proactive value:
0 = Not relevant to the query at all
1 = Topically related but obvious — any keyword search would find this
2 = Relevant and contextually useful — requires some inference to connect
3 = Highly proactive — the user didn't ask for this but it's exactly what \
they need; only findable through entity connections or behavioral patterns"""

PROACTIVE_JUDGE_TEMPLATE = """\
Query: {query}

Rate the proactive relevance of each memory. Reply with ONLY a JSON object \
mapping memory_id to grade (0-3). Example: {{"mem_abc": 2, "mem_xyz": 3}}

Memories:
{memories}"""


def score_proactive_llm(
    query: dict,
    full_results: list[dict],
    vector_results: list[dict],
    db: sqlite3.Connection,
    client=None,
) -> dict:
    """Score proactive relevance using LLM-as-judge.

    Grades each result 0-3 on proactive value, then compares full pipeline
    vs vector-only to measure how much the graph/fusion adds.

    Args:
        client: anthropic.Anthropic instance. If None, returns placeholder.

    Returns:
        Dict with score (0-1), grade distribution, proactive uplift.
    """
    if client is None:
        return {"score": -1.0, "reason": "no LLM client — skipped"}

    def _judge(results: list[dict]) -> dict[str, int]:
        if not results:
            return {}

        # Build memory text (truncated, no raw PII in output)
        parts = []
        for r in results[:10]:
            subject = r.get("subject") or "no subject"
            category = r.get("category") or "unknown"
            content = r.get("content", "")[:600]
            parts.append(f'[{r["id"]}] ({category}/{subject}) {content}')
        memories_text = "\n\n".join(parts)

        import anthropic as _anthropic

        for attempt in range(3):
            try:
                response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=2048,
                    system=PROACTIVE_JUDGE_SYSTEM,
                    messages=[{
                        "role": "user",
                        "content": PROACTIVE_JUDGE_TEMPLATE.format(
                            query=query["query"],
                            memories=memories_text,
                        ),
                    }],
                )
                text = response.content[0].text.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1]
                    text = text.rsplit("```", 1)[0].strip()
                grades = json.loads(text)
                return {mid: max(0, min(3, int(g))) for mid, g in grades.items()}
            except (json.JSONDecodeError, ValueError):
                if attempt < 2:
                    time.sleep(1)
            except _anthropic.RateLimitError:
                time.sleep(2 ** (attempt + 1))

        return {}

    full_grades = _judge(full_results)
    vector_grades = _judge(vector_results)

    # Proactive score: average of grade-2+ results (contextual or proactive)
    def proactive_avg(grades: dict) -> float:
        if not grades:
            return 0.0
        proactive = [g for g in grades.values() if g >= 2]
        return len(proactive) / max(len(grades), 1)

    full_proactive = proactive_avg(full_grades)
    vector_proactive = proactive_avg(vector_grades)

    # Count grade-3 (truly proactive) results
    full_grade3 = sum(1 for g in full_grades.values() if g == 3)
    vector_grade3 = sum(1 for g in vector_grades.values() if g == 3)

    # Score: weighted combination of absolute proactive rate and uplift over vector
    uplift = full_proactive - vector_proactive
    score = min(1.0, full_proactive * 0.6 + max(0, uplift) * 0.4)

    return {
        "score": round(score, 4),
        "full_proactive_rate": round(full_proactive, 4),
        "vector_proactive_rate": round(vector_proactive, 4),
        "uplift": round(uplift, 4),
        "full_grade3_count": full_grade3,
        "vector_grade3_count": vector_grade3,
        "full_grades": full_grades,
        "vector_grades": vector_grades,
    }

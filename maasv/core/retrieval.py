"""
maasv Memory Retrieval

3-signal retrieval pipeline (vector + BM25 + graph), RRF fusion,
cross-encoder reranking, diversity-aware selection, and tiered context.
"""

import logging
import math
import re
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

from maasv.core.db import (
    _db,
    get_query_embedding,
    serialize_embedding,
    _record_memory_access,
    _escape_like,
    _sanitize_fts_input,
)

logger = logging.getLogger(__name__)


# ============================================================================
# IMPORTANCE SCORING
# ============================================================================

def _importance_score(
    candidates: list[dict],
    protected: set[str],
    now: datetime,
    vector_distances: dict[str, float],
    bm25_ids: set[str],
    graph_ids: set[str],
) -> tuple[list[dict], list[dict]]:
    """
    Score candidates by importance-weighted formula. Separates into primary
    (have vector distance) and supplementary (no vector distance) lists,
    both sorted by _imp_score descending.

    Scoring: (1 - distance) * importance * decay * log(2 + access_count) + agreement_bonus
    """
    primary = []
    supplementary = []

    for mem in candidates:
        importance = mem.get('importance') or 0.5
        access_count = mem.get('access_count') or 0

        if mem.get('category') in protected:
            decay_factor = 1.0
        else:
            try:
                created = datetime.fromisoformat(mem['created_at'])
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                days_old = (now - created).total_seconds() / 86400
            except (ValueError, TypeError):
                days_old = 0
            decay_factor = math.exp(-days_old / 180)

        dampened_access = min(access_count, 5)
        distance = vector_distances.get(mem['id'])

        if distance is not None:
            base_score = (1.0 - distance) * importance * decay_factor * math.log(2 + dampened_access)
            signal_count = 1
            if mem['id'] in bm25_ids:
                signal_count += 1
            if mem['id'] in graph_ids:
                signal_count += 1
            mem['_imp_score'] = base_score + (signal_count - 1) * 0.01
            primary.append(mem)
        else:
            mem['_imp_score'] = importance * decay_factor * math.log(2 + dampened_access) * 0.0001
            supplementary.append(mem)

    primary.sort(key=lambda m: m['_imp_score'], reverse=True)
    supplementary.sort(key=lambda m: m['_imp_score'], reverse=True)

    return primary, supplementary


# ============================================================================
# MULTI-SIGNAL RETRIEVAL HELPERS
# ============================================================================

def _query_to_entity_fts(query: str) -> str:
    """
    Convert a natural-language query to OR-separated FTS5 terms for entity search.

    FTS5 defaults to AND, so "MyApp architecture" requires both words in entity
    names — which misses the "MyApp" entity. Converting to "MyApp OR architecture"
    ensures we find entities matching ANY query term.

    Strips FTS5 special characters and skips short/common words.
    """
    stop_words = {"the", "a", "an", "is", "of", "in", "on", "for", "and", "or", "to", "with"}
    words = re.findall(r'\w+', query)
    terms = [w for w in words if len(w) > 1 and w.lower() not in stop_words]
    if not terms:
        return query
    return " OR ".join(terms)


def _expand_query_from_graph(db, query: str) -> str:
    """
    Expand a query with related entity names from the knowledge graph.

    "MyApp architecture" -> graph says MyApp uses_tech FastAPI ->
    returns "MyApp architecture OR FastAPI"

    This provides redundant coverage with _find_memories_by_graph():
    if graph signal or BM25 fails independently, the other catches it.
    """
    entity_fts_query = _query_to_entity_fts(query)
    try:
        entities = db.execute("""
            SELECT e.id, e.name
            FROM entities_fts f
            JOIN entities e ON f.rowid = e.rowid
            WHERE entities_fts MATCH ?
            LIMIT 5
        """, (entity_fts_query,)).fetchall()
    except Exception:
        return query

    if not entities:
        return query

    entity_ids = [e["id"] for e in entities]
    placeholders = ",".join("?" * len(entity_ids))

    try:
        related = db.execute(f"""
            SELECT DISTINCT e.name
            FROM relationships r
            JOIN entities e ON (
                CASE
                    WHEN r.subject_id IN ({placeholders}) THEN r.object_id
                    ELSE r.subject_id
                END
            ) = e.id
            WHERE (r.subject_id IN ({placeholders}) OR r.object_id IN ({placeholders}))
            AND r.valid_to IS NULL
            LIMIT 10
        """, entity_ids * 3).fetchall()
    except Exception:
        return query

    # Build expanded query: original OR "related term 1" OR "related term 2"
    expansion_terms = []
    for row in related:
        name = row["name"]
        if name:
            clean = re.sub(r'[^\w\s]', '', name).strip()
            if clean and clean.lower() not in query.lower():
                expansion_terms.append(f'"{clean}"')

    if not expansion_terms:
        return query

    # FTS5 OR syntax
    return query + " OR " + " OR ".join(expansion_terms)


def _find_memories_by_bm25(db, query: str, limit: int = 50) -> list[dict]:
    """
    Return memories ranked by BM25 relevance from the FTS5 index.

    Uses bm25() scoring function with weights: content=10, category=1, subject=5.
    Only returns active (non-superseded) memories.
    Expands query with graph-connected entity names before searching.
    Returns dicts with 'id' key (required for RRF) and 'bm25_score'.
    """
    query = _sanitize_fts_input(query)
    if not query:
        return []
    expanded_query = _expand_query_from_graph(db, query)
    if expanded_query != query:
        logger.debug("BM25 query expanded: %s -> %s", query, expanded_query)

    try:
        rows = db.execute("""
            SELECT m.id, m.content, m.category, m.subject, m.confidence,
                   m.created_at, m.metadata, m.importance, m.access_count,
                   m.origin, m.origin_interface,
                   bm25(memories_fts, 10.0, 1.0, 5.0) as bm25_score
            FROM memories_fts f
            JOIN memories m ON f.rowid = m.rowid
            WHERE memories_fts MATCH ?
            AND m.superseded_by IS NULL
            ORDER BY bm25(memories_fts, 10.0, 1.0, 5.0)
            LIMIT ?
        """, (expanded_query, limit)).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        logger.debug("BM25 search failed for expanded query: %s", expanded_query, exc_info=True)
        # Fallback: try original query without expansion
        if expanded_query != query:
            try:
                rows = db.execute("""
                    SELECT m.id, m.content, m.category, m.subject, m.confidence,
                           m.created_at, m.metadata, m.importance, m.access_count,
                           bm25(memories_fts, 10.0, 1.0, 5.0) as bm25_score
                    FROM memories_fts f
                    JOIN memories m ON f.rowid = m.rowid
                    WHERE memories_fts MATCH ?
                    AND m.superseded_by IS NULL
                    ORDER BY bm25(memories_fts, 10.0, 1.0, 5.0)
                    LIMIT ?
                """, (query, limit)).fetchall()
                return [dict(r) for r in rows]
            except Exception:
                logger.debug("BM25 fallback also failed for: %s", query, exc_info=True)
        return []


def _get_graph_expanded_names(db, query: str) -> set[str]:
    """
    Get entity names reachable via 1-hop graph expansion from query entities.
    Used by graph slot injection to score candidates by entity density.
    """
    entity_fts_query = _query_to_entity_fts(query)
    try:
        entities = db.execute("""
            SELECT e.id, e.name
            FROM entities_fts f
            JOIN entities e ON f.rowid = e.rowid
            WHERE entities_fts MATCH ?
            LIMIT 10
        """, (entity_fts_query,)).fetchall()
    except Exception:
        return set()

    if not entities:
        return set()

    direct_ids = [e["id"] for e in entities]
    direct_names = {e["name"] for e in entities if e["name"]}
    placeholders = ",".join("?" * len(direct_ids))

    expanded = set()
    try:
        rows = db.execute(f"""
            SELECT DISTINCT e.name
            FROM relationships r
            JOIN entities e ON (
                CASE WHEN r.subject_id IN ({placeholders}) THEN r.object_id
                     ELSE r.subject_id END
            ) = e.id
            WHERE (r.subject_id IN ({placeholders}) OR r.object_id IN ({placeholders}))
            AND r.valid_to IS NULL
            LIMIT 30
        """, direct_ids * 3).fetchall()
        for r in rows:
            if r["name"] and r["name"] not in direct_names:
                expanded.add(r["name"].lower())
    except Exception:
        logger.debug("Graph expansion query failed in _get_graph_expanded_names", exc_info=True)

    return expanded


def _find_memories_by_graph(db, query: str, limit: int = 50) -> list[dict]:
    """
    Find memories connected to entities mentioned in the query via graph traversal.

    Flow:
    1. Entity FTS finds entities matching query terms
    2. 1-hop expansion: follow non-noise relationships to related entities
    3. Build set of all entity names (direct + 1-hop related)
    4. Search memories_fts (FTS5) for content mentioning any entity name
    5. Fall back to subject LIKE matching if FTS yields too few results

    The 1-hop expansion is what enables "MyApp architecture" -> MyApp entity ->
    MyApp-uses_tech->FastAPI -> memories mentioning "FastAPI".

    Returns dicts with 'id' key (required for RRF) and 'graph_score'.
    """
    # Step 1: Find entities mentioned in the query via FTS
    # Convert to OR terms so "MyApp architecture" matches "MyApp" entity
    entity_fts_query = _query_to_entity_fts(query)
    try:
        entities = db.execute("""
            SELECT e.id, e.canonical_name, e.name, e.entity_type
            FROM entities_fts f
            JOIN entities e ON f.rowid = e.rowid
            WHERE entities_fts MATCH ?
            LIMIT 10
        """, (entity_fts_query,)).fetchall()
    except Exception:
        logger.debug("Entity FTS failed for query: %s", query, exc_info=True)
        return []

    if not entities:
        return []

    # Collect direct entity IDs and names
    direct_entity_ids = set()
    direct_entity_names = set()
    for e in entities:
        direct_entity_ids.add(e["id"])
        if e["name"]:
            direct_entity_names.add(e["name"])

    # Step 2: 1-hop expansion — follow non-noise relationships
    # Only collect EXPANDED entity names (from related entities, not the query matches)
    expanded_entity_names = set()
    if direct_entity_ids:
        placeholders = ",".join("?" * len(direct_entity_ids))
        try:
            related_rows = db.execute(f"""
                SELECT DISTINCT
                    e.name
                FROM relationships r
                JOIN entities e ON (
                    CASE
                        WHEN r.subject_id IN ({placeholders}) THEN r.object_id
                        ELSE r.subject_id
                    END
                ) = e.id
                WHERE (r.subject_id IN ({placeholders}) OR r.object_id IN ({placeholders}))
                AND r.valid_to IS NULL
                LIMIT 20
            """, list(direct_entity_ids) * 3).fetchall()

            for row in related_rows:
                if row["name"] and row["name"] not in direct_entity_names:
                    expanded_entity_names.add(row["name"])
        except Exception:
            logger.debug("1-hop expansion failed", exc_info=True)

    # Combine both sets for different search strategies
    all_entity_names = direct_entity_names | expanded_entity_names

    if not all_entity_names:
        return []

    # Step 3: Search memories_fts for content mentioning expanded entity names
    # AND a direct entity name. This cross-referencing ensures results are
    # relevant to the original query context (e.g., "FastAPI" AND "MyApp").
    # NOTE: Iterates per entity name (not batched) to differentiate graph_score
    # (1.0 contextual vs 0.8 plain). Could batch into single OR query if
    # graph_score differentiation is removed.
    # Without the AND, searching a large number of tech entities returns noise.
    fts_results = []
    seen_fts_ids = set()

    if expanded_entity_names and direct_entity_names:
        # Sort for deterministic iteration order (sets are unordered)
        sorted_expanded = sorted(expanded_entity_names)

        # Build the direct entity part of the FTS query: "MyApp" OR "architecture" etc.
        direct_terms = []
        for name in direct_entity_names:
            clean = re.sub(r'[^\w\s]', '', name).strip()
            if clean:
                direct_terms.append(f'"{clean}"')

        direct_clause = " OR ".join(direct_terms) if direct_terms else None

        for name in sorted_expanded:
            clean = re.sub(r'[^\w\s]', '', name).strip()
            if not clean or len(clean) < 2:
                continue

            # Search for memories mentioning this expanded entity
            # (with or without direct entity context depending on availability)
            fts_query = f'"{clean}"'
            if direct_clause:
                # Prefer memories mentioning both: "FastAPI" AND ("MyApp")
                fts_query_with_context = f'({direct_clause}) AND "{clean}"'
            else:
                fts_query_with_context = fts_query

            try:
                # Try contextual query first (e.g., "MyApp" AND "FastAPI")
                rows = db.execute("""
                    SELECT m.id, m.content, m.category, m.subject, m.confidence,
                           m.created_at, m.metadata, m.importance, m.access_count,
                           m.origin, m.origin_interface,
                           1.0 as graph_score
                    FROM memories_fts f
                    JOIN memories m ON f.rowid = m.rowid
                    WHERE memories_fts MATCH ?
                    AND m.superseded_by IS NULL
                    LIMIT 3
                """, (fts_query_with_context,)).fetchall()

                if not rows:
                    # Fall back to just the expanded entity name
                    rows = db.execute("""
                        SELECT m.id, m.content, m.category, m.subject, m.confidence,
                               m.created_at, m.metadata, m.importance, m.access_count,
                               m.origin, m.origin_interface,
                               0.8 as graph_score
                        FROM memories_fts f
                        JOIN memories m ON f.rowid = m.rowid
                        WHERE memories_fts MATCH ?
                        AND m.superseded_by IS NULL
                        LIMIT 3
                    """, (fts_query,)).fetchall()

                for row in rows:
                    r = dict(row)
                    if r["id"] not in seen_fts_ids:
                        fts_results.append(r)
                        seen_fts_ids.add(r["id"])
            except Exception:
                # If contextual query fails (syntax), try plain entity search
                try:
                    rows = db.execute("""
                        SELECT m.id, m.content, m.category, m.subject, m.confidence,
                               m.created_at, m.metadata, m.importance, m.access_count,
                               m.origin, m.origin_interface,
                               0.8 as graph_score
                        FROM memories_fts f
                        JOIN memories m ON f.rowid = m.rowid
                        WHERE memories_fts MATCH ?
                        AND m.superseded_by IS NULL
                        LIMIT 3
                    """, (f'"{clean}"',)).fetchall()
                    for row in rows:
                        r = dict(row)
                        if r["id"] not in seen_fts_ids:
                            fts_results.append(r)
                            seen_fts_ids.add(r["id"])
                except Exception:
                    continue

            if len(fts_results) >= limit:
                break

    elif direct_entity_names:
        # No expansion — search for direct entity names only
        for name in direct_entity_names:
            clean = re.sub(r'[^\w\s]', '', name).strip()
            if not clean:
                continue
            try:
                rows = db.execute("""
                    SELECT m.id, m.content, m.category, m.subject, m.confidence,
                           m.created_at, m.metadata, m.importance, m.access_count,
                           m.origin, m.origin_interface,
                           1.0 as graph_score
                    FROM memories_fts f
                    JOIN memories m ON f.rowid = m.rowid
                    WHERE memories_fts MATCH ?
                    AND m.superseded_by IS NULL
                    LIMIT 5
                """, (f'"{clean}"',)).fetchall()
                for row in rows:
                    r = dict(row)
                    if r["id"] not in seen_fts_ids:
                        fts_results.append(r)
                        seen_fts_ids.add(r["id"])
            except Exception:
                continue
            if len(fts_results) >= limit:
                break

    fts_results = fts_results[:limit]

    # Step 4: Fallback — subject LIKE matching (catches memories without FTS hits)
    # NOTE: where_clause is built from entity names via parameterized LIKE (?).
    # The column references are hardcoded strings, not user input.
    seen_ids = {r["id"] for r in fts_results}
    remaining = limit - len(fts_results)

    if remaining > 0:
        conditions = []
        params = []
        for name in all_entity_names:
            conditions.append("LOWER(m.subject) LIKE ? ESCAPE '\\'")
            params.append(f"%{_escape_like(name.lower())}%")

        if conditions:
            where_clause = " OR ".join(conditions)
            params.append(remaining)
            try:
                subject_rows = db.execute(f"""
                    SELECT DISTINCT m.id, m.content, m.category, m.subject, m.confidence,
                           m.created_at, m.metadata, m.importance, m.access_count,
                           m.origin, m.origin_interface,
                           0.8 as graph_score
                    FROM memories m
                    WHERE m.superseded_by IS NULL
                    AND ({where_clause})
                    ORDER BY m.importance DESC, m.created_at DESC
                    LIMIT ?
                """, params).fetchall()

                for row in subject_rows:
                    row_dict = dict(row)
                    if row_dict["id"] not in seen_ids:
                        fts_results.append(row_dict)
                        seen_ids.add(row_dict["id"])
            except Exception:
                logger.debug("Subject fallback search failed", exc_info=True)

    return fts_results


def _reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    Each list is a sequence of dicts with at least an 'id' key.
    RRF score for each item = sum over lists of 1/(k + rank + 1).
    Returns fused list sorted by combined RRF score descending.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list):
            item_id = item["id"]
            rrf_score = 1.0 / (k + rank + 1)
            scores[item_id] = scores.get(item_id, 0.0) + rrf_score
            if item_id not in items:
                items[item_id] = item

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = []
    for item_id, score in fused:
        entry = items[item_id].copy()
        entry["rrf_score"] = score
        result.append(entry)

    return result


# ============================================================================
# MAIN RETRIEVAL FUNCTION
# ============================================================================

MAX_RETRIEVAL_LIMIT = 200


def find_similar_memories(
    query: str,
    limit: int = 5,
    category: Optional[str] = None,
    subject: Optional[str] = None,
    origin: Optional[str] = None,
    origin_interface: Optional[str] = None,
) -> list[dict]:
    """
    Find memories using 3-signal retrieval with cross-encoder reranking.

    Pipeline:
    1. Dense vector similarity -> top N candidates
    2. BM25 keyword matching (FTS5) -> top N candidates
    3. Graph connectivity (entity mentions -> subject match) -> top N candidates
    4. RRF fusion -> unified candidate pool
    5. Filter by category/subject/origin (if specified)
    6. Cross-encoder reranking (query-document relevance scoring)
       Fallback: importance-weighted formula if cross-encoder unavailable
    7. Diversity-aware selection (Jaccard dedup)
    8. Record access
    """
    # Hard cap on limit to prevent excessive resource usage
    if limit < 1:
        limit = 1
    limit = min(limit, MAX_RETRIEVAL_LIMIT)

    # Per-signal retrieval depth. Higher than Phase 1's 3x because multi-signal
    # fusion benefits from broader candidate pools — BM25 and graph may surface
    # relevant results at deeper ranks. Cap keeps total candidates manageable
    # (~50-75 unique after RRF dedup).
    RETRIEVAL_DEPTH = max(limit * 5, 25)

    import maasv
    protected = maasv.get_config().protected_categories
    now = datetime.now(timezone.utc)

    with _db() as db:
        # === Signal 1: Dense vector similarity ===
        query_embedding = get_query_embedding(query)
        vector_rows = db.execute("""
            SELECT
                v.id, m.content, m.category, m.subject, m.confidence,
                m.created_at, m.metadata, m.importance, m.access_count,
                m.origin, m.origin_interface,
                distance
            FROM memory_vectors v
            JOIN memories m ON v.id = m.id
            WHERE m.superseded_by IS NULL
            AND v.embedding MATCH ?
            AND k = ?
            ORDER BY distance
        """, (serialize_embedding(query_embedding), RETRIEVAL_DEPTH)).fetchall()
        vector_results = [dict(row) for row in vector_rows]

        # === Signal 2: BM25 keyword matching ===
        bm25_results = _find_memories_by_bm25(db, query, limit=RETRIEVAL_DEPTH)

        # === Signal 3: Graph connectivity ===
        graph_results = _find_memories_by_graph(db, query, limit=RETRIEVAL_DEPTH)

        # === Fusion: Reciprocal Rank Fusion ===
        signals = [vector_results, bm25_results, graph_results]
        # Only include non-empty signals
        active_signals = [s for s in signals if s]

        if not active_signals:
            return []

        if len(active_signals) == 1:
            # Single signal — skip RRF overhead
            candidates = active_signals[0]
        else:
            candidates = _reciprocal_rank_fusion(active_signals, k=60)

        # === Filter by category/subject/origin ===
        if category:
            candidates = [c for c in candidates if c['category'] == category]
        if subject:
            candidates = [c for c in candidates if c.get('subject') and subject.lower() in c['subject'].lower()]
        if origin:
            candidates = [c for c in candidates if c.get('origin') == origin]
        if origin_interface:
            candidates = [c for c in candidates if c.get('origin_interface') == origin_interface]

        # === Reranking ===
        # Try cross-encoder first (best quality). Falls back to importance-weighted
        # formula if cross-encoder is unavailable.
        from maasv.core.reranker import rerank as ce_rerank
        ce_scores = ce_rerank(query, candidates)

        vector_distances = {r['id']: r['distance'] for r in vector_results}
        bm25_ids = {r['id'] for r in bm25_results}
        graph_ids = {r['id'] for r in graph_results}

        # === Importance scoring ===
        # Try learned ranker first; falls back to heuristic formula.
        from maasv.core.learned_ranker import score as learned_score
        lr_result = learned_score(
            candidates, protected, now, vector_distances, bm25_ids, graph_ids
        )
        if lr_result is not None:
            primary, supplementary = lr_result
        else:
            primary, supplementary = _importance_score(
                candidates, protected, now, vector_distances, bm25_ids, graph_ids
            )

        if ce_scores is not None:
            # === Two-stage reranking ===
            # Stage 1: importance scoring (done above).
            # Stage 2: CE reshuffles within top tier only.
            #
            # The MS MARCO cross-encoder prefers short exact matches over
            # informationally rich memories. Pure CE scoring regresses quality
            # because it displaces well-established, high-access memories with
            # semantically precise but shallow matches. Two-stage prevents this:
            # importance determines WHICH memories are candidates, CE only
            # refines the ORDER within that set.

            rerank_size = min(limit * 2, len(primary) + len(supplementary))
            importance_ranked = (primary + supplementary)[:rerank_size]

            # Map CE scores to this subset by memory ID
            ce_score_map = {}
            for mem, score in zip(candidates, ce_scores):
                ce_score_map[mem['id']] = score

            def _sigmoid(x):
                if x >= 0:
                    return 1.0 / (1.0 + math.exp(-x))
                exp_x = math.exp(x)
                return exp_x / (1.0 + exp_x)

            # Min-max normalize importance scores within the rerank window
            imp_scores = [m['_imp_score'] for m in importance_ranked]
            imp_min = min(imp_scores) if imp_scores else 0
            imp_max = max(imp_scores) if imp_scores else 1
            imp_range = imp_max - imp_min if imp_max > imp_min else 1.0

            for mem in importance_ranked:
                ce_raw = ce_score_map.get(mem['id'], 0.0)
                ce_norm = _sigmoid(ce_raw)
                imp_norm = (mem['_imp_score'] - imp_min) / imp_range

                # Importance-dominant blend: CE is a tiebreaker, not the decider.
                # 0.75 importance + 0.25 CE ensures the existing 9/10 baseline
                # is preserved while CE can swap close-ranked candidates.
                mem['_score'] = 0.75 * imp_norm + 0.25 * ce_norm

            importance_ranked.sort(key=lambda m: m['_score'], reverse=True)

            # Append any remaining candidates after the rerank window
            reranked_ids = {m['id'] for m in importance_ranked}
            remainder = [m for m in (primary + supplementary)[rerank_size:]
                         if m['id'] not in reranked_ids]
            scored_pool = importance_ranked + remainder
        else:
            # === Fallback: importance-weighted reranking ===
            # Copy _imp_score to _score for downstream compatibility
            for mem in primary + supplementary:
                mem['_score'] = mem['_imp_score']
            scored_pool = primary + supplementary

        # === Diversity-aware selection (MMR-inspired) ===
        # Greedily select from scored candidates, skipping those too similar to
        # already-selected results. Prevents near-duplicate memories from filling
        # all slots. Threshold adapts: short memories (< 12 words) use 0.5,
        # longer memories use 0.7.
        result = []
        selected_words = []

        for mem in scored_pool:
            if len(result) >= limit:
                break
            mem_words = set(re.findall(r'\w+', mem.get('content', '').lower()))
            threshold = 0.5 if len(mem_words) < 12 else 0.7
            is_diverse = True
            for sw in selected_words:
                if not mem_words or not sw:
                    continue
                intersection = len(mem_words & sw)
                union = len(mem_words | sw)
                jaccard = intersection / union if union > 0 else 0
                smaller = min(len(mem_words), len(sw))
                containment = intersection / smaller if smaller > 0 else 0
                if jaccard > threshold or containment > 0.8:
                    is_diverse = False
                    break
            if is_diverse:
                result.append(mem)
                selected_words.append(mem_words)

        # === Graph slot injection ===
        # If the graph signal found content (via 1-hop expansion) that didn't
        # make it into the top results, inject the graph result that mentions
        # the most graph-connected entity names. This ensures graph-discovered
        # connections (e.g., MyApp->FastAPI) surface even when importance scoring
        # favors high-access vector matches.
        if graph_results and len(result) >= limit:
            result_ids = {m['id'] for m in result}
            result_content = " ".join(m.get('content', '').lower() for m in result)
            graph_only = [m for m in graph_results if m['id'] not in result_ids]

            if graph_only:
                # Get expanded entity names from graph
                expanded_names = _get_graph_expanded_names(db, query)

                if expanded_names:
                    # Filter to entity names not already mentioned in results
                    novel_names = {n for n in expanded_names
                                   if n not in result_content}

                    if novel_names:
                        # Score candidates by how many novel entity names they contain.
                        # Only consider candidates that mention at least one query term
                        # — the injection should bring in relevant novel content, not
                        # random memories that happen to mention many entity names.
                        query_terms = [t.lower() for t in query.split() if len(t) >= 3]
                        best_candidate = None
                        best_score = (0, 0)  # (novel_count, query_term_count)

                        for gm in graph_only:
                            content_lower = gm.get('content', '').lower()
                            query_count = sum(1 for t in query_terms if t in content_lower)
                            if query_terms and query_count == 0:
                                continue  # Skip candidates irrelevant to the query
                            novel_count = sum(1 for n in novel_names if n in content_lower)
                            score = (novel_count, query_count)
                            if score > best_score:
                                best_score = score
                                best_candidate = gm

                        if best_candidate and best_score[0] > 0:
                            result[-1] = best_candidate

        # Clean up internal scoring fields, expose relevance
        for mem in result:
            # Expose relevance from L2 distance on normalized vectors.
            # For unit vectors: L2² = 2 - 2·cos(θ), so cos(θ) = 1 - L2²/2.
            dist = mem.pop('distance', None)
            if dist is not None:
                cosine_sim = 1.0 - (dist * dist) / 2.0
                mem['relevance'] = round(cosine_sim, 4)
            mem.pop('_score', None)
            mem.pop('_imp_score', None)
            mem.pop('rrf_score', None)
            mem.pop('bm25_score', None)
            mem.pop('graph_score', None)

        _record_memory_access(db, [r['id'] for r in result])

        # Log retrieval for learned ranker training data (best-effort)
        try:
            from maasv.core.learned_ranker import log_retrieval
            log_retrieval(
                query=query,
                candidates=candidates,
                returned_ids=[r['id'] for r in result],
                vector_distances=vector_distances,
                bm25_ids=bm25_ids,
                graph_ids=graph_ids,
                protected=protected,
                now=now,
            )
        except Exception:
            pass

    return result


# ============================================================================
# TIERED MEMORY CONTEXT
# ============================================================================

def _get_category_priority() -> dict[str, int]:
    """Get category priority from config."""
    import maasv
    return maasv.get_config().category_priority

_core_memories_cache: list[dict] = []
_cache_timestamp: float = 0
_cache_lock = threading.Lock()
CACHE_TTL = 300  # 5 minutes


def get_core_memories(refresh: bool = False) -> list[dict]:
    """Get core memories (family, identity, preference). Cached for 5 minutes."""
    global _core_memories_cache, _cache_timestamp
    import time

    now = time.time()
    if not refresh and _core_memories_cache and (now - _cache_timestamp) < CACHE_TTL:
        return _core_memories_cache

    with _cache_lock:
        # Double-check after acquiring lock
        now = time.time()
        if not refresh and _core_memories_cache and (now - _cache_timestamp) < CACHE_TTL:
            return _core_memories_cache

        with _db() as db:
            rows = db.execute("""
                SELECT id, content, category, subject, confidence, created_at, importance
                FROM memories
                WHERE superseded_by IS NULL
                AND category IN ('family', 'identity', 'preference')
                ORDER BY
                    CASE category
                        WHEN 'family' THEN 1
                        WHEN 'identity' THEN 2
                        WHEN 'preference' THEN 3
                    END,
                    importance DESC,
                    created_at DESC
            """).fetchall()

        _core_memories_cache = [dict(row) for row in rows]
        _cache_timestamp = now

        return _core_memories_cache


def get_tiered_memory_context(
    query: str = None,
    core_limit: int = 10,
    relevant_limit: int = 5,
    use_semantic: bool = False
) -> str:
    """
    Smart memory retrieval with tiered approach for low latency.

    Tier 1: Core memories (family, identity, prefs) - cached, instant
    Tier 2: Query-relevant via FTS keyword search - fast (~2ms)
    Tier 3: Semantic search - slow (~400ms), only if use_semantic=True
    """
    seen_ids = set()
    memories = []

    # Tier 1: Always include core memories (cached)
    core = get_core_memories()[:core_limit]
    for mem in core:
        if mem['id'] not in seen_ids:
            memories.append(mem)
            seen_ids.add(mem['id'])

    # Tier 2: Add query-relevant memories via FTS (fast)
    if query and len(memories) < core_limit + relevant_limit:
        try:
            tokens = [_sanitize_fts_input(w) for w in query.split()[:5]]
            tokens = [t for t in tokens if t]
            keywords = ' OR '.join(tokens) if tokens else None
            fts_results = search_fts(keywords, limit=relevant_limit) if keywords else []
            for mem in fts_results:
                if mem['id'] not in seen_ids:
                    memories.append(mem)
                    seen_ids.add(mem['id'])
                    if len(memories) >= core_limit + relevant_limit:
                        break
        except Exception:
            logger.debug("FTS keyword search failed in tiered context", exc_info=True)

    # Tier 3: Semantic search as fallback (SLOW)
    if use_semantic and query and len(memories) < core_limit + relevant_limit:
        remaining = (core_limit + relevant_limit) - len(memories)
        semantic_results = find_similar_memories(query, limit=remaining)
        for mem in semantic_results:
            if mem['id'] not in seen_ids:
                memories.append(mem)
                seen_ids.add(mem['id'])

    # Fill remaining slots with other memories by priority.
    # Fetch a bounded set (not all 5K+) ordered by importance, then sort by category priority in Python.
    remaining_slots = (core_limit + relevant_limit) - len(memories)
    if remaining_slots > 0:
        category_priority = _get_category_priority()
        # Over-fetch to account for seen_ids filtering, but cap at a reasonable limit
        fetch_limit = remaining_slots * 3

        with _db() as db:
            filler_rows = db.execute("""
                SELECT id, content, category, subject, confidence, created_at, importance
                FROM memories
                WHERE superseded_by IS NULL
                ORDER BY importance DESC, created_at DESC
                LIMIT ?
            """, (fetch_limit,)).fetchall()

        filler_mems = [dict(row) for row in filler_rows if row['id'] not in seen_ids]
        filler_mems.sort(key=lambda m: category_priority.get(m['category'], 99))

        for mem in filler_mems:
            memories.append(mem)
            seen_ids.add(mem['id'])
            if len(memories) >= core_limit + relevant_limit:
                break

    if not memories:
        return ""

    lines = ["Remembered facts:"]
    for mem in memories:
        subject_str = f"[{mem['subject']}] " if mem.get('subject') else ""
        lines.append(f"- {subject_str}{mem['content']}")

    return "\n".join(lines)


# ============================================================================
# SIMPLE FTS SEARCH (used by tiered context and externally)
# ============================================================================

def search_fts(query: str, limit: int = 10, category: Optional[str] = None) -> list[dict]:
    """Full-text search across memories, optionally filtered by category."""
    import sqlite3

    query = _sanitize_fts_input(query)
    if not query:
        return []

    with _db() as db:
        try:
            if category:
                rows = db.execute("""
                    SELECT
                        m.id, m.content, m.category, m.subject,
                        m.confidence, m.created_at
                    FROM memories_fts f
                    JOIN memories m ON f.rowid = m.rowid
                    WHERE memories_fts MATCH ?
                    AND m.superseded_by IS NULL
                    AND m.category = ?
                    ORDER BY rank
                    LIMIT ?
                """, (query, category, limit)).fetchall()
            else:
                rows = db.execute("""
                    SELECT
                        m.id, m.content, m.category, m.subject,
                        m.confidence, m.created_at
                    FROM memories_fts f
                    JOIN memories m ON f.rowid = m.rowid
                    WHERE memories_fts MATCH ?
                    AND m.superseded_by IS NULL
                    ORDER BY rank
                    LIMIT ?
                """, (query, limit)).fetchall()
        except sqlite3.OperationalError:
            logger.debug("FTS5 query failed (bad syntax?): %s", query, exc_info=True)
            return []

    return [dict(row) for row in rows]


def find_by_subject(subject: str, active_only: bool = True) -> list[dict]:
    """Find all memories about a specific subject."""
    escaped = _escape_like(subject)
    query = """
        SELECT id, content, category, subject, confidence, created_at, metadata
        FROM memories
        WHERE subject LIKE ? ESCAPE '\\'
    """
    if active_only:
        query += " AND superseded_by IS NULL"
    query += " ORDER BY created_at DESC"

    with _db() as db:
        rows = db.execute(query, (f"%{escaped}%",)).fetchall()

    return [dict(row) for row in rows]

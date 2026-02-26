"""
maasv MCP Server

Exposes the maasv cognition layer to any MCP client (Claude Desktop, Claude Code,
ChatGPT, etc.) via the Model Context Protocol.

19 tools across 4 domains:
- Memory (6): bootstrap, query, store, facts, forget, supersede
- Graph (9): entity create/get/search/find_or_create/profile,
             relationship add/expire/update, graph query
- Wisdom (4): log, outcome, search, feedback
- Extraction (1): extract entities/relationships from text

Run modes:
- STDIO (local):  python -m maasv.mcp_server
- HTTP (remote):  MAASV_TRANSPORT=http python -m maasv.mcp_server

Environment variables (all prefixed MAASV_):
- MAASV_TRANSPORT: "stdio" (default) or "http"
- MAASV_PORT: Port for HTTP transport (default 8000)
- MAASV_HOST: Host for HTTP transport (default 127.0.0.1)
- MAASV_AUTH_TOKEN: API token for HTTP auth (required for HTTP)
- MAASV_DB_PATH: Path to SQLite database (default "maasv.db")
- MAASV_EMBED_PROVIDER: "ollama" (default), "voyage", or "openai"
- MAASV_EMBED_MODEL: Embedding model name
- MAASV_LLM_PROVIDER: "anthropic" or "openai"
- MAASV_LLM_API_KEY: API key for LLM provider
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("maasv.mcp")

# Session-level client tracking for origin_interface provenance
_session_client: str = "unknown"


def _create_operation_llm(settings, operation: str):
    """Create a per-operation LLM provider if configured, else return None."""
    from maasv.server.providers import create_llm

    provider = getattr(settings, f"{operation}_llm_provider", "")
    api_key = getattr(settings, f"{operation}_llm_api_key", "") or settings.llm_api_key
    model = getattr(settings, f"{operation}_llm_model", "") or settings.llm_model
    if not provider:
        return None
    return create_llm(provider, api_key, model)


def _init_maasv(settings) -> None:
    """Initialize the maasv cognition layer from MCP settings."""
    from maasv import init
    from maasv.config import MaasvConfig
    from maasv.server.providers import create_embed, create_llm

    # Per-operation LLM overrides
    extraction_llm = _create_operation_llm(settings, "extraction")
    inference_llm = _create_operation_llm(settings, "inference")
    review_llm = _create_operation_llm(settings, "review")

    config = MaasvConfig(
        db_path=Path(settings.db_path).resolve(),
        embed_dims=settings.embed_dims,
        extraction_model=getattr(settings, "extraction_llm_model", "") or settings.llm_model,
        inference_model=getattr(settings, "inference_llm_model", "") or settings.llm_model,
        review_model=getattr(settings, "review_llm_model", "") or settings.llm_model,
        extraction_llm=extraction_llm,
        inference_llm=inference_llm,
        review_llm=review_llm,
        protected_categories=settings.protected_categories_set,
        stale_days=settings.stale_days,
        similarity_threshold=settings.similarity_threshold,
        cross_encoder_enabled=settings.cross_encoder_enabled,
    )

    # Create embedding provider (required)
    embed = create_embed(
        provider=settings.embed_provider,
        api_key=settings.embed_api_key,
        model=settings.embed_model,
        base_url=settings.embed_base_url,
        dims=settings.embed_dims,
    )

    # Create LLM provider (optional — only needed for extraction)
    llm = None
    if settings.llm_api_key:
        llm = create_llm(
            provider=settings.llm_provider,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )

    init(config, llm=llm, embed=embed)


def create_server(settings=None) -> FastMCP:
    """Create and configure the MCP server with all 19 tools.

    Args:
        settings: MCPSettings instance. If None, creates from env vars.

    Returns:
        Configured FastMCP server instance.
    """
    if settings is None:
        from maasv.mcp_server.config import MCPSettings

        settings = MCPSettings()

    # Initialize maasv
    _init_maasv(settings)

    # Transport security for HTTP
    transport_security = None
    if settings.transport == "http":
        from mcp.server.fastmcp.server import TransportSecuritySettings

        transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=False,
            allowed_hosts=["*"],
            allowed_origins=["*"],
        )

    mcp = FastMCP(
        name="maasv",
        host=settings.host,
        port=settings.port,
        transport_security=transport_security,
        instructions=(
            "maasv is a cognition layer providing persistent memory, a knowledge graph, "
            "and a wisdom/learning system. Call maasv_memory_bootstrap at session start "
            "to load context. Use maasv_memory_query to search memories, "
            "maasv_memory_store to save new information, and the graph/wisdom tools "
            "to manage structured knowledge and experiential learning."
        ),
    )

    # Keep a reference for HTTP auth setup
    mcp._maasv_settings = settings

    # ========================================================================
    # MEMORY TOOLS (6)
    # ========================================================================

    @mcp.tool(annotations={"readOnlyHint": True})
    def maasv_memory_bootstrap(
        client: str = "unknown",
        query: Optional[str] = None,
    ) -> dict:
        """
        Load session context: core memories, recent activity, and query-relevant results.

        Call this FIRST at the start of every session. Returns tiered memory context
        (identity, preferences, recent decisions) plus optionally query-relevant memories.

        Args:
            client: Client identifier (e.g., "claude-desktop", "claude-code")
            query: Optional initial query to load relevant context for

        Returns:
            Tiered context string, active facts, and optional query results.
        """
        global _session_client
        _session_client = client

        from maasv.core import get_all_active, get_tiered_memory_context

        # Tier 1+2: core memories + FTS results for query
        context = get_tiered_memory_context(
            query=query,
            core_limit=10,
            relevant_limit=5,
            use_semantic=False,
        )

        # Active facts summary
        all_active = get_all_active()
        category_counts = {}
        for m in all_active:
            cat = m.get("category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        result = {
            "context": context,
            "total_memories": len(all_active),
            "categories": category_counts,
            "client": client,
            "timestamp": datetime.now().isoformat(),
        }

        # If query provided, add semantic results
        if query:
            from maasv.core import find_similar_memories

            relevant = find_similar_memories(query=query, limit=5)
            result["query_results"] = {
                "query": query,
                "count": len(relevant),
                "results": relevant,
            }

        return result

    @mcp.tool(annotations={"readOnlyHint": True})
    def maasv_memory_query(
        query: str,
        limit: int = 10,
        category: Optional[str] = None,
        use_semantic: bool = True,
        min_relevance: float = 0.3,
        origin: Optional[str] = None,
    ) -> dict:
        """
        Search memories using semantic (embedding) or keyword (FTS) search.

        Args:
            query: Search query (natural language for semantic, keywords for FTS)
            limit: Maximum results (default 10)
            category: Optional filter by category (identity, family, preference, project, decision, etc.)
            use_semantic: If True, use embedding search. If False, use keyword/FTS search.
            min_relevance: Minimum relevance score (0-1) for semantic results. Default 0.3.
            origin: Optional filter by origin system

        Returns:
            List of matching memories with content, category, subject, and relevance score.
        """
        from maasv.core import find_similar_memories, search_fts

        if use_semantic:
            results = find_similar_memories(
                query=query,
                limit=limit,
                category=category,
                origin=origin,
            )
            if min_relevance > 0:
                results = [r for r in results if r.get("relevance", 1.0) >= min_relevance]
        else:
            results = search_fts(query, limit=limit, category=category)

        return {
            "query": query,
            "count": len(results),
            "results": results,
        }

    @mcp.tool(annotations={"destructiveHint": False})
    def maasv_memory_store(
        content: str,
        category: str,
        subject: Optional[str] = None,
        source: str = "mcp",
        confidence: float = 0.9,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Store a new memory with automatic deduplication.

        If a near-duplicate exists (>95% similarity), returns the existing memory ID
        instead of creating a duplicate.

        Args:
            content: The memory content (max 50KB)
            category: Category (identity, family, preference, project, decision, learning, person, event, etc.)
            subject: Optional subject/topic this memory is about
            source: Source label (default "mcp")
            confidence: Confidence score 0.0-1.0 (default 0.9)
            metadata: Optional structured metadata dict

        Returns:
            Memory ID (new or existing if deduplicated).
        """
        from maasv.core import store_memory

        try:
            mem_id = store_memory(
                content=content,
                category=category,
                subject=subject,
                source=source,
                confidence=confidence,
                metadata=metadata,
                origin="mcp",
                origin_interface=_session_client,
            )
            return {"status": "stored", "memory_id": mem_id}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @mcp.tool(annotations={"readOnlyHint": True})
    def maasv_memory_facts(
        category: str,
        limit: int = 20,
        subject: Optional[str] = None,
    ) -> dict:
        """
        Get facts by category (fast, no embedding lookup).

        Args:
            category: Category to retrieve (identity, family, preference, project, decision, learning, etc.)
            limit: Maximum results (default 20)
            subject: Optional filter by subject

        Returns:
            List of facts in the category.
        """
        from maasv.core import get_all_active

        memories = get_all_active(category=category)

        if subject:
            memories = [m for m in memories if subject.lower() in (m.get("subject") or "").lower()]

        memories = memories[:limit]

        return {
            "category": category,
            "subject_filter": subject,
            "count": len(memories),
            "facts": [
                {
                    "id": m["id"],
                    "content": m["content"],
                    "subject": m.get("subject"),
                    "confidence": m.get("confidence", 1.0),
                    "created_at": m.get("created_at"),
                }
                for m in memories
            ],
        }

    @mcp.tool(annotations={"destructiveHint": True})
    def maasv_memory_forget(
        memory_id: Optional[str] = None,
        query: Optional[str] = None,
        confirm: bool = False,
    ) -> dict:
        """
        Delete memories (for corrections or privacy).

        Provide a specific memory_id OR a query to find memories to delete.
        Always requires confirm=True to actually delete. With confirm=False,
        shows a preview of what would be deleted.

        Args:
            memory_id: Specific memory ID to delete
            query: Search query to find memories to delete (shows matches first)
            confirm: Must be True to actually delete. False = preview only.

        Returns:
            Preview of affected memories, or deletion confirmation.
        """
        from maasv.core import delete_memory, find_similar_memories, get_all_active

        affected = []

        if memory_id:
            memories = get_all_active()
            target = next((m for m in memories if m["id"] == memory_id), None)
            if target:
                affected.append(
                    {
                        "id": target["id"],
                        "content": target["content"],
                        "category": target["category"],
                    }
                )
        elif query:
            results = find_similar_memories(query, limit=5)
            affected = [{"id": r["id"], "content": r["content"], "category": r["category"]} for r in results]
        else:
            return {"status": "error", "message": "Must provide either memory_id or query"}

        if not affected:
            return {"status": "not_found", "message": "No matching memories found"}

        if not confirm:
            return {
                "status": "preview",
                "message": "Set confirm=True to delete these memories",
                "would_delete": affected,
            }

        deleted = []
        protected = []
        for mem in affected:
            try:
                if delete_memory(mem["id"]):
                    deleted.append(mem)
            except ValueError as e:
                protected.append({"id": mem["id"], "reason": str(e)})

        result = {
            "status": "deleted",
            "count": len(deleted),
            "deleted": deleted,
        }
        if protected:
            result["protected"] = protected
            result["message"] = (
                f"{len(protected)} memory(ies) in protected categories were not deleted. "
                "Use force=True via the API to override."
            )
        return result

    @mcp.tool(annotations={"destructiveHint": False})
    def maasv_memory_supersede(
        old_id: str,
        new_content: str,
        source: str = "correction",
    ) -> dict:
        """
        Replace an outdated memory with corrected information.

        Marks the old memory as superseded and creates a new one.
        The old memory is preserved for history but won't appear in active queries.

        Args:
            old_id: ID of the memory to supersede
            new_content: The corrected/updated content
            source: Source label (default "correction")

        Returns:
            New memory ID and superseded old ID.
        """
        from maasv.core import supersede_memory

        try:
            new_id = supersede_memory(
                old_id=old_id,
                new_content=new_content,
                source=source,
                origin="mcp",
                origin_interface=_session_client,
            )
            return {
                "status": "superseded",
                "old_memory_id": old_id,
                "new_memory_id": new_id,
            }
        except ValueError as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ========================================================================
    # GRAPH TOOLS (9)
    # ========================================================================

    @mcp.tool(annotations={"destructiveHint": False})
    def maasv_graph_entity_create(
        name: str,
        entity_type: str,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Create a new entity in the knowledge graph.

        Args:
            name: Display name (e.g., "Alice", "Project X", "New York")
            entity_type: Type of entity (person, place, project, event, concept)
            metadata: Optional additional data

        Returns:
            Created entity with ID.
        """
        from maasv.core import create_entity

        try:
            entity_id = create_entity(name, entity_type, metadata=metadata)
            return {
                "status": "created",
                "entity": {
                    "id": entity_id,
                    "name": name,
                    "entity_type": entity_type,
                    "metadata": metadata,
                },
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @mcp.tool(annotations={"readOnlyHint": True})
    def maasv_graph_entity_get(entity_id: str) -> dict:
        """
        Get an entity by ID.

        Args:
            entity_id: The entity ID (starts with "ent_")

        Returns:
            Entity details or not_found status.
        """
        from maasv.core import get_entity

        entity = get_entity(entity_id)
        if entity:
            return {"status": "found", "entity": entity}
        return {"status": "not_found", "entity_id": entity_id}

    @mcp.tool(annotations={"readOnlyHint": True})
    def maasv_graph_entity_search(
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 10,
    ) -> dict:
        """
        Search for entities by name using full-text search.

        Args:
            query: Search query (name or partial name)
            entity_type: Optional filter (person, place, project, event, concept)
            limit: Maximum results (default 10)

        Returns:
            List of matching entities.
        """
        from maasv.core import search_entities

        results = search_entities(query, entity_type, limit)
        return {
            "query": query,
            "type_filter": entity_type,
            "count": len(results),
            "entities": results,
        }

    @mcp.tool(annotations={"destructiveHint": False})
    def maasv_graph_entity_find_or_create(
        name: str,
        entity_type: str,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Find an existing entity by name, or create it if it doesn't exist.

        Uses canonical name matching (case-insensitive, normalized).

        Args:
            name: Entity name
            entity_type: Type (person, place, project, event, concept)
            metadata: Optional metadata (only used if creating new)

        Returns:
            Entity ID and whether it was newly created.
        """
        from maasv.core import create_entity, find_entity_by_name

        existing = find_entity_by_name(name, entity_type)
        if existing:
            return {"status": "found", "created": False, "entity": existing}

        entity_id = create_entity(name, entity_type, metadata=metadata)
        return {
            "status": "created",
            "created": True,
            "entity": {
                "id": entity_id,
                "name": name,
                "entity_type": entity_type,
                "metadata": metadata,
            },
        }

    @mcp.tool(annotations={"readOnlyHint": True})
    def maasv_graph_entity_profile(entity_id: str) -> dict:
        """
        Get a complete profile for an entity including all current relationships.

        Returns the entity with relationships organized by predicate and a list
        of related entities. Use this to build full context about a person or thing.

        Args:
            entity_id: Entity ID to get profile for

        Returns:
            Entity with nested relationships and related entities.
        """
        from maasv.core import get_entity_profile

        profile = get_entity_profile(entity_id)
        if not profile:
            return {"status": "not_found", "entity_id": entity_id}
        return {"status": "found", "profile": profile}

    @mcp.tool(annotations={"destructiveHint": False})
    def maasv_graph_relationship_add(
        subject_id: str,
        predicate: str,
        object_id: Optional[str] = None,
        object_value: Optional[str] = None,
        source: Optional[str] = None,
    ) -> dict:
        """
        Add a relationship between entities (or entity to a literal value).

        Relationships are temporal — they have a valid_from timestamp and can be
        expired later when information changes. Duplicates are detected automatically.

        Args:
            subject_id: Source entity ID
            predicate: Relationship type. Common predicates:
                - Family: married_to, parent_of, child_of, sibling_of, friend_of
                - Work: works_on, owns, manages, created, works_at
                - Location: lives_in, located_in, visited
                - Tech: uses_tech, built_with, depends_on
                - Attributes: has_email, has_phone, has_birthday, has_age
                - Causal: caused_by, led_to, resulted_in
            object_id: Target entity ID (for entity-to-entity relationships)
            object_value: Literal value (for entity-to-value, e.g., an email address)
            source: Where this relationship came from

        Returns:
            Created relationship with ID.
        """
        if object_id is None and object_value is None:
            return {"status": "error", "message": "Must provide either object_id or object_value"}

        from maasv.core import add_relationship

        try:
            rel_id = add_relationship(
                subject_id=subject_id,
                predicate=predicate,
                object_id=object_id,
                object_value=object_value,
                source=source,
                origin="mcp",
                origin_interface=_session_client,
            )
            return {
                "status": "created",
                "relationship": {
                    "id": rel_id,
                    "subject_id": subject_id,
                    "predicate": predicate,
                    "object_id": object_id,
                    "object_value": object_value,
                },
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @mcp.tool(annotations={"destructiveHint": False})
    def maasv_graph_relationship_expire(relationship_id: str) -> dict:
        """
        Mark a relationship as expired (no longer current).

        The relationship is NOT deleted — it becomes historical.
        Use this when information changes. The old value remains queryable
        with include_expired=True.

        Args:
            relationship_id: ID of relationship to expire (starts with "rel_")

        Returns:
            Status of the expiration.
        """
        from maasv.core import expire_relationship

        success = expire_relationship(relationship_id)
        if success:
            return {"status": "expired", "relationship_id": relationship_id}
        return {"status": "not_found", "relationship_id": relationship_id}

    @mcp.tool(annotations={"destructiveHint": False})
    def maasv_graph_relationship_update(
        subject_id: str,
        predicate: str,
        new_value: str,
        source: Optional[str] = None,
    ) -> dict:
        """
        Update a relationship value by expiring the old one and creating a new one.

        Preserves history — the old value is still queryable.
        Use this when a value changes (e.g., email, phone, status).

        Args:
            subject_id: Entity ID
            predicate: Relationship type (e.g., 'has_email', 'has_status')
            new_value: The new value
            source: Source of the update

        Returns:
            Old and new relationship IDs.
        """
        from maasv.core import update_relationship_value

        try:
            old_id, new_id = update_relationship_value(
                subject_id=subject_id,
                predicate=predicate,
                new_value=new_value,
                source=source,
                origin="mcp",
                origin_interface=_session_client,
            )
            return {
                "status": "updated",
                "expired_relationship_id": old_id,
                "new_relationship_id": new_id,
                "predicate": predicate,
                "new_value": new_value,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @mcp.tool(annotations={"readOnlyHint": True})
    def maasv_graph_query(
        subject_type: Optional[str] = None,
        predicate: Optional[str] = None,
        object_type: Optional[str] = None,
        include_expired: bool = False,
        limit: int = 50,
    ) -> dict:
        """
        Query the knowledge graph with pattern matching.

        Find relationships matching a pattern. All parameters are optional filters.

        Examples:
            - All family relationships: predicate="parent_of"
            - All project work: subject_type="person", predicate="works_on", object_type="project"

        Args:
            subject_type: Filter by subject entity type
            predicate: Filter by relationship type
            object_type: Filter by object entity type
            include_expired: Include historical (expired) relationships
            limit: Maximum results (default 50)

        Returns:
            List of matching relationships with entity details.
        """
        from maasv.core import graph_query as _graph_query

        results = _graph_query(
            subject_type=subject_type,
            predicate=predicate,
            object_type=object_type,
            include_expired=include_expired,
            limit=limit,
        )
        return {
            "pattern": {
                "subject_type": subject_type,
                "predicate": predicate,
                "object_type": object_type,
            },
            "include_expired": include_expired,
            "count": len(results),
            "results": results,
        }

    # ========================================================================
    # WISDOM TOOLS (4)
    # ========================================================================

    @mcp.tool(annotations={"destructiveHint": False})
    def maasv_wisdom_log(
        action_type: str,
        reasoning: str,
        action_data: Optional[dict] = None,
        trigger: Optional[str] = None,
        context: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> dict:
        """
        Log reasoning before taking an action. Creates a wisdom entry for future learning.

        Use this when making a significant decision — the reasoning and outcome
        can inform future similar decisions.

        Args:
            action_type: Type of action (e.g., "architecture_decision", "tool_selection",
                "debugging_approach", "config_change", "adam_preference")
            reasoning: Why this action was chosen — the thought process
            action_data: Optional structured data about the action
            trigger: What prompted this action (e.g., "user_request", "error_encountered")
            context: Additional context (project, file, etc.)
            tags: Optional tags for categorization

        Returns:
            Wisdom entry ID for recording outcome later.
        """
        from maasv.core import log_reasoning

        try:
            wisdom_id = log_reasoning(
                action_type=action_type,
                reasoning=reasoning,
                action_data=action_data,
                trigger=trigger,
                context=context,
                tags=tags,
            )
            return {"status": "logged", "wisdom_id": wisdom_id}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @mcp.tool(annotations={"destructiveHint": False})
    def maasv_wisdom_outcome(
        wisdom_id: str,
        outcome: str,
        details: Optional[str] = None,
    ) -> dict:
        """
        Record the outcome of a previously logged action.

        Call this after an action completes to close the feedback loop.

        Args:
            wisdom_id: ID from maasv_wisdom_log
            outcome: "success" or "failure"
            details: What happened — especially useful for failures

        Returns:
            Status of the update.
        """
        from maasv.core import record_outcome

        try:
            success = record_outcome(wisdom_id, outcome=outcome, details=details)
            if success:
                return {"status": "recorded", "wisdom_id": wisdom_id, "outcome": outcome}
            return {"status": "not_found", "wisdom_id": wisdom_id}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @mcp.tool(annotations={"readOnlyHint": True})
    def maasv_wisdom_search(
        query: Optional[str] = None,
        action_type: Optional[str] = None,
        limit: int = 10,
    ) -> dict:
        """
        Search past wisdom entries — learn from previous decisions and outcomes.

        Use this before making a similar decision to see what worked/failed before.

        Args:
            query: Free-text search across reasoning and context
            action_type: Filter by action type (e.g., "architecture_decision")
            limit: Maximum results (default 10)

        Returns:
            List of wisdom entries with reasoning, outcomes, and feedback scores.
        """
        from maasv.core import get_relevant_wisdom
        from maasv.core import search_wisdom as _search_wisdom

        try:
            if action_type:
                results = get_relevant_wisdom(action_type=action_type, limit=limit, include_unrated=True)
            elif query:
                results = _search_wisdom(query=query, limit=limit)
            else:
                from maasv.core.wisdom import get_recent_wisdom

                results = get_recent_wisdom(limit=limit)

            entries = []
            for r in results:
                if hasattr(r, "__dict__"):
                    entry = {k: v for k, v in r.__dict__.items() if not k.startswith("_")}
                elif isinstance(r, dict):
                    entry = r
                else:
                    entry = dict(r)
                entries.append(entry)

            return {
                "query": query,
                "action_type": action_type,
                "count": len(entries),
                "entries": entries,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @mcp.tool(annotations={"destructiveHint": False})
    def maasv_wisdom_feedback(
        wisdom_id: str,
        score: int,
        notes: Optional[str] = None,
    ) -> dict:
        """
        Add feedback to a wisdom entry (1-5 rating).

        Use this to rate whether a past decision was good or bad.
        This data trains the system to make better decisions over time.

        Args:
            wisdom_id: ID of the wisdom entry
            score: Rating from 1 (terrible) to 5 (excellent)
            notes: Optional explanation of the rating

        Returns:
            Status of the feedback.
        """
        from maasv.core import add_feedback

        try:
            success = add_feedback(wisdom_id, score=score, notes=notes)
            if success:
                return {"status": "recorded", "wisdom_id": wisdom_id, "score": score}
            return {"status": "not_found", "wisdom_id": wisdom_id}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ========================================================================
    # EXTRACTION TOOL (1)
    # ========================================================================

    @mcp.tool(annotations={"destructiveHint": False, "openWorldHint": True})
    def maasv_extract(
        text: str,
        source: Optional[str] = None,
    ) -> dict:
        """
        Extract entities and relationships from text and add them to the knowledge graph.

        Uses an LLM (requires LLM provider configured) to identify people, places,
        projects, and their relationships from conversation summaries, meeting notes,
        or any text containing structured facts.

        Runs synchronously (1-3 seconds via Claude Haiku) and returns what was created.

        Args:
            text: Text to extract entities and relationships from (max 4000 chars)
            source: Optional source label (default "mcp-extract")

        Returns:
            Created entities and relationships, or error if LLM not configured.
        """
        if not text or not text.strip():
            return {"status": "error", "message": "Text is required"}

        import maasv

        try:
            llm = maasv.get_llm()
        except Exception:
            llm = None

        if llm is None:
            return {
                "status": "error",
                "message": "LLM provider not configured. Set MAASV_LLM_PROVIDER and MAASV_LLM_API_KEY.",
            }

        from maasv.extraction import extract_and_store_entities

        src = source or "mcp-extract"
        try:
            result = extract_and_store_entities(text[:4000], topic=src)
            storage = result.get("storage", {})
            if not storage.get("entities_created") and not storage.get("relationships_created"):
                return {
                    "status": "empty",
                    "message": "No entities or relationships found in text",
                }
            return {"status": "success", **storage}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return mcp


def run() -> None:
    """Entry point for the maasv-mcp script and python -m maasv.mcp_server."""
    from maasv.mcp_server.config import MCPSettings

    settings = MCPSettings()
    mcp = create_server(settings)

    logger.info("Starting maasv MCP server...")
    logger.info("Transport: %s", settings.transport)

    if settings.transport == "http":
        if not settings.auth_token:
            print("[maasv-mcp] ERROR: MAASV_AUTH_TOKEN required for HTTP transport", file=sys.stderr)
            sys.exit(1)

        import hmac

        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import JSONResponse

        starlette_app = mcp.streamable_http_app()

        from starlette.routing import Route

        async def health_check(request):
            return JSONResponse({"status": "healthy"})

        starlette_app.routes.append(Route("/health", health_check))

        auth_token = settings.auth_token

        class APIKeyMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                if request.url.path == "/health":
                    return await call_next(request)
                provided_key = request.headers.get("X-API-Key") or ""
                if not hmac.compare_digest(provided_key, auth_token):
                    return JSONResponse({"error": "Invalid or missing API key"}, status_code=401)
                return await call_next(request)

        starlette_app.add_middleware(APIKeyMiddleware)

        logger.info("Host: %s:%d", settings.host, settings.port)
        logger.info("Auth: enabled (X-API-Key header)")

        import anyio
        import uvicorn

        config = uvicorn.Config(
            starlette_app,
            host=settings.host,
            port=settings.port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        anyio.run(server.serve)
    else:
        mcp.run()

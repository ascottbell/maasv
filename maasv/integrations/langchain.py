"""
LangChain/LangGraph integration for maasv.

Thin adapter layer exposing maasv's cognition engine as LangChain-compatible
tools and retrievers. Requires `langchain-core` to be installed:

    pip install maasv[langchain]

Usage:
    import maasv
    from maasv.config import MaasvConfig
    from maasv.integrations.langchain import (
        MaasvStoreMemoryTool,
        MaasvFindMemoriesTool,
        MaasvExtractEntitiesTool,
        MaasvLogWisdomTool,
        MaasvRetriever,
    )

    maasv.init(config=MaasvConfig(db_path="data/memory.db"), embed="ollama")

    # Use as LangChain tools
    tools = [MaasvStoreMemoryTool(), MaasvFindMemoriesTool()]

    # Use as a retriever
    retriever = MaasvRetriever(limit=5)
"""

from __future__ import annotations

from typing import Any, Optional, Type

try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun, CallbackManagerForToolRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel, Field
except ImportError as e:
    raise ImportError(
        "LangChain integration requires langchain-core. Install it with: pip install maasv[langchain]"
    ) from e


# ============================================================================
# TOOL INPUT SCHEMAS
# ============================================================================


class StoreMemoryInput(BaseModel):
    """Input for storing a memory."""

    content: str = Field(description="The fact or memory to store")
    category: str = Field(description="Type of memory (e.g., family, preference, project, decision)")
    subject: Optional[str] = Field(default=None, description="Who/what this is about")
    source: str = Field(default="langchain", description="Where this memory came from")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    metadata: Optional[dict] = Field(default=None, description="Additional structured data")


class FindMemoriesInput(BaseModel):
    """Input for finding similar memories."""

    query: str = Field(description="Natural language search query")
    limit: int = Field(default=5, ge=1, le=50, description="Maximum results to return")
    category: Optional[str] = Field(default=None, description="Filter by category")
    subject: Optional[str] = Field(default=None, description="Filter by subject")


class ExtractEntitiesInput(BaseModel):
    """Input for extracting entities from text."""

    summary: str = Field(description="Text to extract entities and relationships from")
    topic: str = Field(default="", description="Topic hint for extraction context")


class LogWisdomInput(BaseModel):
    """Input for logging experiential wisdom."""

    action_type: str = Field(description="Type of action (e.g., debugging_resolution, architecture_decision)")
    reasoning: str = Field(description="Reasoning before taking the action")
    action_data: Optional[dict] = Field(default=None, description="Structured data about the action")
    trigger: Optional[str] = Field(default=None, description="What triggered this action")
    context: Optional[str] = Field(default=None, description="Additional context")
    tags: Optional[list[str]] = Field(default=None, description="Tags for categorization")


class SearchWisdomInput(BaseModel):
    """Input for searching wisdom entries."""

    query: str = Field(description="Search query")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum results")


# ============================================================================
# LANGCHAIN TOOLS
# ============================================================================


class MaasvStoreMemoryTool(BaseTool):
    """LangChain tool for storing memories in maasv."""

    name: str = "maasv_store_memory"
    description: str = (
        "Store a fact or memory in the maasv knowledge base. "
        "Use this when you learn something worth remembering about the user, "
        "a project, a decision, or any persistent fact."
    )
    args_schema: Type[BaseModel] = StoreMemoryInput

    def _run(
        self,
        content: str,
        category: str,
        subject: Optional[str] = None,
        source: str = "langchain",
        confidence: float = 1.0,
        metadata: Optional[dict] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        from maasv.core.store import store_memory

        memory_id = store_memory(
            content=content,
            category=category,
            subject=subject,
            source=source,
            confidence=confidence,
            metadata=metadata,
        )
        return f"Stored memory {memory_id}"


class MaasvFindMemoriesTool(BaseTool):
    """LangChain tool for finding relevant memories in maasv."""

    name: str = "maasv_find_memories"
    description: str = (
        "Search the maasv knowledge base for relevant memories. "
        "Uses 3-signal retrieval (vector similarity, BM25 keyword matching, "
        "and knowledge graph traversal) fused with learned ranking."
    )
    args_schema: Type[BaseModel] = FindMemoriesInput

    def _run(
        self,
        query: str,
        limit: int = 5,
        category: Optional[str] = None,
        subject: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        from maasv.core.retrieval import find_similar_memories

        results = find_similar_memories(
            query=query,
            limit=limit,
            category=category,
            subject=subject,
        )

        if not results:
            return "No relevant memories found."

        lines = []
        for mem in results:
            line = f"- [{mem.get('category', 'unknown')}] {mem['content']}"
            if mem.get("subject"):
                line += f" (subject: {mem['subject']})"
            lines.append(line)
        return "\n".join(lines)


class MaasvExtractEntitiesTool(BaseTool):
    """LangChain tool for extracting entities from text into the knowledge graph."""

    name: str = "maasv_extract_entities"
    description: str = (
        "Extract entities and relationships from text and store them "
        "in the maasv knowledge graph. Requires an LLM provider to be "
        "configured in maasv."
    )
    args_schema: Type[BaseModel] = ExtractEntitiesInput

    def _run(
        self,
        summary: str,
        topic: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        from maasv.extraction.entity_extraction import extract_and_store_entities

        result = extract_and_store_entities(summary=summary, topic=topic)

        status = result.get("status", "unknown")
        if status != "success":
            return f"Extraction failed: {result.get('error', status)}"

        stats = result.get("store_stats", result)
        entities = stats.get("entities_created", 0)
        relationships = stats.get("relationships_created", 0)
        return f"Extracted {entities} entities and {relationships} relationships"


class MaasvLogWisdomTool(BaseTool):
    """LangChain tool for logging experiential wisdom."""

    name: str = "maasv_log_wisdom"
    description: str = (
        "Log reasoning before taking an action. Captures what you're about to do "
        "and why, so maasv can track outcomes and learn from experience. "
        "Returns a wisdom ID for later feedback."
    )
    args_schema: Type[BaseModel] = LogWisdomInput

    def _run(
        self,
        action_type: str,
        reasoning: str,
        action_data: Optional[dict] = None,
        trigger: Optional[str] = None,
        context: Optional[str] = None,
        tags: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        from maasv.core.wisdom import log_reasoning

        wisdom_id = log_reasoning(
            action_type=action_type,
            reasoning=reasoning,
            action_data=action_data,
            trigger=trigger,
            context=context,
            tags=tags,
        )
        return f"Logged wisdom {wisdom_id}"


class MaasvSearchWisdomTool(BaseTool):
    """LangChain tool for searching past wisdom entries."""

    name: str = "maasv_search_wisdom"
    description: str = (
        "Search past experiential wisdom — patterns of what worked and what didn't. "
        "Use this before making decisions to learn from past experience."
    )
    args_schema: Type[BaseModel] = SearchWisdomInput

    def _run(
        self,
        query: str,
        limit: int = 10,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        from maasv.core.wisdom import search_wisdom

        results = search_wisdom(query=query, limit=limit)
        if not results:
            return "No relevant wisdom found."

        lines = []
        for entry in results:
            score = entry.get("feedback_score")
            outcome = entry.get("outcome", "unknown")
            label = f"[{outcome}"
            if score is not None:
                label += f", {score}/5"
            label += "]"
            line = f"- {label} {entry.get('action_type', 'unknown')}: {entry.get('reasoning', '')[:200]}"
            lines.append(line)
        return "\n".join(lines)


# ============================================================================
# LANGCHAIN RETRIEVER
# ============================================================================


class MaasvRetriever(BaseRetriever):
    """LangChain retriever backed by maasv's 3-signal retrieval pipeline.

    Wraps find_similar_memories() and returns results as LangChain Document objects.

    Usage:
        retriever = MaasvRetriever(limit=5, category="project")
        docs = retriever.invoke("What framework does the app use?")
    """

    limit: int = 5
    category: Optional[str] = None
    subject: Optional[str] = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> list[Document]:
        from maasv.core.retrieval import find_similar_memories

        results = find_similar_memories(
            query=query,
            limit=self.limit,
            category=self.category,
            subject=self.subject,
        )

        documents = []
        for mem in results:
            metadata = {
                "memory_id": mem["id"],
                "category": mem.get("category", ""),
                "subject": mem.get("subject", ""),
                "confidence": mem.get("confidence", 1.0),
                "created_at": mem.get("created_at", ""),
            }
            if mem.get("importance") is not None:
                metadata["importance"] = mem["importance"]
            if mem.get("metadata"):
                try:
                    import json

                    extra = json.loads(mem["metadata"]) if isinstance(mem["metadata"], str) else mem["metadata"]
                    if isinstance(extra, dict):
                        metadata["extra"] = extra
                except (json.JSONDecodeError, TypeError):
                    pass

            documents.append(
                Document(
                    page_content=mem["content"],
                    metadata=metadata,
                )
            )

        return documents


# ============================================================================
# LANGGRAPH NODE HELPERS
# ============================================================================


def memory_store_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node that stores the latest message as a memory.

    Expects state to have:
        - "messages": list of message dicts with "content" key
        - "memory_category": str (default "conversation")
        - "memory_subject": optional str

    Adds "memory_id" to state.
    """
    from maasv.core.store import store_memory

    messages = state.get("messages", [])
    if not messages:
        return state

    last_message = messages[-1]
    content = last_message if isinstance(last_message, str) else last_message.get("content", "")
    if not content:
        return state

    memory_id = store_memory(
        content=content,
        category=state.get("memory_category", "conversation"),
        subject=state.get("memory_subject"),
        source="langgraph",
    )

    return {**state, "memory_id": memory_id}


def memory_retrieve_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node that retrieves relevant memories for the current query.

    Expects state to have:
        - "messages": list of message dicts with "content" key
        - "memory_limit": int (default 5)
        - "memory_category": optional str
        - "memory_subject": optional str

    Adds "memories" (list of dicts) to state.
    """
    from maasv.core.retrieval import find_similar_memories

    messages = state.get("messages", [])
    if not messages:
        return state

    last_message = messages[-1]
    query = last_message if isinstance(last_message, str) else last_message.get("content", "")
    if not query:
        return state

    memories = find_similar_memories(
        query=query,
        limit=state.get("memory_limit", 5),
        category=state.get("memory_category"),
        subject=state.get("memory_subject"),
    )

    return {**state, "memories": memories}


# ============================================================================
# CONVENIENCE
# ============================================================================


def get_maasv_tools() -> list[BaseTool]:
    """Get all maasv LangChain tools.

    Returns:
        List of [MaasvStoreMemoryTool, MaasvFindMemoriesTool,
                 MaasvExtractEntitiesTool, MaasvLogWisdomTool,
                 MaasvSearchWisdomTool]
    """
    return [
        MaasvStoreMemoryTool(),
        MaasvFindMemoriesTool(),
        MaasvExtractEntitiesTool(),
        MaasvLogWisdomTool(),
        MaasvSearchWisdomTool(),
    ]

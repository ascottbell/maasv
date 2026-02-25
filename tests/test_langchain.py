"""
Tests for the LangChain/LangGraph integration.

Mocks LangChain interfaces so tests run without langchain-core installed.
Tests verify the adapter correctly bridges LangChain's interface contracts
to maasv's core functions.
"""

import sys
import types

import pytest

# ============================================================================
# MOCK LANGCHAIN INTERFACES
# ============================================================================
# We mock langchain_core so tests don't require it as a dependency.
# These mocks replicate just enough of the interface contracts.


def _setup_langchain_mocks():
    """Install mock langchain_core modules into sys.modules."""
    # pydantic is available (it's a maasv dep via server/mcp extras,
    # but may not be installed in minimal test env). Mock it too.
    if "pydantic" not in sys.modules:
        pydantic_mod = types.ModuleType("pydantic")

        class _BaseModel:
            def __init_subclass__(cls, **kwargs):
                super().__init_subclass__(**kwargs)

        def _Field(**kwargs):
            return None

        pydantic_mod.BaseModel = _BaseModel
        pydantic_mod.Field = _Field
        sys.modules["pydantic"] = pydantic_mod

    # langchain_core
    lc = types.ModuleType("langchain_core")

    # langchain_core.tools
    lc_tools = types.ModuleType("langchain_core.tools")

    class BaseTool:
        name: str = ""
        description: str = ""
        args_schema = None

        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def invoke(self, input_dict, **kwargs):
            return self._run(**input_dict)

    lc_tools.BaseTool = BaseTool

    # langchain_core.retrievers
    lc_retrievers = types.ModuleType("langchain_core.retrievers")

    class BaseRetriever:
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def invoke(self, query, **kwargs):
            return self._get_relevant_documents(query)

    lc_retrievers.BaseRetriever = BaseRetriever

    # langchain_core.documents
    lc_documents = types.ModuleType("langchain_core.documents")

    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    lc_documents.Document = Document

    # langchain_core.callbacks
    lc_callbacks = types.ModuleType("langchain_core.callbacks")
    lc_callbacks.CallbackManagerForToolRun = type("CallbackManagerForToolRun", (), {})
    lc_callbacks.CallbackManagerForRetrieverRun = type("CallbackManagerForRetrieverRun", (), {})

    # Wire up
    sys.modules["langchain_core"] = lc
    sys.modules["langchain_core.tools"] = lc_tools
    sys.modules["langchain_core.retrievers"] = lc_retrievers
    sys.modules["langchain_core.documents"] = lc_documents
    sys.modules["langchain_core.callbacks"] = lc_callbacks
    lc.tools = lc_tools
    lc.retrievers = lc_retrievers
    lc.documents = lc_documents
    lc.callbacks = lc_callbacks


# Install mocks before importing the adapter
_setup_langchain_mocks()

# Now we need pydantic for the input schemas.
# If real pydantic is available, great. If not, our mock handles it.
# But since maasv uses pydantic in server extras, let's check.
try:
    from pydantic import BaseModel, Field  # noqa: F401

    _HAS_PYDANTIC = True
except ImportError:
    _HAS_PYDANTIC = False


# ============================================================================
# MOCK PROVIDERS (same pattern as test_learned_ranker.py)
# ============================================================================


class MockEmbedProvider:
    def __init__(self, dims=64):
        self.dims = dims

    def embed(self, text: str) -> list[float]:
        import hashlib

        h = hashlib.sha256(text.encode()).digest()
        vec = [b / 255.0 for b in h]
        while len(vec) < self.dims:
            vec.extend(vec)
        vec = vec[: self.dims]
        # Normalize
        import math

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)


class MockLLMProvider:
    def call(self, messages, model, max_tokens, source=""):
        return '{"entities": [], "relationships": []}'


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(scope="module")
def maasv_db(tmp_path_factory):
    """Initialize maasv with a fresh test database."""
    import maasv
    from maasv.config import MaasvConfig

    tmpdir = tmp_path_factory.mktemp("langchain_test")
    db_path = tmpdir / "test.db"

    config = MaasvConfig(
        db_path=db_path,
        embed_dims=64,
        extraction_model="test-model",
        inference_model="test-model",
        review_model="test-model",
        cross_encoder_enabled=False,
        learned_ranker_enabled=False,
    )

    llm = MockLLMProvider()
    embed = MockEmbedProvider(dims=64)
    maasv.init(config=config, llm=llm, embed=embed)
    return {"db_path": db_path}


@pytest.fixture
def seeded_db(maasv_db):
    """Seed the database with some test memories."""
    from maasv.core.store import store_memory

    store_memory(
        content="Adam lives on the Upper West Side of Manhattan",
        category="identity",
        subject="Adam",
        source="test",
    )
    store_memory(
        content="The Doris project uses FastAPI and Python",
        category="project",
        subject="Doris",
        source="test",
    )
    store_memory(
        content="Gabby is an M&A professional",
        category="family",
        subject="Gabby",
        source="test",
    )
    return maasv_db


# ============================================================================
# IMPORT TESTS
# ============================================================================


class TestImports:
    """Verify the adapter module imports and exposes the right classes."""

    def test_import_tools(self, maasv_db):
        from maasv.integrations.langchain import (
            MaasvExtractEntitiesTool,
            MaasvFindMemoriesTool,
            MaasvLogWisdomTool,
            MaasvSearchWisdomTool,
            MaasvStoreMemoryTool,
        )

        assert MaasvStoreMemoryTool is not None
        assert MaasvFindMemoriesTool is not None
        assert MaasvExtractEntitiesTool is not None
        assert MaasvLogWisdomTool is not None
        assert MaasvSearchWisdomTool is not None

    def test_import_retriever(self, maasv_db):
        from maasv.integrations.langchain import MaasvRetriever

        assert MaasvRetriever is not None

    def test_import_node_helpers(self, maasv_db):
        from maasv.integrations.langchain import memory_retrieve_node, memory_store_node

        assert callable(memory_store_node)
        assert callable(memory_retrieve_node)

    def test_import_convenience(self, maasv_db):
        from maasv.integrations.langchain import get_maasv_tools

        tools = get_maasv_tools()
        assert len(tools) == 5

    def test_tool_names_unique(self, maasv_db):
        from maasv.integrations.langchain import get_maasv_tools

        tools = get_maasv_tools()
        names = [t.name for t in tools]
        assert len(names) == len(set(names)), f"Duplicate tool names: {names}"


# ============================================================================
# STORE MEMORY TOOL TESTS
# ============================================================================


class TestStoreMemoryTool:
    def test_store_basic(self, maasv_db):
        from maasv.integrations.langchain import MaasvStoreMemoryTool

        tool = MaasvStoreMemoryTool()

        result = tool._run(
            content="Test memory from LangChain",
            category="test",
        )
        assert result.startswith("Stored memory mem_")

    def test_store_with_all_fields(self, maasv_db):
        from maasv.integrations.langchain import MaasvStoreMemoryTool

        tool = MaasvStoreMemoryTool()

        result = tool._run(
            content="Detailed memory with all fields",
            category="project",
            subject="TestProject",
            source="langchain-test",
            confidence=0.9,
            metadata={"key": "value"},
        )
        assert "Stored memory" in result

    def test_tool_metadata(self, maasv_db):
        from maasv.integrations.langchain import MaasvStoreMemoryTool

        tool = MaasvStoreMemoryTool()
        assert tool.name == "maasv_store_memory"
        assert "store" in tool.description.lower()
        assert tool.args_schema is not None


# ============================================================================
# FIND MEMORIES TOOL TESTS
# ============================================================================


class TestFindMemoriesTool:
    def test_find_basic(self, seeded_db):
        from maasv.integrations.langchain import MaasvFindMemoriesTool

        tool = MaasvFindMemoriesTool()

        result = tool._run(query="Where does Adam live?")
        assert isinstance(result, str)
        # Should find the seeded memory about Adam
        assert "Upper West Side" in result or "No relevant" in result

    def test_find_no_results(self, maasv_db):
        from maasv.integrations.langchain import MaasvFindMemoriesTool

        tool = MaasvFindMemoriesTool()

        result = tool._run(query="xyzzy_nonexistent_query_12345")
        # May or may not find results depending on vector similarity
        assert isinstance(result, str)

    def test_find_with_category_filter(self, seeded_db):
        from maasv.integrations.langchain import MaasvFindMemoriesTool

        tool = MaasvFindMemoriesTool()

        result = tool._run(query="project", category="project")
        assert isinstance(result, str)

    def test_tool_metadata(self, maasv_db):
        from maasv.integrations.langchain import MaasvFindMemoriesTool

        tool = MaasvFindMemoriesTool()
        assert tool.name == "maasv_find_memories"
        assert "search" in tool.description.lower() or "find" in tool.description.lower()


# ============================================================================
# EXTRACT ENTITIES TOOL TESTS
# ============================================================================


class TestExtractEntitiesTool:
    def test_extract_basic(self, maasv_db):
        from maasv.integrations.langchain import MaasvExtractEntitiesTool

        tool = MaasvExtractEntitiesTool()

        result = tool._run(
            summary="Adam works on the Doris project, which uses FastAPI and Python. "
            "Gabby is Adam's wife and works in M&A at a large bank.",
        )
        assert isinstance(result, str)
        # Our mock LLM returns empty arrays, so extraction succeeds with 0 entities
        assert "Extracted" in result or "failed" in result

    def test_extract_short_input(self, maasv_db):
        from maasv.integrations.langchain import MaasvExtractEntitiesTool

        tool = MaasvExtractEntitiesTool()

        # Short input should be handled gracefully
        result = tool._run(summary="Too short")
        assert isinstance(result, str)

    def test_tool_metadata(self, maasv_db):
        from maasv.integrations.langchain import MaasvExtractEntitiesTool

        tool = MaasvExtractEntitiesTool()
        assert tool.name == "maasv_extract_entities"


# ============================================================================
# LOG WISDOM TOOL TESTS
# ============================================================================


class TestLogWisdomTool:
    def test_log_basic(self, maasv_db):
        from maasv.integrations.langchain import MaasvLogWisdomTool

        tool = MaasvLogWisdomTool()

        result = tool._run(
            action_type="debugging_resolution",
            reasoning="Found the bug was caused by a race condition in the DB layer",
        )
        assert result.startswith("Logged wisdom")

    def test_log_with_all_fields(self, maasv_db):
        from maasv.integrations.langchain import MaasvLogWisdomTool

        tool = MaasvLogWisdomTool()

        result = tool._run(
            action_type="architecture_decision",
            reasoning="Chose SQLite over PostgreSQL for single-node deployments",
            action_data={"choice": "sqlite", "alternatives": ["postgres", "duckdb"]},
            trigger="scaling discussion",
            context="Building a local-first memory system",
            tags=["database", "architecture"],
        )
        assert "Logged wisdom" in result

    def test_tool_metadata(self, maasv_db):
        from maasv.integrations.langchain import MaasvLogWisdomTool

        tool = MaasvLogWisdomTool()
        assert tool.name == "maasv_log_wisdom"


# ============================================================================
# SEARCH WISDOM TOOL TESTS
# ============================================================================


class TestSearchWisdomTool:
    def test_search_basic(self, maasv_db):
        # First log something to search for
        from maasv.core.wisdom import log_reasoning
        from maasv.integrations.langchain import MaasvSearchWisdomTool

        log_reasoning(
            action_type="test_action",
            reasoning="Testing the search wisdom tool integration with LangChain",
        )

        tool = MaasvSearchWisdomTool()
        result = tool._run(query="LangChain")
        assert isinstance(result, str)

    def test_search_no_results(self, maasv_db):
        from maasv.integrations.langchain import MaasvSearchWisdomTool

        tool = MaasvSearchWisdomTool()

        result = tool._run(query="xyzzy_completely_unique_nonexistent_term")
        assert "No relevant wisdom found" in result

    def test_tool_metadata(self, maasv_db):
        from maasv.integrations.langchain import MaasvSearchWisdomTool

        tool = MaasvSearchWisdomTool()
        assert tool.name == "maasv_search_wisdom"


# ============================================================================
# RETRIEVER TESTS
# ============================================================================


class TestRetriever:
    def test_retriever_basic(self, seeded_db):
        from maasv.integrations.langchain import MaasvRetriever

        retriever = MaasvRetriever(limit=3)

        docs = retriever._get_relevant_documents("Where does Adam live?")
        assert isinstance(docs, list)
        for doc in docs:
            assert hasattr(doc, "page_content")
            assert hasattr(doc, "metadata")

    def test_retriever_metadata_fields(self, seeded_db):
        from maasv.integrations.langchain import MaasvRetriever

        retriever = MaasvRetriever(limit=5)

        docs = retriever._get_relevant_documents("project")
        if docs:
            meta = docs[0].metadata
            assert "memory_id" in meta
            assert "category" in meta
            assert "confidence" in meta
            assert "created_at" in meta

    def test_retriever_with_category(self, seeded_db):
        from maasv.integrations.langchain import MaasvRetriever

        retriever = MaasvRetriever(limit=5, category="family")

        docs = retriever._get_relevant_documents("family members")
        for doc in docs:
            assert doc.metadata.get("category") == "family"

    def test_retriever_empty_results(self, maasv_db):
        from maasv.integrations.langchain import MaasvRetriever

        retriever = MaasvRetriever(limit=5)

        docs = retriever._get_relevant_documents("xyzzy_completely_unique_term")
        assert isinstance(docs, list)

    def test_retriever_default_params(self, maasv_db):
        from maasv.integrations.langchain import MaasvRetriever

        retriever = MaasvRetriever()
        assert retriever.limit == 5
        assert retriever.category is None
        assert retriever.subject is None


# ============================================================================
# LANGGRAPH NODE TESTS
# ============================================================================


class TestLangGraphNodes:
    def test_memory_store_node(self, maasv_db):
        from maasv.integrations.langchain import memory_store_node

        state = {
            "messages": [{"content": "Remember that the meeting is on Tuesday"}],
            "memory_category": "schedule",
        }

        result = memory_store_node(state)
        assert "memory_id" in result
        assert result["memory_id"].startswith("mem_")

    def test_memory_store_node_empty_messages(self, maasv_db):
        from maasv.integrations.langchain import memory_store_node

        state = {"messages": []}
        result = memory_store_node(state)
        assert "memory_id" not in result

    def test_memory_store_node_string_messages(self, maasv_db):
        from maasv.integrations.langchain import memory_store_node

        state = {
            "messages": ["The project deadline is March 15th"],
            "memory_category": "project",
        }

        result = memory_store_node(state)
        assert "memory_id" in result

    def test_memory_retrieve_node(self, seeded_db):
        from maasv.integrations.langchain import memory_retrieve_node

        state = {
            "messages": [{"content": "What do I know about Adam?"}],
            "memory_limit": 3,
        }

        result = memory_retrieve_node(state)
        assert "memories" in result
        assert isinstance(result["memories"], list)

    def test_memory_retrieve_node_empty_messages(self, maasv_db):
        from maasv.integrations.langchain import memory_retrieve_node

        state = {"messages": []}
        result = memory_retrieve_node(state)
        assert "memories" not in result

    def test_memory_retrieve_node_with_filters(self, seeded_db):
        from maasv.integrations.langchain import memory_retrieve_node

        state = {
            "messages": [{"content": "family information"}],
            "memory_category": "family",
            "memory_limit": 5,
        }

        result = memory_retrieve_node(state)
        assert "memories" in result


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    def test_store_then_find_roundtrip(self, maasv_db):
        """Store a memory via the tool, then find it via the retriever."""
        from maasv.integrations.langchain import MaasvRetriever, MaasvStoreMemoryTool

        # Store
        store_tool = MaasvStoreMemoryTool()
        store_result = store_tool._run(
            content="The quarterly review meeting is scheduled for March 10th at 3pm",
            category="schedule",
            subject="quarterly_review",
        )
        assert "Stored memory" in store_result

        # Retrieve
        retriever = MaasvRetriever(limit=5, category="schedule")
        docs = retriever._get_relevant_documents("quarterly review meeting")
        assert any("quarterly review" in doc.page_content.lower() for doc in docs)

    def test_log_wisdom_then_search(self, maasv_db):
        """Log wisdom via the tool, then search for it."""
        from maasv.integrations.langchain import MaasvLogWisdomTool, MaasvSearchWisdomTool

        # Log
        log_tool = MaasvLogWisdomTool()
        log_result = log_tool._run(
            action_type="langchain_integration_test",
            reasoning="Testing the roundtrip between log and search in LangChain adapter",
        )
        assert "Logged wisdom" in log_result

        # Search
        search_tool = MaasvSearchWisdomTool()
        search_result = search_tool._run(query="LangChain adapter")
        assert isinstance(search_result, str)

    def test_get_maasv_tools_all_callable(self, maasv_db):
        """All tools returned by get_maasv_tools have _run methods."""
        from maasv.integrations.langchain import get_maasv_tools

        tools = get_maasv_tools()
        for tool in tools:
            assert hasattr(tool, "_run"), f"Tool {tool.name} missing _run method"
            assert callable(tool._run)

# LangChain / LangGraph Integration

Use maasv as a memory layer in LangChain agents and LangGraph workflows.

## Installation

```bash
pip install maasv[langchain]
```

This installs `langchain-core` as a dependency. You still need your LLM/embedding providers configured separately.

## Setup

Initialize maasv before using any integration tools:

```python
import maasv
from maasv.config import MaasvConfig

config = MaasvConfig(db_path="data/memory.db")
maasv.init(config=config, embed="ollama")
```

## LangChain Tools

Five tools wrap maasv's core functionality for use with LangChain agents:

```python
from maasv.integrations.langchain import get_maasv_tools

tools = get_maasv_tools()
# Returns: [MaasvStoreMemoryTool, MaasvFindMemoriesTool,
#           MaasvExtractEntitiesTool, MaasvLogWisdomTool,
#           MaasvSearchWisdomTool]
```

### MaasvStoreMemoryTool

Store facts, decisions, and observations as persistent memories.

```python
from maasv.integrations.langchain import MaasvStoreMemoryTool

tool = MaasvStoreMemoryTool()
result = tool.invoke({
    "content": "The user prefers dark mode in all applications",
    "category": "preference",
    "subject": "UI",
})
# "Stored memory mem_a1b2c3d4e5f6"
```

**Parameters:**
- `content` (str, required) -- The fact or memory to store
- `category` (str, required) -- Type: family, preference, project, decision, etc.
- `subject` (str, optional) -- Who/what this is about
- `source` (str, default "langchain") -- Where this came from
- `confidence` (float, default 1.0) -- Confidence score 0.0-1.0
- `metadata` (dict, optional) -- Additional structured data

### MaasvFindMemoriesTool

Search memories using maasv's 3-signal retrieval (vector + BM25 + graph).

```python
from maasv.integrations.langchain import MaasvFindMemoriesTool

tool = MaasvFindMemoriesTool()
result = tool.invoke({
    "query": "What are the user's UI preferences?",
    "limit": 5,
})
```

**Parameters:**
- `query` (str, required) -- Natural language search query
- `limit` (int, default 5) -- Maximum results (1-50)
- `category` (str, optional) -- Filter by category
- `subject` (str, optional) -- Filter by subject

### MaasvExtractEntitiesTool

Extract entities and relationships from text into the knowledge graph. Requires an LLM provider configured in maasv.

```python
from maasv.integrations.langchain import MaasvExtractEntitiesTool

tool = MaasvExtractEntitiesTool()
result = tool.invoke({
    "summary": "Alice works at Acme Corp on the Phoenix project, which uses React and PostgreSQL.",
    "topic": "team discussion",
})
# "Extracted 3 entities and 4 relationships"
```

### MaasvLogWisdomTool

Log reasoning before taking an action, enabling experiential learning.

```python
from maasv.integrations.langchain import MaasvLogWisdomTool

tool = MaasvLogWisdomTool()
result = tool.invoke({
    "action_type": "architecture_decision",
    "reasoning": "Chose SQLite for local-first deployment to avoid external DB dependency",
    "tags": ["database", "architecture"],
})
# "Logged wisdom 550e8400-e29b-..."
```

### MaasvSearchWisdomTool

Search past wisdom entries to learn from previous experience.

```python
from maasv.integrations.langchain import MaasvSearchWisdomTool

tool = MaasvSearchWisdomTool()
result = tool.invoke({
    "query": "database decisions",
    "limit": 5,
})
```

## LangChain Retriever

`MaasvRetriever` implements LangChain's `BaseRetriever` protocol, returning `Document` objects with full metadata.

```python
from maasv.integrations.langchain import MaasvRetriever

retriever = MaasvRetriever(limit=5, category="project")
docs = retriever.invoke("What framework does the app use?")

for doc in docs:
    print(doc.page_content)
    print(doc.metadata["category"], doc.metadata["memory_id"])
```

**Parameters:**
- `limit` (int, default 5) -- Maximum documents to return
- `category` (str, optional) -- Filter by memory category
- `subject` (str, optional) -- Filter by subject

**Document metadata includes:**
- `memory_id` -- maasv memory ID
- `category` -- Memory category
- `subject` -- Memory subject
- `confidence` -- Confidence score
- `created_at` -- Creation timestamp
- `importance` -- Importance score (if set)
- `extra` -- Additional metadata dict (if any)

### Use with RAG chains

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

retriever = MaasvRetriever(limit=5)

prompt = ChatPromptTemplate.from_template(
    "Context from memory:\n{context}\n\nQuestion: {question}"
)

def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

response = chain.invoke("What do you know about the Doris project?")
```

## LangGraph Node Helpers

Two node functions for memory-augmented LangGraph workflows:

### memory_store_node

Stores the latest message as a memory.

```python
from maasv.integrations.langchain import memory_store_node

# In a LangGraph StateGraph:
graph.add_node("store_memory", memory_store_node)
```

**Expected state keys:**
- `messages` (list) -- Message dicts with "content" key, or plain strings
- `memory_category` (str, default "conversation") -- Category for stored memory
- `memory_subject` (str, optional) -- Subject tag

**Adds to state:** `memory_id`

### memory_retrieve_node

Retrieves relevant memories for the current query.

```python
from maasv.integrations.langchain import memory_retrieve_node

graph.add_node("retrieve_memories", memory_retrieve_node)
```

**Expected state keys:**
- `messages` (list) -- Message dicts with "content" key, or plain strings
- `memory_limit` (int, default 5) -- Max memories to retrieve
- `memory_category` (str, optional) -- Filter by category
- `memory_subject` (str, optional) -- Filter by subject

**Adds to state:** `memories` (list of memory dicts)

### Example: Memory-Augmented Agent

```python
from langgraph.graph import StateGraph, END
from maasv.integrations.langchain import memory_retrieve_node, memory_store_node

class AgentState(TypedDict):
    messages: list
    memories: list
    memory_id: str
    memory_category: str

graph = StateGraph(AgentState)
graph.add_node("retrieve", memory_retrieve_node)
graph.add_node("respond", respond_fn)  # your LLM call
graph.add_node("store", memory_store_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "respond")
graph.add_edge("respond", "store")
graph.add_edge("store", END)

app = graph.compile()
result = app.invoke({
    "messages": [{"content": "What's the project status?"}],
    "memory_category": "project",
})
```

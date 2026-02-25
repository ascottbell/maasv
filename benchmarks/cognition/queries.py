"""Cognition benchmark query definitions.

Queries are designed against real production data patterns. Content references
use generic terms (e.g. "the user" not real names) but will match against
real memories in the DB. Published results show only scores, never content.
"""

from __future__ import annotations


# ============================================================================
# 1. TEMPORAL REASONING
# ============================================================================
# Tests: Does the system prefer recent decisions over stale ones on the same
# topic? Vector similarity is time-agnostic; a cognition system is not.

TEMPORAL_QUERIES = [
    {
        "id": "temporal-01",
        "query": "What text-to-speech system does Doris currently use?",
        "description": "TTS evolved through 4+ architectures. Only the most recent is correct.",
    },
    {
        "id": "temporal-02",
        "query": "How often does the school calendar scout run?",
        "description": "Frequency changed multiple times: hourly -> every 2 hours -> daily at 6 AM.",
    },
    {
        "id": "temporal-03",
        "query": "What model does Doris use for graph extraction?",
        "description": "Switched from Ollama/Qwen to Claude Haiku. Supersession chain exists.",
    },
    {
        "id": "temporal-04",
        "query": "What is the current Doris LLM routing strategy?",
        "description": "Evolved from single-model to per-operation routing (MAASV-39 pattern).",
    },
    {
        "id": "temporal-05",
        "query": "What voice architecture does My Doris use?",
        "description": "Evolved from Claude+TTS -> GPT-4o Realtime -> GPT-5 streaming + Azure SSML.",
    },
    {
        "id": "temporal-06",
        "query": "What are the current maasv retrieval improvements?",
        "description": "Active development area with daily changes. Recent > old.",
    },
]


# ============================================================================
# 2. SESSION COHERENCE
# ============================================================================
# Tests: Do earlier queries in a session narrow and contextualize later results?
# Each chain is a sequence where query N depends on context from queries 1..N-1.

SESSION_CHAINS = [
    {
        "id": "session-A",
        "description": "Project deep-dive: maasv retrieval pipeline",
        "queries": [
            "What is maasv?",
            "How does the retrieval pipeline work?",
            "What about the learned ranker?",
            "How do we benchmark it?",
        ],
    },
    {
        "id": "session-B",
        "description": "Family context: children and scheduling",
        "queries": [
            "Tell me about the kids",
            "What school do they go to?",
            "Any schedule changes recently?",
        ],
    },
    {
        "id": "session-C",
        "description": "Architecture evolution: Doris voice system",
        "queries": [
            "What is the Doris voice architecture?",
            "Why did we change the TTS system?",
            "What about latency?",
        ],
    },
    {
        "id": "session-D",
        "description": "Cross-project: technology comparison",
        "queries": [
            "What technologies does Tod use?",
            "How is that different from Doris?",
            "What about authentication?",
        ],
    },
    {
        "id": "session-E",
        "description": "Debugging context: system reliability",
        "queries": [
            "What system reliability issues have there been?",
            "What about email failures?",
            "How was that fixed?",
        ],
    },
]


# ============================================================================
# 3. CROSS-DOMAIN GRAPH TRAVERSAL
# ============================================================================
# Tests: Can the system connect information across different memory domains
# via the entity-relationship graph? The answer isn't in any single memory.

GRAPH_QUERIES = [
    {
        "id": "graph-01",
        "query": "What technologies do Doris and maasv share?",
        "description": "Requires traversing uses_tech from two project entities and computing intersection.",
    },
    {
        "id": "graph-02",
        "query": "Which projects use Claude and what for?",
        "description": "Requires traversing Claude entity's relationships to all connected projects.",
    },
    {
        "id": "graph-03",
        "query": "What databases are used across all projects?",
        "description": "Requires aggregating depends_on/uses_tech for database entities across projects.",
    },
    {
        "id": "graph-04",
        "query": "How does the infrastructure connect the Mac Mini to all services?",
        "description": "Multi-hop: Mac Mini -> runs_on -> Docker/OrbStack -> hosts -> services.",
    },
    {
        "id": "graph-05",
        "query": "What's the relationship between the person TerryAnn and the project?",
        "description": "Cross-domain link: person entity -> project naming origin. Requires graph.",
    },
    {
        "id": "graph-06",
        "query": "What Python packages are shared between Doris and TerryAnn?",
        "description": "Requires traversing dependency relationships for two separate projects.",
    },
    {
        "id": "graph-07",
        "query": "How are the family members connected in the knowledge graph?",
        "description": "Traversal through person entities and relationship predicates.",
    },
]


# ============================================================================
# 4. CONSOLIDATION RESISTANCE
# ============================================================================
# Tests: Does the system avoid surfacing superseded/duplicate memories?
# The DB has 2,439 superseded memories (32% of total).

CONSOLIDATION_QUERIES = [
    {
        "id": "consolidation-01",
        "query": "What TTS voice does Doris use?",
        "description": "Dense supersession chain: ElevenLabs -> Supertonic -> Azure -> current.",
    },
    {
        "id": "consolidation-02",
        "query": "What is the Doris architecture?",
        "description": "Multiple early architecture memories were superseded by consolidated ones.",
    },
    {
        "id": "consolidation-03",
        "query": "What extraction backend does Doris use?",
        "description": "Ollama-era memory superseded by Claude Haiku decision.",
    },
    {
        "id": "consolidation-04",
        "query": "What audio format does Doris TTS output?",
        "description": "Chain: 16kHz MP3 -> 24kHz WAV -> 48kHz WAV. Only latest should surface.",
    },
    {
        "id": "consolidation-05",
        "query": "Where is the Doris project located on disk?",
        "description": "Multiple superseded memories about project path. One authoritative version.",
    },
    {
        "id": "consolidation-06",
        "query": "What embedding model does maasv use?",
        "description": "Evolved from generic to Qwen3-Embedding-8B. Supersession chain exists.",
    },
]


# ============================================================================
# 5. DECAY + IDENTITY PROTECTION
# ============================================================================
# Tests: Old identity/family memories resist decay (protected categories).
# Old transient events decay normally. Two opposing behaviors.

DECAY_IDENTITY_QUERIES = [
    # Protected — should resist decay
    {
        "id": "decay-protect-01",
        "query": "What is the user's git email?",
        "type": "protected",
        "expected_categories": ["identity"],
        "description": "Identity memory, ~7 weeks old. Should still rank high.",
    },
    {
        "id": "decay-protect-02",
        "query": "How many kids does the user have?",
        "type": "protected",
        "expected_categories": ["family"],
        "description": "Family memory, oldest in DB. Should always rank high.",
    },
    {
        "id": "decay-protect-03",
        "query": "What is the family pet's name?",
        "type": "protected",
        "expected_categories": ["family", "person"],
        "description": "Family/person memory with high access count (45+). Should resist decay.",
    },
    {
        "id": "decay-protect-04",
        "query": "Who is the user's mother?",
        "type": "protected",
        "expected_categories": ["family"],
        "description": "Family identity memory. 7 weeks old, should still be top-1.",
    },
    # Transient — should decay
    {
        "id": "decay-event-01",
        "query": "What was the weather like in late January?",
        "type": "transient",
        "expected_categories": ["event"],
        "description": "Weather event from 5+ weeks ago. Should rank below recent content.",
    },
    {
        "id": "decay-event-02",
        "query": "Was there a system malfunction in January?",
        "type": "transient",
        "expected_categories": ["event"],
        "description": "Old event. Recent reliability memories should rank higher.",
    },
    {
        "id": "decay-event-03",
        "query": "What happened at the diner last month?",
        "type": "transient",
        "expected_categories": ["event"],
        "description": "Old dining event should decay below persistent preference memories.",
    },
    {
        "id": "decay-event-04",
        "query": "What breakfast plans were there for a recent Sunday?",
        "type": "transient",
        "expected_categories": ["event", "reminder"],
        "description": "Specific old schedule event. Should decay below persistent preferences.",
    },
]


# ============================================================================
# 6. PROACTIVE RELEVANCE
# ============================================================================
# Tests: Can the system surface useful results that the user didn't explicitly
# ask for? The most useful result isn't always the closest semantic match.

PROACTIVE_QUERIES = [
    {
        "id": "proactive-01",
        "query": "What should I plan for the weekend?",
        "description": "Should surface family activity patterns, diner habits, Hudson Valley trips.",
    },
    {
        "id": "proactive-02",
        "query": "I'm starting a new Python project",
        "description": "Should surface common tech stack patterns: FastAPI, SQLite, Claude API, etc.",
    },
    {
        "id": "proactive-03",
        "query": "We need to follow up with the school",
        "description": "Should surface specific pending school items, not just generic school memories.",
    },
    {
        "id": "proactive-04",
        "query": "What's the status of the open-source work?",
        "description": "Should surface doris-public AND maasv AND Tod open-source decisions.",
    },
    {
        "id": "proactive-05",
        "query": "I want to build something for the kids",
        "description": "Should surface children's specific interests via person entity graph.",
    },
    {
        "id": "proactive-06",
        "query": "What's failing right now?",
        "description": "Should aggregate failure states across domains: email, calendar, system.",
    },
    {
        "id": "proactive-07",
        "query": "Remind me about the security lessons",
        "description": "Should surface doris-public post-mortem, specific failures, pre-ship checklist.",
    },
    {
        "id": "proactive-08",
        "query": "How does the user typically handle bedtime?",
        "description": "Should surface bedtime story preferences AND sleep boundary behavior.",
    },
]

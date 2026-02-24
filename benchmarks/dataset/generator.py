"""Synthetic dataset generator for retrieval benchmarks.

Generates thematic clusters of memories, entities, relationships, and
queries with graded relevance judgments. Deterministic given a seed.
"""

from __future__ import annotations

import random

from benchmarks.dataset.schemas import (
    BenchmarkDataset,
    Entity,
    Memory,
    QueryJudgment,
    Relationship,
)

# ---------------------------------------------------------------------------
# Cluster definitions
# ---------------------------------------------------------------------------

CLUSTERS: dict[str, dict] = {
    "work_project_a": {
        "keywords": ["atlas", "dashboard", "analytics", "metrics", "kpi"],
        "category": "project",
        "subject": "Atlas",
        "entities": [
            ("Atlas", "project"),
            ("React", "technology"),
            ("PostgreSQL", "technology"),
        ],
        "relationships": [
            ("Atlas", "built_with", "React"),
            ("Atlas", "built_with", "PostgreSQL"),
        ],
        "memories": [
            "Atlas dashboard shows real-time analytics metrics for the team",      # 0
            "Atlas uses React for the frontend with a PostgreSQL backend",          # 1
            "The Atlas KPI dashboard was redesigned last quarter",                  # 2
            "Atlas metrics pipeline processes events through a streaming architecture",  # 3
            "Atlas dashboard added a new user retention analytics panel",           # 4
            "The Atlas project has a weekly sprint review every Thursday",          # 5
            "Atlas dashboard performance improved after adding query caching",      # 6
            "Atlas frontend uses React Server Components for analytics pages",      # 7
            "Atlas PostgreSQL queries were optimized to reduce dashboard load time",  # 8
            "The Atlas analytics team requested drill-down filters for metrics",    # 9
            "Atlas KPI targets were revised based on Q3 performance data",          # 10
            "Atlas dashboard error rates decreased after the latest deploy",        # 11
        ],
        "queries": [
            # BM25: exact keywords in memories
            {
                "query": "Atlas dashboard",
                "relevant": [0, 2, 4, 6, 8, 11],
                "grades": [1.0, 0.5, 0.5, 0.5, 0.5, 0.5],
            },
            # Vector: cluster keywords trigger embedding similarity
            {
                "query": "analytics KPI data",
                "relevant": [0, 3, 9, 10],
                "grades": [1.0, 1.0, 0.5, 0.5],
            },
            # Graph: "React" entity -> Atlas relationship -> Atlas memories
            {
                "query": "React",
                "relevant": [1, 7],
                "grades": [1.0, 0.5],
            },
            # Graph: "PostgreSQL" entity -> Atlas relationship -> Atlas memories
            {
                "query": "PostgreSQL",
                "relevant": [1, 8],
                "grades": [1.0, 0.5],
            },
        ],
    },
    "work_project_b": {
        "keywords": ["beacon", "notification", "alert", "push", "messaging"],
        "category": "project",
        "subject": "Beacon",
        "entities": [
            ("Beacon", "project"),
            ("Firebase", "technology"),
            ("Redis", "technology"),
        ],
        "relationships": [
            ("Beacon", "built_with", "Firebase"),
            ("Beacon", "uses_tech", "Redis"),
        ],
        "memories": [
            "Beacon is the notification service that handles push alerts",          # 0
            "Beacon uses Firebase for push notifications and Redis for queuing",    # 1
            "Beacon messaging system supports email, SMS, and push channels",       # 2
            "The Beacon alert priority system was reworked to reduce noise",         # 3
            "Beacon notification templates are stored in a JSON configuration",     # 4
            "Beacon Redis queue handles 10k messages per second at peak",           # 5
            "Beacon push notification delivery rate is 98.5 percent",               # 6
            "The Beacon project migrated from SNS to Firebase last year",           # 7
            "Beacon alert throttling prevents duplicate notifications within 5 minutes",  # 8
            "Beacon messaging added rich notification support for iOS",             # 9
        ],
        "queries": [
            # BM25: exact keywords
            {
                "query": "Beacon alert",
                "relevant": [0, 3, 8],
                "grades": [0.5, 1.0, 1.0],
            },
            # Vector: cluster keywords trigger similarity
            {
                "query": "push notification messaging",
                "relevant": [0, 1, 2, 6, 9],
                "grades": [1.0, 0.5, 0.5, 0.5, 0.5],
            },
            # Graph: "Firebase" entity -> Beacon relationship
            {
                "query": "Firebase",
                "relevant": [1, 7],
                "grades": [1.0, 0.5],
            },
            # Graph: "Redis" entity -> Beacon relationship
            {
                "query": "Redis",
                "relevant": [1, 5],
                "grades": [1.0, 0.5],
            },
        ],
    },
    "family": {
        "keywords": [
            "maria", "ethan", "sophie", "wife", "son", "daughter",
            "birthday", "family", "kids",
        ],
        "category": "family",
        "subject": None,
        "entities": [
            ("Alex", "person"),
            ("Maria", "person"),
            ("Ethan", "person"),
            ("Sophie", "person"),
        ],
        "relationships": [
            ("Alex", "married_to", "Maria"),
            ("Alex", "parent_of", "Ethan"),
            ("Alex", "parent_of", "Sophie"),
        ],
        "memories": [
            "Maria is my wife, she works in finance doing consulting",             # 0
            "Ethan is my son, born March 15th, currently 8 years old",             # 1
            "Sophie is my daughter, born September 22nd, currently 5 years old",   # 2
            "Ethan loves building with Lego and playing Minecraft",                # 3
            "Sophie is really into drawing and painting right now",                # 4
            "We live in the Riverside neighborhood downtown",                      # 5
            "The family has a cabin upstate near the lake",                        # 6
            "Maria and I usually do date night on Fridays",                        # 7
            "Ethan started piano lessons last month",                              # 8
            "Sophie wants a birthday party at the science museum",                 # 9
            "Family movie night is usually on Sundays",                            # 10
            "Kids bedtime routine is bath, story, lights out by 8pm",              # 11
            "Maria prefers Italian restaurants for date night",                     # 12
            "Ethan has soccer practice on Wednesdays after school",                # 13
            "Sophie goes to kindergarten at Lincoln Elementary",                    # 14
        ],
        "queries": [
            # BM25: exact name match
            {
                "query": "Ethan son",
                "relevant": [1, 3, 8, 13],
                "grades": [1.0, 0.5, 0.5, 0.5],
            },
            # BM25: exact name match
            {
                "query": "Maria wife",
                "relevant": [0, 7, 12],
                "grades": [1.0, 0.5, 0.5],
            },
            # Vector: cluster keywords family/kids/birthday
            {
                "query": "kids birthday family",
                "relevant": [9, 10, 11],
                "grades": [1.0, 0.5, 0.5],
            },
            # Graph: Alex entity -> parent_of -> Sophie -> memories about Sophie
            {
                "query": "Sophie",
                "relevant": [2, 4, 9, 14],
                "grades": [1.0, 0.5, 0.5, 0.5],
            },
        ],
    },
    "preferences": {
        "keywords": [
            "prefer", "like", "favorite", "always", "never", "hate",
            "love", "choice", "style",
        ],
        "category": "preference",
        "subject": "Alex",
        "entities": [
            ("Alex", "person"),
        ],
        "relationships": [],
        "memories": [
            "I prefer dark mode in all applications",                              # 0
            "My favorite programming language is Python",                          # 1
            "I always use Vim keybindings in my editor",                            # 2
            "I hate unnecessary meetings, prefer async communication",             # 3
            "My favorite coffee is a cortado from the local shop",                 # 4
            "I prefer TypeScript over plain JavaScript for frontend",              # 5
            "I like to work in focused 2-hour blocks",                             # 6
            "I prefer quality over speed when building software",                  # 7
            "I never use tabs for indentation, always spaces",                     # 8
            "My favorite restaurant is the Italian place on Amsterdam Ave",         # 9
            "I prefer direct communication without corporate speak",               # 10
            "I like to read technical books before bed",                            # 11
        ],
        "queries": [
            # BM25: exact keyword match
            {
                "query": "favorite coffee",
                "relevant": [4],
                "grades": [1.0],
            },
            # Vector: cluster keywords prefer/like/favorite
            {
                "query": "preferences style choices",
                "relevant": [0, 5, 7, 8],
                "grades": [1.0, 0.5, 0.5, 0.5],
            },
            # BM25: exact keyword
            {
                "query": "prefer communication",
                "relevant": [3, 10],
                "grades": [1.0, 1.0],
            },
        ],
    },
    "people": {
        "keywords": [
            "sarah", "mike", "chen", "colleague", "engineer", "designer",
            "manager", "team",
        ],
        "category": "person",
        "subject": None,
        "entities": [
            ("Sarah Chen", "person"),
            ("Mike Rodriguez", "person"),
            ("Emily Park", "person"),
        ],
        "relationships": [
            ("Alex", "works_with", "Sarah Chen"),
            ("Alex", "works_with", "Mike Rodriguez"),
            ("Sarah Chen", "manages", "Atlas"),
            ("Mike Rodriguez", "works_on", "Beacon"),
        ],
        "memories": [
            "Sarah Chen is the engineering manager for the Atlas team",             # 0
            "Mike Rodriguez is the lead engineer on Beacon",                        # 1
            "Emily Park is the UX designer who works across both teams",            # 2
            "Sarah is very detail-oriented and prefers written proposals",           # 3
            "Mike is great at system design but sometimes overengineers",            # 4
            "Emily joined the team six months ago from a startup",                  # 5
            "Sarah Chen has a PhD from MIT in distributed systems",                 # 6
            "Mike Rodriguez previously worked at Google on their notification infra",  # 7
            "Emily Park specializes in accessibility and inclusive design",          # 8
            "Sarah prefers biweekly one-on-ones over weekly standups",              # 9
            "Mike mentors two junior engineers on the Beacon team",                 # 10
        ],
        "queries": [
            # BM25: exact name
            {
                "query": "Sarah Chen",
                "relevant": [0, 6],
                "grades": [1.0, 0.5],
            },
            # Vector: cluster keywords engineer/manager/team
            {
                "query": "engineer manager team colleague",
                "relevant": [0, 1, 2, 10],
                "grades": [1.0, 1.0, 0.5, 0.5],
            },
            # Graph: Mike Rodriguez -> works_on -> Beacon -> Beacon memories
            {
                "query": "Mike Rodriguez",
                "relevant": [1, 4, 7, 10],
                "grades": [1.0, 0.5, 0.5, 0.5],
            },
        ],
    },
    "places": {
        "keywords": [
            "office", "building", "floor", "room", "location", "address",
            "downtown", "lakeside",
        ],
        "category": "history",
        "subject": None,
        "entities": [
            ("Downtown Office", "place"),
            ("Lakeside Cabin", "place"),
        ],
        "relationships": [
            ("Alex", "lives_in", "Downtown Office"),
            ("Alex", "has_property_in", "Lakeside Cabin"),
        ],
        "memories": [
            "The downtown office is on the 12th floor at 100 Main Street",          # 0
            "The lakeside cabin has a detached garage we use as an office",         # 1
            "Conference room A on the 12th floor has the best video setup",         # 2
            "The office kitchen was renovated and now has a good espresso machine", # 3
            "The lakeside property has 2 acres with a garden",                      # 4
            "Parking at the downtown office building is validated for 2 hours",      # 5
            "The office moved from floor 8 to floor 12 last year",                  # 6
            "The lakeside cabin needs a new roof before winter",                     # 7
            "Closest bus stop to the office is Central Station",                     # 8
            "The office has a standing desk policy, everyone gets one",              # 9
        ],
        "queries": [
            # BM25: exact keywords
            {
                "query": "downtown office",
                "relevant": [0, 5],
                "grades": [1.0, 0.5],
            },
            # BM25: exact keywords
            {
                "query": "lakeside cabin",
                "relevant": [1, 4, 7],
                "grades": [1.0, 0.5, 0.5],
            },
            # Vector: cluster keywords floor/building/room/office
            {
                "query": "building room floor location",
                "relevant": [0, 2, 6],
                "grades": [1.0, 0.5, 0.5],
            },
        ],
    },
    "decisions": {
        "keywords": [
            "decided", "decision", "chose", "switched", "migrated",
            "adopted", "approved",
        ],
        "category": "decision",
        "subject": None,
        "entities": [
            ("TypeScript Migration", "project"),
            ("Vercel", "technology"),
        ],
        "relationships": [
            ("Atlas", "built_with", "TypeScript Migration"),
            ("Atlas", "hosted_on", "Vercel"),
        ],
        "memories": [
            "Decided to migrate Atlas frontend from JavaScript to TypeScript",      # 0
            "Chose Vercel over AWS for Atlas hosting due to simpler deployment",    # 1
            "Approved the decision to adopt GraphQL for the new API layer",         # 2
            "Switched from Jest to Vitest for faster test execution",               # 3
            "Decided to use Tailwind CSS instead of styled-components",             # 4
            "Migrated the CI pipeline from Jenkins to GitHub Actions",               # 5
            "Chose SQLite over DynamoDB for the embedded analytics cache",           # 6
            "Adopted conventional commits for the entire organization",              # 7
            "Decided to sunset the legacy notification system by Q2",               # 8
            "Approved switching from Slack to Discord for team communication",       # 9
        ],
        "queries": [
            # BM25: exact keywords
            {
                "query": "Vercel hosting",
                "relevant": [1],
                "grades": [1.0],
            },
            # Vector: cluster keywords decided/chose/switched/migrated
            {
                "query": "decided switched adopted",
                "relevant": [0, 3, 4, 7],
                "grades": [0.5, 0.5, 0.5, 0.5],
            },
            # Graph: Atlas entity -> hosted_on -> Vercel -> decision memories
            {
                "query": "Atlas",
                "relevant": [0, 1],
                "grades": [1.0, 0.5],
            },
        ],
    },
    "learning": {
        "keywords": [
            "learned", "study", "reading", "book", "course", "tutorial",
            "understand", "insight",
        ],
        "category": "learning",
        "subject": "Alex",
        "entities": [
            ("Alex", "person"),
        ],
        "relationships": [],
        "memories": [
            "Learned that Rust borrow checker prevents data races at compile time",  # 0
            "Reading Designing Data-Intensive Applications by Martin Kleppmann",    # 1
            "Took a course on distributed systems and CAP theorem tradeoffs",       # 2
            "Insight: event sourcing works well for audit trails but adds complexity",  # 3
            "Learned about CRDT data structures for conflict-free replication",     # 4
            "Studying the Raft consensus algorithm for distributed state machines", # 5
            "Book recommendation from Sarah: The Staff Engineer's Path",            # 6
            "Tutorial on WebAssembly showed 3x speedup for compute-heavy tasks",   # 7
            "Understood why B-trees are preferred over LSM trees for read-heavy workloads",  # 8
            "Learned that SQLite can handle 100k reads per second easily",          # 9
            "Reading about information retrieval and BM25 ranking algorithms",      # 10
            "Course on machine learning covered gradient descent and backpropagation",  # 11
        ],
        "queries": [
            # BM25: exact keywords
            {
                "query": "reading book",
                "relevant": [1, 6, 10],
                "grades": [1.0, 0.5, 0.5],
            },
            # BM25: exact keywords
            {
                "query": "distributed systems",
                "relevant": [2, 5],
                "grades": [1.0, 0.5],
            },
            # Vector: cluster keywords learned/study/course/insight
            {
                "query": "learned study insight understanding",
                "relevant": [0, 3, 4, 8, 9],
                "grades": [0.5, 0.5, 0.5, 0.5, 0.5],
            },
        ],
    },
}


def _build_cluster_keywords() -> dict[str, list[str]]:
    """Extract cluster_keywords mapping for the DeterministicEmbedProvider."""
    return {name: cluster["keywords"] for name, cluster in CLUSTERS.items()}


def generate_dataset(
    seed: int = 42,
    scale: str = "medium",
) -> BenchmarkDataset:
    """Generate a synthetic benchmark dataset.

    Args:
        seed: Random seed for reproducibility.
        scale: "small" (base clusters only), "medium" (2x), or "large" (5x).
            Controls how many duplicate/variant memories are generated per cluster.

    Returns:
        BenchmarkDataset with memories, entities, relationships, and query judgments.
    """
    rng = random.Random(seed)

    multiplier = {"small": 1, "medium": 2, "large": 5}.get(scale, 2)

    all_memories: list[Memory] = []
    all_entities: list[Entity] = []
    all_relationships: list[Relationship] = []
    all_judgments: list[QueryJudgment] = []
    # Track entity names we've already added (avoid dupes across clusters)
    seen_entities: set[str] = set()

    for cluster_name, cluster in CLUSTERS.items():
        cluster_start_idx = len(all_memories)

        # --- Memories ---
        base_memories = cluster["memories"]
        for mem_text in base_memories:
            all_memories.append(
                Memory(
                    content=mem_text,
                    category=cluster["category"],
                    subject=cluster.get("subject"),
                    importance=round(rng.uniform(0.3, 1.0), 2),
                    created_days_ago=rng.randint(0, 180),
                )
            )

        # For scale > small, add paraphrased variants
        if multiplier > 1:
            suffixes = [
                " (updated)", " — confirmed", " — as of recently",
                " (still relevant)", " (important)",
            ]
            for _ in range(multiplier - 1):
                for mem_text in base_memories:
                    suffix = rng.choice(suffixes)
                    all_memories.append(
                        Memory(
                            content=mem_text + suffix,
                            category=cluster["category"],
                            subject=cluster.get("subject"),
                            importance=round(rng.uniform(0.2, 0.8), 2),
                            created_days_ago=rng.randint(0, 365),
                        )
                    )

        # --- Entities ---
        for ent_name, ent_type in cluster["entities"]:
            if ent_name not in seen_entities:
                all_entities.append(Entity(name=ent_name, entity_type=ent_type))
                seen_entities.add(ent_name)

        # --- Relationships ---
        for subj, pred, obj in cluster["relationships"]:
            all_relationships.append(Relationship(subj, pred, obj))

        # --- Query judgments ---
        # Remap local indices to global indices
        for q in cluster["queries"]:
            global_indices = [cluster_start_idx + i for i in q["relevant"]]
            all_judgments.append(
                QueryJudgment(
                    query=q["query"],
                    relevant_memory_indices=global_indices,
                    relevance_grades=q["grades"],
                )
            )

    return BenchmarkDataset(
        memories=all_memories,
        entities=all_entities,
        relationships=all_relationships,
        judgments=all_judgments,
        cluster_keywords=_build_cluster_keywords(),
        seed=seed,
    )

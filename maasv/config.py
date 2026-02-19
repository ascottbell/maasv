"""
maasv configuration.

All paths, model names, and tuning parameters are set here.
No hardcoded values in the rest of the package.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class MaasvConfig:
    """Configuration for the maasv cognition layer."""

    # Database
    db_path: Path

    # Embedding dimensions (must match the EmbedProvider output)
    embed_dims: int = 1024

    # Models (passed to LLMProvider.call — provider decides how to route)
    extraction_model: str = "claude-haiku-4-5-20251001"
    inference_model: str = "claude-haiku-4-5-20251001"
    review_model: str = "claude-haiku-4-5-20251001"

    # Memory hygiene
    backup_dir: Optional[Path] = None
    max_hygiene_backups: int = 3
    protected_categories: set[str] = field(default_factory=lambda: {"identity", "family"})
    protected_subjects: set[str] = field(default_factory=set)

    # Hygiene thresholds
    similarity_threshold: float = 0.95
    stale_days: int = 30
    min_confidence_threshold: float = 0.5
    cluster_similarity: float = 0.85

    # Cross-encoder reranking (opt-in: requires sentence-transformers + torch ~2GB)
    cross_encoder_enabled: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Sleep worker
    idle_threshold_seconds: int = 30
    idle_check_interval: int = 5

    # Known entities for extraction prompts (name -> type)
    known_entities: dict[str, str] = field(default_factory=dict)

    # Hygiene log path (optional — if None, no log file written)
    hygiene_log_path: Optional[Path] = None

    # Action type groupings for wisdom "similar enough" matching
    action_families: dict[str, list[str]] = field(default_factory=dict)

    # Category priority for tiered memory context (lower = higher priority)
    category_priority: dict[str, int] = field(default_factory=lambda: {
        'identity': 1,
        'family': 2,
        'preference': 3,
        'project': 4,
        'decision': 5,
        'person': 6,
        'learning': 7,
        'history': 8,
        'home': 9,
        'conversation': 10,
    })

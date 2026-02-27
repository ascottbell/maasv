"""
maasv Model Router — Tiered model selection for cost-effective inference.

Routes requests to the cheapest model capable of handling them:
- Tier 0: Local 8B (free, fast) — routing, classification, simple extraction
- Tier 1: Haiku ($1/MTok) — moderate complexity, structured extraction
- Tier 2: Sonnet ($3/MTok) — complex tasks, code generation, creative
- Tier 3: Opus ($5/MTok) — reasoning-heavy, critical decisions, multi-step

The router classifies a request by complexity and picks the appropriate tier.
Callers can also force a minimum tier for tasks they know are hard.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

logger = logging.getLogger(__name__)


class ModelTier(IntEnum):
    """Model capability tiers, ordered by cost/capability."""

    LOCAL = 0  # Local 8B — free, ~50 tok/s on M4
    HAIKU = 1  # Cloud fast/cheap — $1/MTok in, $5/MTok out
    SONNET = 2  # Cloud capable — $3/MTok in, $15/MTok out
    OPUS = 3  # Cloud powerful — $5/MTok in, $25/MTok out


@dataclass
class TierConfig:
    """Configuration for a model tier."""

    tier: ModelTier
    model_id: str
    description: str
    max_context_tokens: int = 128_000
    max_output_tokens: int = 4096
    supports_tools: bool = True
    is_local: bool = False


# Default tier configurations — can be overridden via ModelRouter constructor
DEFAULT_TIERS = {
    ModelTier.LOCAL: TierConfig(
        tier=ModelTier.LOCAL,
        model_id="qwen3:8b",
        description="Local 8B for routing, classification, simple extraction",
        max_context_tokens=32_000,
        max_output_tokens=2048,
        supports_tools=False,
        is_local=True,
    ),
    ModelTier.HAIKU: TierConfig(
        tier=ModelTier.HAIKU,
        model_id="claude-haiku-4-5-20251001",
        description="Fast cloud model for structured extraction and moderate tasks",
        max_context_tokens=200_000,
        max_output_tokens=8192,
    ),
    ModelTier.SONNET: TierConfig(
        tier=ModelTier.SONNET,
        model_id="claude-sonnet-4-6",
        description="Capable cloud model for complex tasks and code generation",
        max_context_tokens=200_000,
        max_output_tokens=16384,
    ),
    ModelTier.OPUS: TierConfig(
        tier=ModelTier.OPUS,
        model_id="claude-opus-4-6",
        description="Most capable model for reasoning-heavy and critical decisions",
        max_context_tokens=200_000,
        max_output_tokens=16384,
    ),
}


# ============================================================================
# TASK COMPLEXITY CLASSIFICATION
# ============================================================================


@dataclass
class TaskClassification:
    """Result of classifying a task's complexity."""

    tier: ModelTier
    reason: str
    confidence: float = 0.8


# Keywords/patterns that suggest different complexity levels
_LOCAL_PATTERNS = [
    # Classification tasks
    r"\b(classify|categorize|label|tag|sort)\b",
    r"\b(urgent|routine|noise|priority)\b.*\b(classify|determine|is this)\b",
    # Simple routing
    r"\b(route|forward|redirect|which model|can handle)\b",
    # Yes/no questions
    r"^(is|are|was|were|do|does|did|can|could|should|would|will)\s",
    # Simple extraction
    r"\b(extract|pull out|find the|get the)\s+(name|date|email|phone|number)\b",
    # Intent detection
    r"\b(intent|what does .* want|what is .* asking)\b",
]

_HAIKU_PATTERNS = [
    # Structured extraction
    r"\b(extract|parse|structure)\b.*\b(json|entities|relationships|fields)\b",
    # Summarization
    r"\b(summarize|summary|tldr|brief|digest)\b",
    # Simple generation
    r"\b(draft|write|compose)\s+(a\s+)?(short|brief|quick)\b",
    # Lookup-style questions
    r"\b(what is|who is|when did|where is|how many)\b",
    # Format conversion
    r"\b(convert|transform|reformat|translate)\b",
]

_SONNET_PATTERNS = [
    # Code generation
    r"\b(write|generate|create|build|implement)\b.{0,30}\b(code|function|class|script|module|api|endpoint)\b",
    # Complex analysis
    r"\b(analyze|compare|evaluate|review|assess)\b",
    # Creative tasks
    r"\b(creative|brainstorm|imagine|design|propose)\b",
    # Multi-step reasoning
    r"\b(step by step|first.*then|plan|strategy|approach)\b",
    # Long-form generation
    r"\b(write|draft|compose)\s+(a\s+)?(detailed|comprehensive|long|full)\b",
]

_OPUS_PATTERNS = [
    # Critical decisions
    r"\b(critical|important|sensitive|security|legal|financial)\b.*\b(decision|review|audit)\b",
    # Architecture decisions
    r"\b(architect|design|system design|trade.?off)\b",
    # Complex reasoning chains
    r"\b(reason|deduce|infer|prove|justify|explain why)\b.*\b(complex|difficult|nuanced)\b",
    # Multi-document synthesis
    r"\b(synthesize|combine|integrate|cross.?reference)\b.*\b(sources|documents|reports)\b",
]


def classify_task(
    task_description: str,
    min_tier: ModelTier = ModelTier.LOCAL,
    requires_tools: bool = False,
    context_tokens: int = 0,
) -> TaskClassification:
    """Classify a task and recommend a model tier.

    Uses pattern matching on the task description plus structural hints
    (tool requirements, context size) to pick the cheapest adequate tier.

    Args:
        task_description: Natural language description of what needs to be done
        min_tier: Minimum tier to consider (caller override)
        requires_tools: Whether the task needs tool/function calling
        context_tokens: Approximate context size in tokens

    Returns:
        TaskClassification with recommended tier and reasoning
    """
    desc_lower = task_description.lower()

    # Start with pattern matching — check from most expensive down
    matched_tier = ModelTier.LOCAL
    reason = "default: simple task"

    for pattern in _OPUS_PATTERNS:
        if re.search(pattern, desc_lower):
            matched_tier = ModelTier.OPUS
            reason = f"complex reasoning/critical task (matched: {pattern[:40]})"
            break

    if matched_tier < ModelTier.SONNET:
        for pattern in _SONNET_PATTERNS:
            if re.search(pattern, desc_lower):
                matched_tier = ModelTier.SONNET
                reason = f"complex generation/analysis (matched: {pattern[:40]})"
                break

    if matched_tier < ModelTier.HAIKU:
        for pattern in _HAIKU_PATTERNS:
            if re.search(pattern, desc_lower):
                matched_tier = ModelTier.HAIKU
                reason = f"structured extraction/summarization (matched: {pattern[:40]})"
                break

    # Structural overrides
    if requires_tools and matched_tier == ModelTier.LOCAL:
        matched_tier = ModelTier.HAIKU
        reason = "requires tool calling (local doesn't support it)"

    if context_tokens > 32_000 and matched_tier == ModelTier.LOCAL:
        matched_tier = ModelTier.HAIKU
        reason = f"context too large for local ({context_tokens} tokens > 32K limit)"

    if context_tokens > 100_000 and matched_tier < ModelTier.SONNET:
        matched_tier = ModelTier.SONNET
        reason = f"very large context ({context_tokens} tokens) benefits from larger model"

    # Apply minimum tier
    final_tier = max(matched_tier, min_tier)
    if final_tier > matched_tier:
        reason = f"caller minimum tier override ({min_tier.name}); original: {reason}"

    return TaskClassification(
        tier=final_tier,
        reason=reason,
    )


# ============================================================================
# MODEL ROUTER
# ============================================================================


class ModelRouter:
    """Routes tasks to appropriate model tiers.

    Usage:
        router = ModelRouter()

        # Automatic routing
        result = router.route("classify this email as urgent or routine")
        print(result.model_id)  # "qwen3:8b"

        # With minimum tier
        result = router.route("summarize this document", min_tier=ModelTier.HAIKU)

        # Explicit tier
        config = router.get_tier(ModelTier.SONNET)
        print(config.model_id)  # "claude-sonnet-4-6"
    """

    def __init__(self, tiers: Optional[dict[ModelTier, TierConfig]] = None):
        self._tiers = tiers or dict(DEFAULT_TIERS)

    def get_tier(self, tier: ModelTier) -> TierConfig:
        """Get configuration for a specific tier."""
        return self._tiers[tier]

    def get_model_id(self, tier: ModelTier) -> str:
        """Get the model ID for a specific tier."""
        return self._tiers[tier].model_id

    def set_model(self, tier: ModelTier, model_id: str) -> None:
        """Override the model ID for a tier."""
        self._tiers[tier].model_id = model_id
        logger.info(f"Model router: {tier.name} -> {model_id}")

    def route(
        self,
        task_description: str,
        min_tier: ModelTier = ModelTier.LOCAL,
        requires_tools: bool = False,
        context_tokens: int = 0,
    ) -> TierConfig:
        """Route a task to the appropriate model tier.

        Args:
            task_description: What needs to be done
            min_tier: Minimum tier to consider
            requires_tools: Whether the task needs tool calling
            context_tokens: Approximate context size

        Returns:
            TierConfig for the selected model
        """
        classification = classify_task(
            task_description,
            min_tier=min_tier,
            requires_tools=requires_tools,
            context_tokens=context_tokens,
        )

        config = self._tiers[classification.tier]
        logger.debug(
            f"Routed to {classification.tier.name} ({config.model_id}): {classification.reason}"
        )
        return config

    def route_with_reason(
        self,
        task_description: str,
        min_tier: ModelTier = ModelTier.LOCAL,
        requires_tools: bool = False,
        context_tokens: int = 0,
    ) -> tuple[TierConfig, TaskClassification]:
        """Route a task and return the classification reasoning."""
        classification = classify_task(
            task_description,
            min_tier=min_tier,
            requires_tools=requires_tools,
            context_tokens=context_tokens,
        )
        return self._tiers[classification.tier], classification

    def get_stats(self) -> dict:
        """Get router configuration summary."""
        return {
            tier.name: {
                "model_id": config.model_id,
                "is_local": config.is_local,
                "supports_tools": config.supports_tools,
                "max_context": config.max_context_tokens,
            }
            for tier, config in sorted(self._tiers.items())
        }


# Module-level singleton
_router: Optional[ModelRouter] = None


def get_router() -> ModelRouter:
    """Get the global model router instance."""
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router


def route(
    task_description: str,
    min_tier: ModelTier = ModelTier.LOCAL,
    requires_tools: bool = False,
    context_tokens: int = 0,
) -> TierConfig:
    """Convenience: route a task using the global router."""
    return get_router().route(
        task_description,
        min_tier=min_tier,
        requires_tools=requires_tools,
        context_tokens=context_tokens,
    )

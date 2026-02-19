"""
maasv — Cognition layer for AI agents.

Stores, structures, connects, consolidates, retrieves, decays, forgets, and learns.

Usage:
    import maasv
    from maasv.config import MaasvConfig

    config = MaasvConfig(db_path=Path("data/my.db"))
    maasv.init(config, llm=my_llm_provider, embed="ollama")

    # Now use maasv.core.store, maasv.core.wisdom, etc.
"""

import logging
import math
import threading

from maasv.config import MaasvConfig
from maasv.protocols import LLMProvider, EmbedProvider

__version__ = "0.1.1"

_log = logging.getLogger(__name__)

_config: MaasvConfig | None = None
_llm: LLMProvider | None = None
_embed: EmbedProvider | None = None
_initialized: bool = False
_init_lock = threading.Lock()

_PROVIDER_SHORTCUTS: dict[str, type] = {}


def _get_provider_class(name: str) -> type:
    """Lazy-load provider classes to avoid import cost when not used."""
    if not _PROVIDER_SHORTCUTS:
        from maasv.providers.ollama import OllamaEmbed
        _PROVIDER_SHORTCUTS["ollama"] = OllamaEmbed
    cls = _PROVIDER_SHORTCUTS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown embed provider shortcut {name!r}. "
            f"Available: {', '.join(sorted(_PROVIDER_SHORTCUTS))}"
        )
    return cls


def init(
    config: MaasvConfig,
    llm: LLMProvider,
    embed: "EmbedProvider | str",
    *,
    embed_model: str = "",
    embed_base_url: str = "",
) -> None:
    """
    Initialize maasv with configuration and providers.

    Must be called before using any maasv functionality.

    Args:
        config: Database path, model names, tuning parameters
        llm: Provider for LLM calls (entity extraction, inference, review)
        embed: Provider for text embeddings, or a shortcut string (e.g. "ollama")
        embed_model: Override the default model when using a shortcut
        embed_base_url: Override the default base URL when using a shortcut
    """
    global _config, _llm, _embed, _initialized

    # Resolve string shortcut to provider instance
    if isinstance(embed, str):
        cls = _get_provider_class(embed)
        kwargs: dict = {"dims": config.embed_dims}
        if embed_model:
            kwargs["model"] = embed_model
        if embed_base_url:
            kwargs["base_url"] = embed_base_url
        embed = cls(**kwargs)

    with _init_lock:
        _config = config
        _llm = llm
        _embed = embed
        _initialized = True

    # Validate embedding dimensions and normalization
    _validate_embed(embed, config)

    # Initialize database schema (outside lock — these do their own DB locking)
    from maasv.core.db import init_db
    init_db()

    from maasv.core.wisdom import ensure_wisdom_tables
    ensure_wisdom_tables()


def _validate_embed(embed: EmbedProvider, config: MaasvConfig) -> None:
    """Check that the provider returns vectors matching config expectations."""
    try:
        vec = embed.embed("maasv validation")
    except Exception as exc:
        raise RuntimeError(
            f"Embedding provider failed validation call: {exc}"
        ) from exc

    if len(vec) != config.embed_dims:
        raise ValueError(
            f"Embedding dimension mismatch: provider returned {len(vec)}d "
            f"but config.embed_dims={config.embed_dims}. "
            f"Either change config.embed_dims or fix the provider."
        )

    norm = math.sqrt(sum(x * x for x in vec))
    if abs(norm - 1.0) > 0.05:
        _log.warning(
            "Embedding vector is not L2-normalized (norm=%.4f). "
            "maasv thresholds assume normalized vectors — retrieval quality "
            "may degrade.",
            norm,
        )


def get_config() -> MaasvConfig:
    """Get the current config. Raises if not initialized."""
    if not _initialized or _config is None:
        raise RuntimeError("maasv not initialized. Call maasv.init() first.")
    return _config


def get_llm() -> LLMProvider:
    """Get the LLM provider. Raises if not initialized."""
    if not _initialized or _llm is None:
        raise RuntimeError("maasv not initialized. Call maasv.init() first.")
    return _llm


def get_embed() -> EmbedProvider:
    """Get the embedding provider. Raises if not initialized."""
    if not _initialized or _embed is None:
        raise RuntimeError("maasv not initialized. Call maasv.init() first.")
    return _embed

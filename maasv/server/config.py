"""Server configuration via environment variables."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """maasv-server configuration. All values from env vars or .env file."""

    # Server
    host: str = "127.0.0.1"
    port: int = 18790
    log_level: str = "info"

    # Auth
    api_key: str = ""  # empty = no auth required

    # maasv database
    db_path: str = "maasv.db"
    embed_dims: int = 1024

    # LLM provider: "anthropic" or "openai"
    llm_provider: str = "anthropic"
    llm_api_key: str = ""
    llm_model: str = "claude-haiku-4-5-20251001"

    # Embedding provider: "voyage", "openai", or "local"
    embed_provider: str = "voyage"
    embed_api_key: str = ""
    embed_model: str = "voyage-3-lite"

    # maasv tuning
    protected_categories: str = "identity,family"  # comma-separated
    stale_days: int = 30
    similarity_threshold: float = 0.95
    cross_encoder_enabled: bool = False

    model_config = {"env_prefix": "MAASV_", "env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def db_path_resolved(self) -> Path:
        return Path(self.db_path).resolve()

    @property
    def protected_categories_set(self) -> set[str]:
        return {c.strip() for c in self.protected_categories.split(",") if c.strip()}


# Singleton — import this everywhere instead of creating new Settings()
settings = Settings()

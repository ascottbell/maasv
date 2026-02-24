"""MCP server configuration via environment variables.

Reuses the MAASV_ env prefix and adds MCP-specific transport settings.
"""

from pydantic_settings import BaseSettings


class MCPSettings(BaseSettings):
    """MCP server configuration. All values from MAASV_ env vars or .env file."""

    # MCP transport
    transport: str = "stdio"  # "stdio" or "http"
    host: str = "127.0.0.1"
    port: int = 8000
    auth_token: str = ""  # required for http transport

    # maasv database
    db_path: str = "maasv.db"
    embed_dims: int = 1024

    # LLM provider: "anthropic" or "openai"
    llm_provider: str = "anthropic"
    llm_api_key: str = ""
    llm_model: str = "claude-haiku-4-5-20251001"

    # Embedding provider: "ollama" (default), "voyage", or "openai"
    embed_provider: str = "ollama"
    embed_api_key: str = ""
    embed_model: str = "qwen3-embedding:8b"
    embed_base_url: str = "http://localhost:11434"

    # maasv tuning
    protected_categories: str = "identity,family"  # comma-separated
    stale_days: int = 30
    similarity_threshold: float = 0.95
    cross_encoder_enabled: bool = False

    model_config = {"env_prefix": "MAASV_", "env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def protected_categories_set(self) -> set[str]:
        return {c.strip() for c in self.protected_categories.split(",") if c.strip()}

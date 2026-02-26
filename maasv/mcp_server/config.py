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

    # Per-operation LLM overrides (optional — falls back to llm_* above)
    extraction_llm_provider: str = ""
    extraction_llm_api_key: str = ""
    extraction_llm_model: str = ""
    inference_llm_provider: str = ""
    inference_llm_api_key: str = ""
    inference_llm_model: str = ""
    review_llm_provider: str = ""
    review_llm_api_key: str = ""
    review_llm_model: str = ""

    # Embedding provider: "ollama" (default), "voyage", or "openai"
    embed_provider: str = "ollama"
    embed_api_key: str = ""
    embed_model: str = "qwen3-embedding:8b"
    embed_base_url: str = "http://localhost:11434"

    # Cloudflare Access JWT auth (for MCP Server Portals / ChatGPT)
    cf_team: str = ""  # CF Zero Trust team name
    cf_aud: str = ""  # CF Access application audience tag

    # maasv tuning
    protected_categories: str = "identity,family"  # comma-separated
    stale_days: int = 30
    similarity_threshold: float = 0.95
    cross_encoder_enabled: bool = False

    model_config = {"env_prefix": "MAASV_", "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def cf_jwks_url(self) -> str:
        return f"https://{self.cf_team}.cloudflareaccess.com/cdn-cgi/access/certs"

    @property
    def cf_issuer(self) -> str:
        return f"https://{self.cf_team}.cloudflareaccess.com"

    @property
    def cf_enabled(self) -> bool:
        return bool(self.cf_team and self.cf_aud)

    @property
    def protected_categories_set(self) -> set[str]:
        return {c.strip() for c in self.protected_categories.split(",") if c.strip()}

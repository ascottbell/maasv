"""maasv-server: HTTP API for the maasv cognition layer."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI

from maasv.server.auth import require_auth
from maasv.server.config import settings

logger = logging.getLogger("maasv_server")


def _init_maasv():
    """Initialize maasv with configured providers."""
    import maasv
    from maasv.config import MaasvConfig
    from maasv.server.providers import create_llm, create_embed

    config = MaasvConfig(
        db_path=Path(settings.db_path).resolve(),
        embed_dims=settings.embed_dims,
        extraction_model=settings.llm_model,
        protected_categories=settings.protected_categories_set,
        stale_days=settings.stale_days,
        similarity_threshold=settings.similarity_threshold,
        cross_encoder_enabled=settings.cross_encoder_enabled,
    )

    llm = create_llm(settings.llm_provider, settings.llm_api_key, settings.llm_model)
    embed = create_embed(
        settings.embed_provider,
        api_key=settings.embed_api_key,
        model=settings.embed_model,
        base_url=getattr(settings, "embed_base_url", ""),
        dims=settings.embed_dims,
    )

    maasv.init(config=config, llm=llm, embed=embed)
    logger.info(
        "maasv initialized: db=%s, embed_dims=%d, llm=%s, embed=%s/%s",
        config.db_path, config.embed_dims, settings.llm_provider,
        settings.embed_provider, settings.embed_model,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_maasv()
    logger.info("maasv-server ready on %s:%d", settings.host, settings.port)
    yield
    logger.info("maasv-server shutting down")


app = FastAPI(
    title="maasv-server",
    description="HTTP API for the maasv cognition layer",
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/docs" if not settings.api_key else None,
    openapi_url="/openapi.json" if not settings.api_key else None,
)


# --- Register routers ---

from maasv.server.routers import memory, extraction, graph, wisdom, health  # noqa: E402

# Protected routers — auth enforced via dependency injection
app.include_router(
    memory.router, prefix="/v1/memory", tags=["memory"],
    dependencies=[Depends(require_auth)],
)
app.include_router(
    extraction.router, prefix="/v1", tags=["extraction"],
    dependencies=[Depends(require_auth)],
)
app.include_router(
    graph.router, prefix="/v1/graph", tags=["graph"],
    dependencies=[Depends(require_auth)],
)
app.include_router(
    wisdom.router, prefix="/v1/wisdom", tags=["wisdom"],
    dependencies=[Depends(require_auth)],
)

# Health router — /health is public, /stats is protected at the route level
app.include_router(health.router, prefix="/v1", tags=["health"])


def run():
    """Entry point for `maasv-server` CLI command."""
    import uvicorn

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    uvicorn.run(
        "maasv.server.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    run()

"""API key authentication via FastAPI dependency injection."""

import hmac

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from maasv.server.config import settings

_api_key_header = APIKeyHeader(name="X-Maasv-Key", auto_error=False)


async def require_auth(
    api_key: str | None = Security(_api_key_header),
) -> None:
    """Enforce API key authentication when MAASV_API_KEY is configured.

    Attach as a dependency to any route or router that should be protected.
    When MAASV_API_KEY is empty, all requests are allowed (local dev mode).
    """
    if not settings.api_key:
        return

    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-Maasv-Key header")

    if not hmac.compare_digest(api_key.encode(), settings.api_key.encode()):
        raise HTTPException(status_code=403, detail="Invalid API key")
